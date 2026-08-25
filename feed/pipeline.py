from __future__ import annotations
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from feed.lock import LockHeld, PipelineLock

log = logging.getLogger("feed.pipeline")

# The almanac repo is a fixed collaborator of this project (see
# web/src/lib/bundle.ts and web/.env.example, which hardcode the same
# raw.githubusercontent.com URL) -- overridable via --almanac-repo purely
# so tests and a future fork don't have to touch this file.
DEFAULT_ALMANAC_REPO = "sbalasubramanian14/observatory-almanac"
DEFAULT_ALMANAC_DIR = Path(".cache") / "observatory-almanac"
DEFAULT_LOGS_DIR = Path("logs")
DEFAULT_KEEP_LOGS = 30
DEFAULT_STAGE_TIMEOUT = 3600.0  # seconds; `run` on a big backlog is the long pole

# Explicit allowlist of what a publish actually writes (feed/stages/publish.py),
# mirrored one-for-one into the almanac clone. NOT "everything in public/",
# and NOT a blind directory mirror of the almanac repo itself -- the repo
# also holds a README.md that `feed publish` never writes and this pipeline
# must never touch, let alone delete.
BUNDLE_DIRS = ("embeddings", "feed", "story")
BUNDLE_FILES = ("manifest.json", "sources.json")

# Exit code contract (documented in PIPELINE-CLI.md -- this is what Task
# Scheduler's "Last Run Result" reflects):
EXIT_OK = 0              # full success; site updated (or nothing new to publish)
EXIT_LOCKED = 1           # refused to start: another run already in progress
EXIT_SITE_NOT_UPDATED = 2  # publish or the almanac push failed -- the live site
                          # was NOT updated this run. The one code worth paging on.
EXIT_DEGRADED = 3         # site WAS updated, but sync/run/enrich reported errors
                          # along the way -- worth a look, not an emergency.
EXIT_UNEXPECTED = 99      # bug in the orchestrator itself (safety net)


@dataclass
class StageOutcome:
    name: str
    ok: bool
    elapsed: float
    fatal_if_failed: bool
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    note: str = ""


@dataclass
class PipelineResult:
    exit_code: int = EXIT_OK
    stages: list[StageOutcome] = field(default_factory=list)
    almanac_ok: bool | None = None
    almanac_detail: str = ""
    log_path: Path | None = None
    elapsed: float = 0.0


# ---------------------------------------------------------------------------
# Subprocess seams -- tests monkeypatch these three names rather than let a
# unit test spawn a real `feed` process, real `git`, or real `gh`. Mirrors
# the existing convention in this codebase (see feed.providers.claude_code
# ._run_cli, feed.providers.gemini._post, etc.) of one call-site-adjacent
# seam per external-process boundary.
# ---------------------------------------------------------------------------

def _run_feed_command(python: str, config: Path | None, args: list[str], *,
                      cwd: Path, timeout: float) -> subprocess.CompletedProcess:
    cmd = [python, "-m", "feed"]
    if config is not None:
        cmd += ["--config", str(config)]
    cmd += args
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def _run_git(args: list[str], *, cwd: Path, timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout)


def _run_gh(args: list[str], *, cwd: Path, timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout)


# ---------------------------------------------------------------------------
# Logging / console
# ---------------------------------------------------------------------------

class _Reporter:
    """Writes a curated, human-scale narrative to both the console and the
    log file, while the RAW stdout/stderr of each stage subprocess goes to
    the log file only. Python's own `logging` module already defaults to
    stderr (see feed.cli.main's basicConfig), and every `feed` subcommand
    already prints a one-line, human-readable summary to stdout (spec:
    "collect: new=5 dupes=2 ...") -- so echoing just stdout to the console
    IS the "glance-able" summary the brief asks for, without re-implementing
    the parsing of those lines here.
    """

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._fh = log_path.open("a", encoding="utf-8")

    def close(self) -> None:
        self._fh.close()

    def say(self, msg: str = "") -> None:
        print(msg)
        self._fh.write(msg + "\n")
        self._fh.flush()

    def log_only(self, msg: str) -> None:
        self._fh.write(msg + "\n")
        self._fh.flush()


def _prune_old_logs(logs_dir: Path, keep: int) -> None:
    logs = sorted(logs_dir.glob("pipeline_*.log"))
    excess = len(logs) - keep
    for old in logs[:max(excess, 0)]:
        try:
            old.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Stage sequence
# ---------------------------------------------------------------------------

@dataclass
class _StageSpec:
    name: str
    args: list[str]
    fatal_if_failed: bool


def _stage_specs(*, catalogue: Path | None, out_dir: Path | None,
                 days: int | None = None) -> list[_StageSpec]:
    sync_args = ["sources", "sync"]
    if catalogue is not None:
        sync_args += ["--catalogue", str(catalogue)]
    publish_args = ["publish"]
    if out_dir is not None:
        publish_args += ["--out", str(out_dir)]
    # `observatory.bat N` lands here. Only publish gets it: `days` narrows
    # what the BUNDLE contains, which is a separate question from how far
    # back collect() fetches ([collect].max_backfill_days). Widening the
    # window later needs no re-collection -- every story is still in the
    # db, only the published subset changed.
    if days is not None:
        publish_args += ["--days", str(days)]
    # Rank gets the SAME window as publish. "Top 50" means the top of what
    # a reader can actually see, so ranking a wider (or narrower) set than
    # the bundle carries would put stories on the Top 50 page that are not
    # in the feed at all.
    rank_args = ["rank"]
    if days is not None:
        rank_args += ["--days", str(days)]
    return [
        # Non-fatal (spec: source failures are normal, already visible on the
        # health page). A sync failure just means sources didn't change this
        # run; the existing source list in the db is used as-is.
        _StageSpec("sources sync", sync_args, fatal_if_failed=False),
        # Non-fatal: collect() already isolates a single bad source
        # internally and never raises for that (feed.stages.collect). If
        # `run` itself dies (e.g. a DB error), whatever normalize/embed/
        # cluster/score already committed in earlier drain() rounds this
        # process is still on disk -- publish must still run against it.
        _StageSpec("run (collect->score)", ["run"], fatal_if_failed=False),
        # Non-fatal: enrich() commits per-story (see feed.stages.enrich); a
        # crash partway through does not undo stories already enriched, and
        # "publishing failing should not lose the enrichment work" cuts the
        # other way too -- enrichment failing must not block publish.
        _StageSpec("enrich", ["enrich"], fatal_if_failed=False),
        # Non-fatal, and deliberately AFTER enrich: rank judges importance
        # from the Tier 1 headline and summary, so running it earlier would
        # ask Claude Code to rank bare, unsummarised titles. Non-fatal
        # because Claude Code is a local CLI with no SLA -- losing today's
        # Top 50 leaves the feed degraded, and rank_top() explicitly keeps
        # the previous ranking on failure rather than blanking the page.
        # Losing publish, by contrast, kills the site for the day.
        _StageSpec("rank", rank_args, fatal_if_failed=False),
        # FATAL: this is the one stage that produces the bundle. If it
        # fails there is nothing new (or nothing valid) to push, so the
        # almanac step is skipped entirely rather than pushing stale/absent
        # content.
        _StageSpec("publish", publish_args, fatal_if_failed=True),
    ]


def _run_stage(spec: _StageSpec, *, python: str, config: Path | None, cwd: Path,
               timeout: float, reporter: _Reporter, index: int, total: int) -> StageOutcome:
    reporter.say(f"[{index}/{total}] {spec.name} ...")
    start = time.monotonic()
    try:
        proc = _run_feed_command(python, config, spec.args, cwd=cwd, timeout=timeout)
        elapsed = time.monotonic() - start
        ok = proc.returncode == 0
        outcome = StageOutcome(spec.name, ok, elapsed, spec.fatal_if_failed,
                               returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        outcome = StageOutcome(spec.name, False, elapsed, spec.fatal_if_failed,
                               note=f"timed out after {timeout:.0f}s")
    except OSError as exc:
        elapsed = time.monotonic() - start
        outcome = StageOutcome(spec.name, False, elapsed, spec.fatal_if_failed,
                               note=f"{type(exc).__name__}: {exc}")

    for line in outcome.stdout.splitlines():
        if line.strip():
            reporter.say(f"    {line}")
    if outcome.note:
        reporter.say(f"    {outcome.note}")
    reporter.log_only(f"--- stdout ({spec.name}) ---\n{outcome.stdout}")
    reporter.log_only(f"--- stderr ({spec.name}) ---\n{outcome.stderr}")

    # feed.cli.main() sets up logging at INFO (or DEBUG with -v), so stderr
    # is full of routine "INFO httpx: ..." lines on a normal run -- counting
    # those as "warnings" would be exactly the misleading noise this
    # console output is supposed to avoid. Only WARNING/ERROR/CRITICAL
    # (feed.cli's own format is "%(levelname)s %(name)s: ...") are worth a
    # human's attention at a glance; everything else is still in the log.
    warn_lines = [l for l in outcome.stderr.splitlines()
                 if l.startswith(("WARNING ", "ERROR ", "CRITICAL "))]
    status = "OK" if outcome.ok else "FAILED"
    tail = f", {len(warn_lines)} warning/error line(s) -- see log" if warn_lines else ""
    reporter.say(f"    -> {status}  ({elapsed:.1f}s){tail}")
    reporter.say("")
    return outcome


# ---------------------------------------------------------------------------
# Almanac push
# ---------------------------------------------------------------------------

def _mirror_dir(src: Path, dst: Path) -> None:
    """Make `dst` contain exactly the files `src` contains (recursively),
    without touching anything outside `dst` -- a scoped, pure-Python
    equivalent of `robocopy /MIR src dst`, chosen over shelling out to
    robocopy so this is directly unit-testable and portable.
    """
    dst.mkdir(parents=True, exist_ok=True)
    src_files = {p.relative_to(src) for p in src.rglob("*") if p.is_file()}
    dst_files = {p.relative_to(dst) for p in dst.rglob("*") if p.is_file()}
    for rel in src_files:
        d = dst / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, d)
    for rel in dst_files - src_files:
        (dst / rel).unlink()
    for dirpath, dirnames, filenames in os.walk(dst, topdown=False):
        if Path(dirpath) == dst:
            continue
        if not dirnames and not filenames:
            try:
                Path(dirpath).rmdir()
            except OSError:
                pass


def _push_to_almanac(*, repo: str, almanac_dir: Path, bundle_dir: Path,
                     timeout: float, reporter: _Reporter) -> tuple[bool, str]:
    try:
        if not (almanac_dir / ".git").is_dir():
            almanac_dir.parent.mkdir(parents=True, exist_ok=True)
            reporter.say(f"    cloning {repo} -> {almanac_dir}")
            proc = _run_gh(["repo", "clone", repo, str(almanac_dir)],
                           cwd=almanac_dir.parent, timeout=timeout)
            if proc.returncode != 0:
                return False, f"gh repo clone failed: {proc.stderr.strip()[:500]}"
        else:
            proc = _run_git(["fetch", "origin"], cwd=almanac_dir, timeout=timeout)
            if proc.returncode != 0:
                return False, f"git fetch failed: {proc.stderr.strip()[:500]}"
            proc = _run_git(["reset", "--hard", "origin/HEAD"], cwd=almanac_dir, timeout=timeout)
            if proc.returncode != 0:
                return False, f"git reset failed: {proc.stderr.strip()[:500]}"
            proc = _run_git(["clean", "-fd"], cwd=almanac_dir, timeout=timeout)
            if proc.returncode != 0:
                return False, f"git clean failed: {proc.stderr.strip()[:500]}"

        if not bundle_dir.exists():
            return False, f"bundle directory {bundle_dir} does not exist -- publish did not run?"

        for name in BUNDLE_DIRS:
            src = bundle_dir / name
            if src.exists():
                _mirror_dir(src, almanac_dir / name)
        for name in BUNDLE_FILES:
            src = bundle_dir / name
            if src.exists():
                shutil.copy2(src, almanac_dir / name)

        status = _run_git(["status", "--porcelain"], cwd=almanac_dir, timeout=30)
        changed = [line[3:].strip() for line in status.stdout.splitlines() if line.strip()]

        # Belt-and-braces (spec: "the almanac push must only ever include
        # bundle files. Never commit .env."). The mirror above only ever
        # touches BUNDLE_DIRS/BUNDLE_FILES, so this should never trip --
        # it exists as a hard stop in case that ever changes.
        unsafe = [p for p in changed if ".env" in Path(p).name.lower()]
        if unsafe:
            return False, f"refusing to push: unexpected path(s) staged: {unsafe}"

        if not changed:
            return True, "nothing to publish (bundle unchanged since last push)"

        add = _run_git(["add", "-A", "--", *BUNDLE_DIRS, *BUNDLE_FILES],
                       cwd=almanac_dir, timeout=30)
        if add.returncode != 0:
            return False, f"git add failed: {add.stderr.strip()[:500]}"

        story_count = "?"
        manifest = almanac_dir / "manifest.json"
        if manifest.exists():
            try:
                story_count = json.loads(manifest.read_text(encoding="utf-8")).get(
                    "story_count", "?")
            except (json.JSONDecodeError, OSError):
                pass

        msg = (f"Publish bundle: {datetime.now().strftime('%Y-%m-%d %H:%M')} "
              f"({story_count} stories, {len(changed)} file(s) changed)")
        commit = _run_git(["commit", "-m", msg], cwd=almanac_dir, timeout=30)
        if commit.returncode != 0:
            return False, f"git commit failed: {commit.stderr.strip()[:500]}"

        push = _run_git(["push", "origin", "HEAD:main"], cwd=almanac_dir, timeout=timeout)
        if push.returncode != 0:
            # The commit exists locally but never left this machine -- not a
            # "half-pushed" state (git push is all-or-nothing for one ref),
            # just an unpushed one. Next successful run pushes it too, on
            # top of whatever changed since. Report loudly regardless: the
            # site was not updated this run.
            return False, (f"git push failed (site NOT updated; commit exists "
                          f"locally, unpushed): {push.stderr.strip()[:500]}")
        return True, f"pushed {len(changed)} changed file(s), {story_count} stories"
    except subprocess.TimeoutExpired as exc:
        return False, f"timed out: {exc}"
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    *,
    config: Path | None = None,
    cwd: Path | None = None,
    python: str | None = None,
    catalogue: Path | None = None,
    out_dir: Path | None = None,
    logs_dir: Path | None = None,
    keep_logs: int = DEFAULT_KEEP_LOGS,
    lock_path: Path | None = None,
    almanac_dir: Path | None = None,
    almanac_repo: str = DEFAULT_ALMANAC_REPO,
    skip_almanac_push: bool = False,
    stage_timeout: float = DEFAULT_STAGE_TIMEOUT,
    days: int | None = None,
) -> PipelineResult:
    """Run the full one-click pipeline: sources sync -> run -> enrich ->
    publish -> push the bundle to the almanac repo. See PIPELINE-CLI.md for
    the failure policy this implements; the short version: only a failed
    publish or a failed almanac push stops the site from updating (exit
    EXIT_SITE_NOT_UPDATED) -- every earlier stage is best-effort (exit
    EXIT_DEGRADED if any of them failed but the site still updated).
    """
    cwd = Path(cwd) if cwd is not None else Path.cwd()
    python = python or sys.executable
    logs_dir = Path(logs_dir) if logs_dir is not None else (cwd / DEFAULT_LOGS_DIR)
    logs_dir.mkdir(parents=True, exist_ok=True)
    lock_path = Path(lock_path) if lock_path is not None else (logs_dir / "pipeline.lock")
    almanac_dir = Path(almanac_dir) if almanac_dir is not None else (cwd / DEFAULT_ALMANAC_DIR)
    out_dir = Path(out_dir) if out_dir is not None else None
    bundle_dir = out_dir if out_dir is not None else (cwd / "public")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"pipeline_{timestamp}.log"
    reporter = _Reporter(log_path)
    wall_start = time.monotonic()
    result = PipelineResult(log_path=log_path)

    try:
        reporter.say(f"=== Observatory pipeline -- {datetime.now().isoformat(timespec='seconds')} ===")

        lock = PipelineLock(lock_path)
        try:
            stale = lock.acquire()
        except LockHeld as exc:
            reporter.say(f"REFUSED: {exc}")
            reporter.say("Another run is already in progress -- exiting without touching the database.")
            result.exit_code = EXIT_LOCKED
            return result
        if stale is not None:
            reporter.say(f"(recovered a stale lock left by pid={stale.pid}, "
                         f"started {stale.started_at} -- that process is no longer running)")

        try:
            specs = _stage_specs(catalogue=catalogue, out_dir=out_dir, days=days)
            degraded = False
            publish_ok = False
            for i, spec in enumerate(specs, start=1):
                outcome = _run_stage(spec, python=python, config=config, cwd=cwd,
                                     timeout=stage_timeout, reporter=reporter,
                                     index=i, total=len(specs))
                result.stages.append(outcome)
                if not outcome.ok:
                    if spec.fatal_if_failed:
                        reporter.say(f"STOPPING: {spec.name} failed and nothing downstream "
                                     "of it can proceed (no valid bundle to push).")
                        result.exit_code = EXIT_SITE_NOT_UPDATED
                        return result
                    degraded = True
                elif spec.name == "publish":
                    publish_ok = True

            if not publish_ok:
                # Should be unreachable (publish is fatal_if_failed=True,
                # so a failure there already returned above) -- guarded
                # anyway so a future stage-list edit can't silently push a
                # stale bundle.
                result.exit_code = EXIT_SITE_NOT_UPDATED
                return result

            if skip_almanac_push:
                reporter.say("almanac push: SKIPPED (--skip-almanac-push)")
                result.exit_code = EXIT_DEGRADED if degraded else EXIT_OK
                return result

            reporter.say(f"[{len(specs) + 1}/{len(specs) + 1}] push to {almanac_repo} ...")
            start = time.monotonic()
            almanac_ok, detail = _push_to_almanac(
                repo=almanac_repo, almanac_dir=almanac_dir, bundle_dir=bundle_dir,
                timeout=stage_timeout, reporter=reporter,
            )
            elapsed = time.monotonic() - start
            result.almanac_ok = almanac_ok
            result.almanac_detail = detail
            reporter.say(f"    {detail}")
            reporter.say(f"    -> {'OK' if almanac_ok else 'FAILED'}  ({elapsed:.1f}s)")
            reporter.say("")

            if not almanac_ok:
                reporter.say("SITE NOT UPDATED this run -- see the failure above.")
                result.exit_code = EXIT_SITE_NOT_UPDATED
            else:
                result.exit_code = EXIT_DEGRADED if degraded else EXIT_OK
            return result
        finally:
            lock.release()
    except Exception:
        import traceback
        reporter.log_only(traceback.format_exc())
        reporter.say("UNEXPECTED ERROR in the pipeline orchestrator itself -- see log for the traceback.")
        result.exit_code = EXIT_UNEXPECTED
        return result
    finally:
        result.elapsed = time.monotonic() - wall_start
        outcome_word = {
            EXIT_OK: "SUCCESS", EXIT_LOCKED: "REFUSED (locked)",
            EXIT_SITE_NOT_UPDATED: "FAILED (site not updated)",
            EXIT_DEGRADED: "DEGRADED (published with earlier errors)",
            EXIT_UNEXPECTED: "ERROR (orchestrator bug)",
        }.get(result.exit_code, "UNKNOWN")
        reporter.say(f"=== {outcome_word} -- total {result.elapsed:.1f}s -- "
                     f"exit code {result.exit_code} -- log: {log_path} ===")
        reporter.close()
        _prune_old_logs(logs_dir, keep_logs)

    return result
