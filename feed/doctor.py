from __future__ import annotations
import shutil
import sys
import tomllib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from sqlalchemy import func, select

from feed.config import Config
from feed.db import create_all, make_engine, make_session_factory
from feed.models import Item, Story
from feed.providers._dotenv import load_dotenv
from feed.providers.base import ProviderError

Status = str  # "OK" | "WARN" | "FAIL"


@dataclass
class Check:
    section: str
    name: str
    status: Status
    detail: str = ""
    hint: str = ""


@dataclass
class DoctorReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(c.status == "FAIL" for c in self.checks)

    def add(self, section: str, name: str, status: Status, detail: str = "",
           hint: str = "") -> None:
        self.checks.append(Check(section, name, status, detail, hint))

    def print(self, file: IO[str] | None = None) -> None:
        # `file` resolves to sys.stdout at CALL time, not def time -- a
        # default of `= sys.stdout` would bind the real stdout object once,
        # at import, which silently stops going to whatever stdout is
        # current (e.g. pytest's capsys-substituted stream) at call time.
        file = file if file is not None else sys.stdout
        symbol = {"OK": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]"}
        section = None
        for c in self.checks:
            if c.section != section:
                section = c.section
                print(f"\n{section}", file=file)
            line = f"  {symbol.get(c.status, c.status):<7}{c.name}"
            if c.detail:
                line += f" -- {c.detail}"
            print(line, file=file)
            if c.hint and c.status != "OK":
                print(f"           -> {c.hint}", file=file)

        fails = sum(1 for c in self.checks if c.status == "FAIL")
        warns = sum(1 for c in self.checks if c.status == "WARN")
        print(file=file)
        if fails:
            print(f"{fails} FAILURE(S), {warns} warning(s) -- the pipeline will "
                 "likely NOT complete cleanly. See -> hints above.", file=file)
        elif warns:
            print(f"0 failures, {warns} warning(s) -- the pipeline should run, "
                 "but check the warnings above.", file=file)
        else:
            print("All checks passed.", file=file)


def _check_python(report: DoctorReport) -> None:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    expected = None
    pyproject = Path("pyproject.toml")
    if pyproject.exists():
        try:
            raw = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            expected = raw.get("project", {}).get("requires-python")
        except (tomllib.TOMLDecodeError, OSError):
            pass
    if sys.version_info[:2] != (3, 14):
        report.add("Python", "version", "FAIL", f"{version} (expected 3.14.x)",
                   hint="run with .venv/Scripts/python.exe, not a bare `python`")
    else:
        detail = version if expected is None else f"{version} (pyproject wants {expected})"
        report.add("Python", "version", "OK", detail)

    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv:
        report.add("Python", "virtualenv", "OK", sys.prefix)
    else:
        report.add("Python", "virtualenv", "WARN", f"not running inside a venv ({sys.prefix})",
                   hint="use .venv/Scripts/python.exe -m feed ...")


def _check_database(report: DoctorReport, cfg: Config) -> None:
    try:
        engine = make_engine(cfg.database.url)
        create_all(engine)
        factory = make_session_factory(engine)
        with factory() as s:
            story_count = s.scalar(select(func.count()).select_from(Story)) or 0
            item_count = s.scalar(select(func.count()).select_from(Item)) or 0
        report.add("Database", cfg.database.url, "OK",
                   f"reachable, {story_count} stories / {item_count} items")
    except Exception as exc:
        report.add("Database", cfg.database.url, "FAIL", f"{type(exc).__name__}: {exc}",
                   hint="run `feed init` to create it, or check [database].url in feed.toml")


def _check_env_file(report: DoctorReport, cfg: Config) -> None:
    env_path = Path(".env")
    if not env_path.exists():
        report.add(".env", "file", "WARN", "no .env file at repo root",
                   hint="keys can still come from the real environment; "
                        "create .env if they don't")
    else:
        report.add(".env", "file", "OK", str(env_path.resolve()))

    load_dotenv()  # same loader every provider uses -- populates os.environ
    import os
    for entry in cfg.providers.bulk:
        present = bool(os.environ.get(entry.env_var))
        if present:
            report.add(".env", entry.env_var, "OK", f"set (used by '{entry.name}')")
        elif entry.enabled:
            report.add(".env", entry.env_var, "FAIL",
                       f"not set, but '{entry.name}' is enabled in feed.toml",
                       hint=f"add {entry.env_var}=... to .env, or disable "
                            f"'{entry.name}' in feed.toml")
        else:
            report.add(".env", entry.env_var, "WARN",
                       f"not set ('{entry.name}' is disabled, so this is harmless)")


def _check_providers(report: DoctorReport, cfg: Config, *, probe: bool) -> None:
    # Deferred import: avoids feed.doctor <-> feed.cli becoming a circular
    # import at module load time (cli.py imports run_doctor). Reuses the
    # exact same provider construction `feed providers` uses, so "doctor"
    # and "providers" never drift apart on what a live probe means.
    from feed.cli import _build_bulk_provider

    any_reachable = False
    any_enabled = False
    for entry in cfg.providers.bulk:
        if not entry.enabled:
            report.add("LLM providers", entry.name, "WARN", "disabled in feed.toml")
            continue
        any_enabled = True
        provider = _build_bulk_provider(entry, max_retries=cfg.providers.max_retries,
                                        backoff_base=cfg.providers.backoff_base)
        health = provider.health()
        if not health.healthy:
            report.add("LLM providers", entry.name, "FAIL", health.detail,
                       hint="check the API key in .env")
            continue
        if not probe:
            report.add("LLM providers", entry.name, "OK", "key present (not probed)")
            any_reachable = True
            continue
        start = time.monotonic()
        try:
            provider.complete("Reply with only the single word: OK.")
            latency = (time.monotonic() - start) * 1000
            report.add("LLM providers", entry.name, "OK", f"reachable, {latency:.0f}ms")
            any_reachable = True
        except ProviderError as exc:
            report.add("LLM providers", entry.name, "WARN", f"unreachable right now: {exc}",
                       hint="the failover chain will skip this provider until it recovers")

    if any_enabled and not any_reachable:
        report.add("LLM providers", "TIER 1 (bulk)", "FAIL",
                   "every enabled bulk provider is unreachable -- enrich will produce no summaries",
                   hint="check network access and API keys; run `feed providers` for detail")

    claude_binary = shutil.which("claude")
    if claude_binary:
        report.add("LLM providers", "claude-code (Tier 2)", "OK", claude_binary)
    else:
        report.add("LLM providers", "claude-code (Tier 2)", "WARN",
                   "`claude` not found on PATH -- Tier 2 analysis will fail per-story "
                   "(isolated, does not block Tier 1 or publish)",
                   hint="install the Claude Code CLI and ensure it is on PATH")


def _check_gh(report: DoctorReport, almanac_repo: str) -> None:
    from feed import pipeline as _pl
    try:
        proc = _pl._run_gh(["auth", "status"], cwd=Path.cwd(), timeout=15.0)
    except FileNotFoundError:
        report.add("GitHub / almanac", "gh CLI", "FAIL", "`gh` not found on PATH",
                   hint="install GitHub CLI: https://cli.github.com")
        return
    except Exception as exc:
        report.add("GitHub / almanac", "gh CLI", "FAIL", f"{type(exc).__name__}: {exc}")
        return

    text = (proc.stdout + proc.stderr)
    if proc.returncode != 0 or "Logged in" not in text:
        report.add("GitHub / almanac", "gh auth", "FAIL", "not authenticated",
                   hint="run `gh auth login`")
        return
    scopes_line = next((l for l in text.splitlines() if "Token scopes" in l), "")
    missing = [s for s in ("repo", "workflow") if f"'{s}'" not in scopes_line]
    if missing:
        report.add("GitHub / almanac", "gh auth", "WARN",
                   f"authenticated, but missing scope(s): {missing}",
                   hint=f"run `gh auth refresh -s {','.join(missing)}`")
    else:
        report.add("GitHub / almanac", "gh auth", "OK", "authenticated with repo+workflow scope")

    try:
        proc = _pl._run_gh(["repo", "view", almanac_repo, "--json", "viewerPermission"],
                          cwd=Path.cwd(), timeout=15.0)
    except Exception as exc:
        report.add("GitHub / almanac", almanac_repo, "FAIL", f"{type(exc).__name__}: {exc}")
        return
    if proc.returncode != 0:
        report.add("GitHub / almanac", almanac_repo, "FAIL",
                   f"repo not reachable: {proc.stderr.strip()[:200]}",
                   hint="check the repo exists and gh is authenticated as the right account")
        return
    import json
    try:
        perm = json.loads(proc.stdout).get("viewerPermission", "")
    except json.JSONDecodeError:
        perm = ""
    if perm in ("ADMIN", "WRITE", "MAINTAIN"):
        report.add("GitHub / almanac", almanac_repo, "OK", f"writable ({perm})")
    else:
        report.add("GitHub / almanac", almanac_repo, "FAIL",
                   f"not writable (permission={perm or 'unknown'})",
                   hint="the almanac push will fail; check repo access for this gh account")


def _check_disk_space(report: DoctorReport) -> None:
    try:
        usage = shutil.disk_usage(Path.cwd())
    except OSError as exc:
        report.add("Disk", str(Path.cwd()), "WARN", f"could not check: {exc}")
        return
    free_gb = usage.free / (1024 ** 3)
    detail = f"{free_gb:.1f} GB free"
    if free_gb < 1:
        report.add("Disk", str(Path.cwd()), "FAIL", detail,
                   hint="free up space -- the database and bundle both need headroom")
    elif free_gb < 5:
        report.add("Disk", str(Path.cwd()), "WARN", detail)
    else:
        report.add("Disk", str(Path.cwd()), "OK", detail)


def run_doctor(cfg: Config, *, probe_providers: bool = True,
              almanac_repo: str | None = None) -> DoctorReport:
    """Runs every preflight check and returns a DoctorReport. Each check is
    independent and best-effort -- one check raising is caught locally
    (where it plausibly can, e.g. the database connection) so a single
    broken subsystem still lets every other check report its own result,
    which is the entire point of a diagnostic command.
    """
    from feed.pipeline import DEFAULT_ALMANAC_REPO
    almanac_repo = almanac_repo or DEFAULT_ALMANAC_REPO

    report = DoctorReport()
    _check_python(report)
    _check_database(report, cfg)
    _check_env_file(report, cfg)
    _check_providers(report, cfg, probe=probe_providers)
    _check_gh(report, almanac_repo)
    _check_disk_space(report)
    return report
