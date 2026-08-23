from __future__ import annotations
import json
import subprocess
from pathlib import Path
import pytest
from feed.lock import PipelineLock
from feed.pipeline import (
    EXIT_DEGRADED, EXIT_LOCKED, EXIT_OK, EXIT_SITE_NOT_UPDATED,
    _mirror_dir, run_pipeline,
)


def _cp(args, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr=stderr)


def _ok_feed_command(python, config, args, *, cwd, timeout):
    stage = args[0]
    stdouts = {
        "sources": "sources sync: added=0 updated=1 unchanged=5 disabled=0 deleted=0",
        "run": "collect:   new=3 dupes=1 source_errors=0\nscore:     ok=3 failed=0 rounds=1",
        "enrich": "tier1: ok=2 failed=0\ntier2: ok=1 failed=0 degraded=0",
        "publish": "published 5 stories across 1 page(s) to public (pruned 0 stale file(s))",
    }
    return _cp(args, 0, stdouts.get(stage, ""), "")


def _ok_gh_git(bundle_dir: Path):
    """Fakes a fresh `gh repo clone` + `git add/commit/push` all succeeding,
    for the given bundle_dir. Returns (run_gh, run_git) callables.
    """
    def run_gh(args, *, cwd, timeout=120.0):
        if args[:2] == ["repo", "clone"]:
            dest = Path(args[3])
            (dest / ".git").mkdir(parents=True)
            return _cp(args, 0)
        raise AssertionError(f"unexpected gh args {args}")

    def run_git(args, *, cwd, timeout=60.0):
        if args[0] == "status":
            return _cp(args, 0, stdout=" M manifest.json\n?? story/x.json\n")
        return _cp(args, 0)

    return run_gh, run_git


def _write_bundle(bundle_dir: Path, n_stories: int = 5) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "manifest.json").write_text(
        json.dumps({"story_count": n_stories}), encoding="utf-8")
    (bundle_dir / "sources.json").write_text("{}", encoding="utf-8")
    story_dir = bundle_dir / "story"
    story_dir.mkdir(exist_ok=True)
    (story_dir / "abc123.json").write_text("{}", encoding="utf-8")


def test_happy_path_runs_every_stage_and_pushes(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "public"
    _write_bundle(bundle_dir)
    monkeypatch.setattr("feed.pipeline._run_feed_command", _ok_feed_command)
    run_gh, run_git = _ok_gh_git(bundle_dir)
    monkeypatch.setattr("feed.pipeline._run_gh", run_gh)
    monkeypatch.setattr("feed.pipeline._run_git", run_git)

    result = run_pipeline(
        cwd=tmp_path, out_dir=bundle_dir,
        logs_dir=tmp_path / "logs", almanac_dir=tmp_path / "almanac",
    )

    assert result.exit_code == EXIT_OK
    assert [s.name for s in result.stages] == [
        "sources sync", "run (collect->score)", "enrich", "publish",
    ]
    assert all(s.ok for s in result.stages)
    assert result.almanac_ok is True
    assert result.log_path.exists()
    log_text = result.log_path.read_text(encoding="utf-8")
    assert "sources sync" in log_text
    assert "SUCCESS" in log_text


def test_non_fatal_stage_failure_still_publishes_and_degrades(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "public"
    _write_bundle(bundle_dir)

    def flaky_feed_command(python, config, args, *, cwd, timeout):
        if args[0] == "sources":
            return _cp(args, 2, "", "sources sync failed: catalogue missing")
        return _ok_feed_command(python, config, args, cwd=cwd, timeout=timeout)

    monkeypatch.setattr("feed.pipeline._run_feed_command", flaky_feed_command)
    run_gh, run_git = _ok_gh_git(bundle_dir)
    monkeypatch.setattr("feed.pipeline._run_gh", run_gh)
    monkeypatch.setattr("feed.pipeline._run_git", run_git)

    result = run_pipeline(
        cwd=tmp_path, out_dir=bundle_dir,
        logs_dir=tmp_path / "logs", almanac_dir=tmp_path / "almanac",
    )

    assert result.exit_code == EXIT_DEGRADED
    sync_outcome = result.stages[0]
    assert sync_outcome.name == "sources sync" and not sync_outcome.ok
    assert result.stages[-1].name == "publish" and result.stages[-1].ok
    assert result.almanac_ok is True  # site WAS updated despite the earlier failure


def test_collect_run_failure_does_not_prevent_publish(tmp_path, monkeypatch):
    """Explicit spec requirement: 'Collection failing should not prevent
    publishing what is already scored.'"""
    bundle_dir = tmp_path / "public"
    _write_bundle(bundle_dir)

    def flaky_feed_command(python, config, args, *, cwd, timeout):
        if args[0] == "run":
            return _cp(args, 1, "", "collect: unhandled exception")
        return _ok_feed_command(python, config, args, cwd=cwd, timeout=timeout)

    monkeypatch.setattr("feed.pipeline._run_feed_command", flaky_feed_command)
    run_gh, run_git = _ok_gh_git(bundle_dir)
    monkeypatch.setattr("feed.pipeline._run_gh", run_gh)
    monkeypatch.setattr("feed.pipeline._run_git", run_git)

    result = run_pipeline(
        cwd=tmp_path, out_dir=bundle_dir,
        logs_dir=tmp_path / "logs", almanac_dir=tmp_path / "almanac",
    )

    publish_outcome = next(s for s in result.stages if s.name == "publish")
    assert publish_outcome.ok
    assert result.almanac_ok is True
    assert result.exit_code == EXIT_DEGRADED


def test_publish_failure_stops_before_almanac_push(tmp_path, monkeypatch):
    """Explicit spec requirement: 'Publishing failing should not lose the
    enrichment work' (nothing here undoes it -- it stays committed in the
    db) -- but a failed publish must never reach the git push at all."""
    bundle_dir = tmp_path / "public"
    _write_bundle(bundle_dir)
    gh_calls = []
    git_calls = []

    def flaky_feed_command(python, config, args, *, cwd, timeout):
        if args[0] == "publish":
            return _cp(args, 1, "", "publish failed: schema validation error")
        return _ok_feed_command(python, config, args, cwd=cwd, timeout=timeout)

    def tracking_gh(args, *, cwd, timeout=120.0):
        gh_calls.append(args)
        return _cp(args, 0)

    def tracking_git(args, *, cwd, timeout=60.0):
        git_calls.append(args)
        return _cp(args, 0)

    monkeypatch.setattr("feed.pipeline._run_feed_command", flaky_feed_command)
    monkeypatch.setattr("feed.pipeline._run_gh", tracking_gh)
    monkeypatch.setattr("feed.pipeline._run_git", tracking_git)

    result = run_pipeline(
        cwd=tmp_path, out_dir=bundle_dir,
        logs_dir=tmp_path / "logs", almanac_dir=tmp_path / "almanac",
    )

    assert result.exit_code == EXIT_SITE_NOT_UPDATED
    assert gh_calls == [] and git_calls == []  # almanac push never attempted
    assert result.almanac_ok is None


def test_almanac_push_failure_is_reported_and_exits_site_not_updated(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "public"
    _write_bundle(bundle_dir)
    monkeypatch.setattr("feed.pipeline._run_feed_command", _ok_feed_command)

    def run_gh(args, *, cwd, timeout=120.0):
        dest = Path(args[3])
        (dest / ".git").mkdir(parents=True)
        return _cp(args, 0)

    def failing_push_git(args, *, cwd, timeout=60.0):
        if args[0] == "status":
            return _cp(args, 0, stdout=" M manifest.json\n")
        if args[0] == "push":
            return _cp(args, 1, "", "remote: authentication failed")
        return _cp(args, 0)

    monkeypatch.setattr("feed.pipeline._run_gh", run_gh)
    monkeypatch.setattr("feed.pipeline._run_git", failing_push_git)

    result = run_pipeline(
        cwd=tmp_path, out_dir=bundle_dir,
        logs_dir=tmp_path / "logs", almanac_dir=tmp_path / "almanac",
    )

    assert result.exit_code == EXIT_SITE_NOT_UPDATED
    assert result.almanac_ok is False
    assert "authentication failed" in result.almanac_detail
    log_text = result.log_path.read_text(encoding="utf-8")
    assert "SITE NOT UPDATED" in log_text


def test_no_changes_to_publish_is_success_not_a_push(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "public"
    _write_bundle(bundle_dir)
    monkeypatch.setattr("feed.pipeline._run_feed_command", _ok_feed_command)

    def run_gh(args, *, cwd, timeout=120.0):
        dest = Path(args[3])
        (dest / ".git").mkdir(parents=True)
        return _cp(args, 0)

    push_calls = []

    def run_git(args, *, cwd, timeout=60.0):
        if args[0] == "status":
            return _cp(args, 0, stdout="")  # nothing changed
        if args[0] == "push":
            push_calls.append(args)
        return _cp(args, 0)

    monkeypatch.setattr("feed.pipeline._run_gh", run_gh)
    monkeypatch.setattr("feed.pipeline._run_git", run_git)

    result = run_pipeline(
        cwd=tmp_path, out_dir=bundle_dir,
        logs_dir=tmp_path / "logs", almanac_dir=tmp_path / "almanac",
    )

    assert result.exit_code == EXIT_OK
    assert result.almanac_ok is True
    assert push_calls == []  # never pushed -- nothing to push


def test_refuses_to_push_if_a_dotenv_like_path_is_staged(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "public"
    _write_bundle(bundle_dir)
    monkeypatch.setattr("feed.pipeline._run_feed_command", _ok_feed_command)

    def run_gh(args, *, cwd, timeout=120.0):
        dest = Path(args[3])
        (dest / ".git").mkdir(parents=True)
        return _cp(args, 0)

    push_calls = []

    def run_git(args, *, cwd, timeout=60.0):
        if args[0] == "status":
            return _cp(args, 0, stdout=" M manifest.json\n?? .env\n")
        if args[0] == "push":
            push_calls.append(args)
        return _cp(args, 0)

    monkeypatch.setattr("feed.pipeline._run_gh", run_gh)
    monkeypatch.setattr("feed.pipeline._run_git", run_git)

    result = run_pipeline(
        cwd=tmp_path, out_dir=bundle_dir,
        logs_dir=tmp_path / "logs", almanac_dir=tmp_path / "almanac",
    )

    assert result.exit_code == EXIT_SITE_NOT_UPDATED
    assert result.almanac_ok is False
    assert "refusing" in result.almanac_detail.lower()
    assert push_calls == []


def test_skip_almanac_push_flag(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "public"
    _write_bundle(bundle_dir)
    monkeypatch.setattr("feed.pipeline._run_feed_command", _ok_feed_command)
    calls = []
    monkeypatch.setattr("feed.pipeline._run_gh", lambda *a, **k: calls.append(1))
    monkeypatch.setattr("feed.pipeline._run_git", lambda *a, **k: calls.append(1))

    result = run_pipeline(
        cwd=tmp_path, out_dir=bundle_dir, skip_almanac_push=True,
        logs_dir=tmp_path / "logs", almanac_dir=tmp_path / "almanac",
    )

    assert result.exit_code == EXIT_OK
    assert calls == []
    assert result.almanac_ok is None


def test_lock_contention_refuses_and_never_touches_a_stage(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("feed.pipeline._run_feed_command",
                        lambda *a, **k: calls.append(1))

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    lock = PipelineLock(logs_dir / "pipeline.lock")
    lock.acquire()  # simulate an already-running instance (this process is alive)
    try:
        result = run_pipeline(
            cwd=tmp_path, out_dir=tmp_path / "public",
            logs_dir=logs_dir, almanac_dir=tmp_path / "almanac",
        )
        assert result.exit_code == EXIT_LOCKED
        assert calls == []
    finally:
        lock.release()


def test_log_retention_prunes_oldest_first(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "public"
    _write_bundle(bundle_dir)
    monkeypatch.setattr("feed.pipeline._run_feed_command", _ok_feed_command)
    run_gh, run_git = _ok_gh_git(bundle_dir)
    monkeypatch.setattr("feed.pipeline._run_gh", run_gh)
    monkeypatch.setattr("feed.pipeline._run_git", run_git)

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    old_names = [f"pipeline_2020010{i}_000000.log" for i in range(1, 6)]
    for name in old_names:
        (logs_dir / name).write_text("old run", encoding="utf-8")

    result = run_pipeline(
        cwd=tmp_path, out_dir=bundle_dir, keep_logs=3,
        logs_dir=logs_dir, almanac_dir=tmp_path / "almanac",
    )

    remaining = sorted(p.name for p in logs_dir.glob("pipeline_*.log"))
    assert len(remaining) == 3
    assert result.log_path.name in remaining  # the run just made is always kept
    assert "pipeline_20200101_000000.log" not in remaining  # oldest, pruned first


def test_mirror_dir_adds_updates_and_removes_stale_files(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src / "sub").mkdir(parents=True)
    (src / "keep.txt").write_text("new-content", encoding="utf-8")
    (src / "sub" / "nested.txt").write_text("nested", encoding="utf-8")

    dst.mkdir()
    (dst / "keep.txt").write_text("stale-content", encoding="utf-8")
    (dst / "stale.txt").write_text("should be removed", encoding="utf-8")
    (dst / "stale_dir").mkdir()
    (dst / "stale_dir" / "x.txt").write_text("also removed", encoding="utf-8")

    _mirror_dir(src, dst)

    assert (dst / "keep.txt").read_text(encoding="utf-8") == "new-content"
    assert (dst / "sub" / "nested.txt").read_text(encoding="utf-8") == "nested"
    assert not (dst / "stale.txt").exists()
    assert not (dst / "stale_dir").exists()  # emptied AND pruned
