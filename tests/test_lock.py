from __future__ import annotations
import json
import os
import subprocess
import sys
import time
import pytest
from feed.lock import LockHeld, LockInfo, PipelineLock, _pid_alive


def test_acquire_writes_a_lock_file_with_our_own_pid(tmp_path):
    lock = PipelineLock(tmp_path / "pipeline.lock")
    stale = lock.acquire()
    assert stale is None
    data = json.loads(lock.path.read_text(encoding="utf-8"))
    assert data["pid"] == os.getpid()
    lock.release()


def test_release_removes_the_file(tmp_path):
    lock = PipelineLock(tmp_path / "pipeline.lock")
    lock.acquire()
    assert lock.path.exists()
    lock.release()
    assert not lock.path.exists()


def test_release_is_a_noop_when_no_lock_file_exists(tmp_path):
    lock = PipelineLock(tmp_path / "pipeline.lock")
    lock.release()  # must not raise


def test_concurrent_invocation_is_refused(tmp_path):
    """Two PipelineLock instances pointed at the same file, from the same
    (very much alive) process, model two concurrent `feed pipeline`
    invocations. The second must be refused, not silently allowed to
    proceed and corrupt the database with two writers.
    """
    path = tmp_path / "pipeline.lock"
    first = PipelineLock(path)
    first.acquire()

    second = PipelineLock(path)
    with pytest.raises(LockHeld) as exc_info:
        second.acquire()
    assert str(os.getpid()) in str(exc_info.value)

    first.release()


def test_stale_lock_from_a_crashed_process_is_recovered_not_blocked_forever(tmp_path):
    """Non-vacuous by mutation: this test starts a REAL child process, proves
    the lock refuses to be acquired while that process is alive (the file's
    pid is genuinely live, not just a number pytest made up), then kills
    that same process and proves the lock recovers -- all in this one test,
    so the "before" and "after" are directly comparable and the "after"
    can't pass merely because nothing was ever actually locked.
    """
    path = tmp_path / "pipeline.lock"
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
    )
    try:
        assert _pid_alive(proc.pid)
        path.write_text(
            json.dumps({"pid": proc.pid, "started_at": "2026-08-23T07:00:00+00:00",
                       "host": "test-host"}),
            encoding="utf-8",
        )
        lock = PipelineLock(path)
        with pytest.raises(LockHeld):
            lock.acquire()  # MUTATION #1: process alive -> refused

        proc.terminate()
        proc.wait(timeout=10)
        deadline = time.monotonic() + 5
        while _pid_alive(proc.pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _pid_alive(proc.pid)

        stale = lock.acquire()  # MUTATION #2: process dead -> reclaimed
        assert stale is not None
        assert stale.pid == proc.pid
        assert json.loads(path.read_text(encoding="utf-8"))["pid"] == os.getpid()
        lock.release()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


def test_unparseable_lock_file_is_treated_as_stale(tmp_path):
    path = tmp_path / "pipeline.lock"
    path.write_text("not json at all", encoding="utf-8")
    lock = PipelineLock(path)
    stale = lock.acquire()
    assert stale is None  # unparseable -> no info to report, but not fatal
    lock.release()


def test_read_returns_none_when_file_absent(tmp_path):
    lock = PipelineLock(tmp_path / "pipeline.lock")
    assert lock.read() is None


def test_pid_alive_false_for_a_pid_that_does_not_exist():
    # PID 0 is reserved/invalid on every platform we run on.
    assert _pid_alive(0) is False
