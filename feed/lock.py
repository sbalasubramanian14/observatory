from __future__ import annotations
import json
import os
import socket
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


class LockHeld(Exception):
    """Raised by PipelineLock.acquire() when another live process already
    holds the lock. Carries the holder's info in .info so callers (the
    `feed pipeline` CLI command) can print a clear, specific message --
    "already running (pid=1234, started 2026-08-23T07:00:01+00:00)" --
    rather than a bare refusal.
    """

    def __init__(self, info: "LockInfo"):
        self.info = info
        super().__init__(
            f"pipeline already running (pid={info.pid}, started {info.started_at}, "
            f"host={info.host})"
        )


@dataclass
class LockInfo:
    pid: int
    started_at: str
    host: str


def _pid_alive(pid: int) -> bool:
    """True iff a process with this PID currently exists.

    Cross-platform on purpose (tests run wherever pytest runs, production
    is Windows): os.kill(pid, 0) -- the usual POSIX liveness probe -- does
    NOT mean "is this pid alive" on Windows (CPython maps signal 0 to
    GenerateConsoleCtrlEvent there, not a harmless existence probe), so
    Windows gets its own check via OpenProcess. Neither branch requires
    elevated privileges: PROCESS_QUERY_LIMITED_INFORMATION is grantable
    against any process the caller can see at all, which is exactly what
    "is my own crashed run still around" needs.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        STILL_ACTIVE = 259
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            # OpenProcess can succeed for a process that has already
            # terminated but whose handle table entry is still around
            # (e.g. a parent -- like our own subprocess.Popen in tests --
            # still holds a handle to it). The PID itself is only truly
            # "alive" if the exit code is still STILL_ACTIVE.
            exit_code = ctypes.c_ulong()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            return bool(ok) and exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists, just owned by someone else -- still alive.
        return True
    return True


class PipelineLock:
    """File-based mutual exclusion for `feed pipeline` runs.

    The whole point (see PIPELINE-CLI.md): two concurrent writers against
    the same SQLite database is a corruption risk, and a scheduled run
    landing on top of a manual run is exactly the scenario Task Scheduler
    makes likely. A lock FILE alone is not enough -- a crashed run (killed
    process, power loss, Ctrl-C past the finally block) leaves the file on
    disk forever, which would silently block every future run. So the
    file's payload is enough to tell a live holder from a stale one: the
    PID plus a start timestamp. acquire() checks PID liveness (_pid_alive)
    before treating an existing lock file as "held" -- if that PID is not
    running, the lock is stale and is reclaimed automatically (with the
    stale info returned so the caller can log what it recovered from).
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def read(self) -> LockInfo | None:
        """Best-effort parse of the current lock file. Returns None if the
        file is absent OR unparseable (a half-written file from a crash
        mid-write is treated the same as "no useful info" -- acquire()
        still needs to decide stale-vs-live, so an unparseable file is
        just as safe to reclaim as a missing one).
        """
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            data = json.loads(raw)
            return LockInfo(pid=int(data["pid"]), started_at=data["started_at"],
                            host=data["host"])
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return None

    def acquire(self) -> LockInfo | None:
        """Take the lock. Returns the stale LockInfo that was reclaimed (or
        None if the lock was free / absent). Raises LockHeld if a live
        process already owns it.
        """
        stale: LockInfo | None = None
        existing = self.read()
        if existing is not None:
            if _pid_alive(existing.pid):
                raise LockHeld(existing)
            stale = existing

        self.path.parent.mkdir(parents=True, exist_ok=True)
        mine = LockInfo(
            pid=os.getpid(),
            started_at=datetime.now(timezone.utc).isoformat(),
            host=socket.gethostname(),
        )
        # Write to a temp file then replace: an interrupted write leaves the
        # OLD lock (or nothing) rather than a half-written, unparseable one
        # sitting where the next run's staleness check has to guess at it.
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps({"pid": mine.pid, "started_at": mine.started_at,
                       "host": mine.host}),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)
        return stale

    def release(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
