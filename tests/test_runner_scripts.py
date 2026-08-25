"""The observatory.bat / run-pipeline.ps1 launcher pair.

These run the real scripts through the real shell rather than asserting on
their source text: the whole failure mode worth guarding against here is
"the argument was accepted and then silently dropped somewhere between
cmd.exe, PowerShell's parameter binder, and the python argv" -- which a
string match on the file would happily miss.

`-DryRun` exists for exactly this: it resolves everything and prints the
argv it WOULD run, then exits 0 without touching the database, the LLM
providers, or the almanac repo.
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path
import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Windows-only launcher scripts"
)

REPO = Path(__file__).resolve().parent.parent
BAT = REPO / "observatory.bat"


def _run_bat(*args: str) -> str:
    proc = subprocess.run(
        ["cmd.exe", "/c", str(BAT), *args],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout


def _argv_line(stdout: str) -> str:
    for line in stdout.splitlines():
        if line.startswith("would run :"):
            return line
    raise AssertionError(f"no 'would run' line in:\n{stdout}")


def test_bat_forwards_a_numeric_argument_as_days():
    """`observatory.bat 7` -- the whole point of this change."""
    assert "--days 7" in _argv_line(_run_bat("7", "dryrun"))


def test_bat_without_a_numeric_argument_passes_no_days():
    """No argument must leave feed.toml's [publish].retention_days in
    charge, so the scheduled task's window is unaffected by this feature."""
    assert "--days" not in _argv_line(_run_bat("dryrun"))


def test_bat_accepts_scheduled_and_days_together():
    """register-schedule.ps1 passes `scheduled`; a future scheduled task
    may also want a fixed window, so the two tokens must compose."""
    line = _argv_line(_run_bat("scheduled", "3", "dryrun"))
    assert "--days 3" in line


def test_bat_token_order_does_not_matter():
    assert "--days 3" in _argv_line(_run_bat("dryrun", "3", "scheduled"))


def test_bat_rejects_a_non_numeric_argument():
    """A typo like `observatory.bat sevendays` must fail loudly rather than
    silently running with the default window -- the user asked for a
    specific window and would otherwise never learn they didn't get it."""
    proc = subprocess.run(
        ["cmd.exe", "/c", str(BAT), "sevendays", "dryrun"],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode != 0
    assert "sevendays" in proc.stdout + proc.stderr


def test_bat_rejects_zero_days():
    """0 would publish an empty bundle and prune every story file."""
    proc = subprocess.run(
        ["cmd.exe", "/c", str(BAT), "0", "dryrun"],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode != 0
