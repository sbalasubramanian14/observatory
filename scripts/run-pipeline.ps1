<#
.SYNOPSIS
    Runs the full one-click Observatory pipeline: sources sync -> run ->
    enrich -> publish -> push the bundle to observatory-almanac.

.DESCRIPTION
    This is the "real work" behind observatory.bat. It resolves the repo
    root and the project's own venv Python (never a bare `python` --
    that's 3.15 on this machine and this project pins 3.14), validates
    both exist with a clear error if not, then hands off to `feed
    pipeline` (feed/pipeline.py), which does the actual stage sequencing,
    locking, logging, and almanac push. This script's job is purely
    getting a double-click or a Task Scheduler trigger into that command
    reliably: resolving paths (a scheduled task's working directory is
    NOT guaranteed to be the repo root), and turning the exit code into
    something a human (or Task Scheduler's "Last Run Result" column) can
    read.

.PARAMETER SkipAlmanacPush
    Run every stage but do not push to the almanac repo. Useful for a
    manual dry run.

.PARAMETER KeepLogs
    How many past run logs to retain under logs\ (default: 30).

.PARAMETER StageTimeout
    Per-stage subprocess timeout in seconds (default: 3600 = 1 hour).

.PARAMETER Config
    Path to an alternate feed.toml (default: the repo's own).

.PARAMETER Days
    Publish only the last N days of news, overriding [publish].retention_days
    in feed.toml for THIS RUN ONLY -- the config file is never rewritten, so
    the next plain run returns to the configured window. Nothing is lost by
    narrowing it: the database keeps every story regardless, and re-running
    with a larger number (or none) brings the older ones straight back.

.PARAMETER DryRun
    Resolve everything and print the command that WOULD run, then exit 0
    without touching the database, the LLM providers, or the almanac repo.
    Useful for confirming an argument landed the way you meant it to.

.PARAMETER Scheduled
    Set by scripts\register-schedule.ps1's registered task. Suppresses
    the "press any key" pause at the end -- Task Scheduler has no console
    to press a key in, and a paused, never-exiting process would just sit
    there until the next scheduled run piles on top of it (which the lock
    file would then correctly refuse -- but it is better not to rely on
    that for something this avoidable).

.EXAMPLE
    scripts\run-pipeline.ps1
    scripts\run-pipeline.ps1 -SkipAlmanacPush
    scripts\run-pipeline.ps1 -Scheduled -KeepLogs 60
    scripts\run-pipeline.ps1 -Days 7
    scripts\run-pipeline.ps1 -Days 7 -DryRun
#>
[CmdletBinding()]
param(
    [switch]$SkipAlmanacPush,
    [int]$KeepLogs = 30,
    [double]$StageTimeout = 3600,
    [string]$Config,
    # 0 means "not supplied" -- fall through to feed.toml's retention_days.
    # Validated below rather than with [ValidateRange], which would reject
    # this sentinel default before the script ever starts.
    [int]$Days = 0,
    [switch]$DryRun,
    [switch]$Scheduled
)

$ErrorActionPreference = "Stop"

# scripts\ is always one level under the repo root -- this holds whether
# invoked by double-click, by observatory.bat, or by Task Scheduler
# (which is given this script's absolute path at registration time, see
# register-schedule.ps1).
$RepoRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"

Write-Host "=== Observatory pipeline launcher ===" -ForegroundColor Cyan
Write-Host "repo root : $RepoRoot"

if (-not (Test-Path $PythonExe)) {
    Write-Host ""
    Write-Host "ERROR: venv Python not found at $PythonExe" -ForegroundColor Red
    Write-Host "Run this once from the repo root to create it:" -ForegroundColor Red
    Write-Host "  py -3.14 -m venv .venv" -ForegroundColor Red
    Write-Host "  .venv\Scripts\python.exe -m pip install -e .[dev]" -ForegroundColor Red
    if (-not $Scheduled) { Read-Host "Press Enter to close" | Out-Null }
    exit 99
}

$FeedToml = if ($Config) { $Config } else { Join-Path $RepoRoot "feed.toml" }
if (-not (Test-Path $FeedToml)) {
    Write-Host ""
    Write-Host "ERROR: config not found at $FeedToml" -ForegroundColor Red
    if (-not $Scheduled) { Read-Host "Press Enter to close" | Out-Null }
    exit 99
}

if ($Days -lt 0) {
    Write-Host ""
    Write-Host "ERROR: -Days must be 1 or more (got $Days)" -ForegroundColor Red
    if (-not $Scheduled) { Read-Host "Press Enter to close" | Out-Null }
    exit 99
}

# Built here rather than inside the try below so -DryRun can report it and
# leave without entering the invocation block at all.
$pyArgs = @("-m", "feed", "--config", $FeedToml, "pipeline",
           "--keep-logs", $KeepLogs, "--stage-timeout", $StageTimeout)
if ($SkipAlmanacPush) { $pyArgs += "--skip-almanac-push" }
if ($Days -gt 0) { $pyArgs += @("--days", $Days) }

Write-Host "python    : $PythonExe"
if ($Days -gt 0) {
    Write-Host "window    : last $Days day(s) of news -- overrides feed.toml for this run only"
} else {
    Write-Host "window    : from feed.toml ([publish].retention_days)"
}

# Printed as a single resolved line so a caller -- including
# tests/test_runner_scripts.py -- can assert on exactly what would reach
# python, rather than on the source text of this script. An argument
# silently dropped somewhere between cmd.exe, PowerShell's parameter
# binder and python's argv is the failure worth guarding against, and only
# the resolved argv can reveal it.
if ($DryRun) {
    Write-Host ""
    Write-Host "would run : $PythonExe $($pyArgs -join ' ')" -ForegroundColor Cyan
    Write-Host "DRY RUN -- nothing was executed." -ForegroundColor Yellow
    exit 0
}

Write-Host ""

Push-Location $RepoRoot
try {
    & $PythonExe @pyArgs
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

Write-Host ""
switch ($exitCode) {
    0  { Write-Host "Pipeline finished: SUCCESS (exit 0)" -ForegroundColor Green }
    1  { Write-Host "Pipeline REFUSED to start: another run is already in progress (exit 1)" -ForegroundColor Yellow }
    2  { Write-Host "Pipeline FAILED: the live site was NOT updated this run (exit 2) -- check logs\" -ForegroundColor Red }
    3  { Write-Host "Pipeline finished DEGRADED: site updated, but an earlier stage had errors (exit 3) -- check logs\" -ForegroundColor Yellow }
    99 { Write-Host "Pipeline hit an UNEXPECTED error in the orchestrator itself (exit 99) -- check logs\" -ForegroundColor Red }
    default { Write-Host "Pipeline exited with code $exitCode" -ForegroundColor Red }
}

if (-not $Scheduled -and $exitCode -ne 0) {
    Read-Host "Press Enter to close" | Out-Null
}

exit $exitCode
