@echo off
REM Observatory one-click pipeline runner.
REM
REM Double-click this file to run the full pipeline (sources sync -> run ->
REM enrich -> publish -> push to observatory-almanac). All the real work is
REM in scripts\run-pipeline.ps1 -- this is a thin shim so double-clicking
REM works without changing Windows' default .ps1 handling (which normally
REM opens scripts in Notepad, not runs them).
REM
REM Pass "scheduled" as the first argument (scripts\register-schedule.ps1
REM does this automatically) to suppress the end-of-run pause, since a
REM scheduled task has no console for a human to press a key in.
setlocal
set "SCRIPT_DIR=%~dp0"

if /I "%~1"=="scheduled" (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%scripts\run-pipeline.ps1" -Scheduled
) else (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%scripts\run-pipeline.ps1"
)

set "EXITCODE=%ERRORLEVEL%"
endlocal & exit /b %EXITCODE%
