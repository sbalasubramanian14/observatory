@echo off
REM Observatory one-click pipeline runner.
REM
REM Double-click this file to run the full pipeline (sources sync -> run ->
REM enrich -> publish -> push to observatory-almanac). All the real work is
REM in scripts\run-pipeline.ps1 -- this is a thin shim so double-clicking
REM works without changing Windows' default .ps1 handling (which normally
REM opens scripts in Notepad, not runs them).
REM
REM   observatory.bat            full run; window comes from feed.toml
REM   observatory.bat 7          full run; publish only the last 7 days
REM   observatory.bat scheduled  full run, no end-of-run pause
REM   observatory.bat 7 dryrun   print the command it WOULD run, then stop
REM
REM Tokens may appear in any order. Anything unrecognised is an ERROR rather
REM than being ignored: a typo like `observatory.bat sevendays` must not
REM quietly run with the default window, because the whole point of passing
REM an argument is that you wanted a different one.
setlocal
set "SCRIPT_DIR=%~dp0"
set "SCHED="
set "DRY="
set "DAYS="

:parse
if "%~1"=="" goto :run
if /I "%~1"=="scheduled" (set "SCHED=-Scheduled" & shift & goto :parse)
if /I "%~1"=="dryrun" (set "DRY=-DryRun" & shift & goto :parse)
echo %~1| findstr /r /c:"^[1-9][0-9]*$" >nul
if not errorlevel 1 (set "DAYS=%~1" & shift & goto :parse)
echo.
echo ERROR: unrecognised argument "%~1"
echo.
echo Usage: observatory.bat [days] [scheduled] [dryrun]
echo   days       a whole number of days of news to publish, 1 or more
echo   scheduled  suppress the end-of-run pause (used by the scheduled task)
echo   dryrun     print the command that would run, without running it
echo.
endlocal
exit /b 2

:run
set "DAYSARG="
if defined DAYS set "DAYSARG=-Days %DAYS%"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%scripts\run-pipeline.ps1" %SCHED% %DRY% %DAYSARG%

set "EXITCODE=%ERRORLEVEL%"
endlocal & exit /b %EXITCODE%
