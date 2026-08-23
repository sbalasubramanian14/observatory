<#
.SYNOPSIS
    Registers (or unregisters) a Windows Task Scheduler job that runs the
    Observatory pipeline twice a day.

.DESCRIPTION
    NOT run automatically by anything -- this is a script the machine's
    owner runs themselves, once, when they decide to turn on scheduling.
    Registering a scheduled task changes the owner's Windows configuration,
    which is their decision to make, not an automated agent's.

    Defaults (all overridable, see PARAMETER below):
      - Runs scripts\..\observatory.bat twice daily at 07:00 and 19:00.
      - "Run only when the user is logged on" (-LogonType Interactive, no
        stored password) -- the task will simply not fire if the machine is
        signed out; it never needs (or asks for) the Windows password.
      - Does NOT wake the machine from sleep (no -WakeToRun). A run that
        was skipped because the machine was asleep just waits for the next
        scheduled time, or a manual double-click of observatory.bat.
      - Does NOT run if the task is somehow still running from before
        (-MultipleInstances IgnoreNew) -- belt-and-braces on top of
        feed/pipeline.py's own lock file, which is the real guard.

.PARAMETER Times
    One or more HH:mm times to run at each day. Default: 07:00 and 19:00.

.PARAMETER TaskName
    Name of the Task Scheduler task. Default: "Observatory Pipeline".

.PARAMETER Unregister
    Remove the task instead of creating it. Trivially reverses everything
    this script does.

.PARAMETER Force
    Skip the confirmation prompt.

.EXAMPLE
    scripts\register-schedule.ps1
    scripts\register-schedule.ps1 -Times "06:30","12:00","22:00"
    scripts\register-schedule.ps1 -Unregister
#>
[CmdletBinding()]
param(
    [string[]]$Times = @("07:00", "19:00"),
    [string]$TaskName = "Observatory Pipeline",
    [switch]$Unregister,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$BatPath = Join-Path $RepoRoot "observatory.bat"

function Confirm-OrExit([string]$Message) {
    if ($Force) { return }
    Write-Host ""
    $answer = Read-Host "$Message Type 'yes' to proceed"
    if ($answer -ne "yes") {
        Write-Host "Cancelled -- nothing changed." -ForegroundColor Yellow
        exit 0
    }
}

if ($Unregister) {
    Write-Host "This will UNREGISTER the scheduled task:" -ForegroundColor Cyan
    Write-Host "  Name: $TaskName"
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $existing) {
        Write-Host "No such task is currently registered -- nothing to do." -ForegroundColor Yellow
        exit 0
    }
    Confirm-OrExit "Remove scheduled task '$TaskName'?"
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed scheduled task '$TaskName'." -ForegroundColor Green
    exit 0
}

if (-not (Test-Path $BatPath)) {
    Write-Host "ERROR: $BatPath not found -- run this from the scripts\ directory of the observatory repo." -ForegroundColor Red
    exit 1
}

Write-Host "This will REGISTER a scheduled task:" -ForegroundColor Cyan
Write-Host "  Name        : $TaskName"
Write-Host "  Action      : `"$BatPath`" scheduled"
Write-Host "  Times       : $($Times -join ', ') -- every day"
Write-Host "  Run as      : $env:USERDOMAIN\$env:USERNAME, only when logged on"
Write-Host "                (no password stored or requested; the task simply"
Write-Host "                 does not fire while signed out)"
Write-Host "  Wake        : will NOT wake the machine from sleep"
Write-Host "  Overlap     : a run already in progress -> this trigger is skipped"
Write-Host "                (Task Scheduler's own guard; feed pipeline's own"
Write-Host "                 lock file is the real one and works regardless)"
Write-Host ""
Write-Host "To undo this later:  scripts\register-schedule.ps1 -Unregister" -ForegroundColor DarkGray

Confirm-OrExit "Register this task?"

$action = New-ScheduledTaskAction -Execute $BatPath -Argument "scheduled" -WorkingDirectory $RepoRoot

$triggers = foreach ($t in $Times) {
    $parsed = [DateTime]::ParseExact($t, "HH:mm", $null)
    New-ScheduledTaskTrigger -Daily -At $parsed
}

$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive

$settings = New-ScheduledTaskSettingsSet `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable `
    -DontStopOnIdleEnd `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3)
# Deliberately NOT setting -WakeToRun: the owner's explicit decision was
# "do not wake the machine from sleep" -- New-ScheduledTaskSettingsSet
# defaults to WakeToRun = $false, so this is simply never enabled.

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $triggers `
    -Principal $principal -Settings $settings `
    -Description "Runs the Observatory pipeline (collect -> enrich -> publish -> push to observatory-almanac). Registered by scripts\register-schedule.ps1." `
    | Out-Null

Write-Host ""
Write-Host "Registered '$TaskName'. View/edit it any time in Task Scheduler," -ForegroundColor Green
Write-Host "or run:  Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo" -ForegroundColor Green
