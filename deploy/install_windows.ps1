# Register ASH as a scheduled task that starts at logon and restarts on failure.
# Run in an elevated PowerShell from the A.S.H directory:
#   powershell -ExecutionPolicy Bypass -File deploy\install_windows.ps1

$AshDir = (Get-Location).Path
$Python = Join-Path $AshDir ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $Python)) { $Python = "pythonw.exe" }

$Action  = New-ScheduledTaskAction -Execute $Python `
             -Argument "`"$AshDir\ashd.py`"" -WorkingDirectory $AshDir
$Trigger = New-ScheduledTaskTrigger -AtLogOn

# RestartCount/Interval is what makes this a supervisor rather than a launcher.
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit 0 -StartWhenAvailable

Register-ScheduledTask -TaskName "ASH" -Action $Action -Trigger $Trigger `
    -Settings $Settings -Description "A.S.H always-on assistant" -Force

Write-Host "Registered. Start now with:  Start-ScheduledTask -TaskName ASH"
Write-Host "Check on it with:            python ashctl.py status"
