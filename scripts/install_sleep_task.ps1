# Install Windows Scheduled Task for Star Graph Sleep

param(
    [string]$ScriptPath = (Join-Path $PSScriptRoot "sleep_daemon.py"),
    [string]$PythonPath = (Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source),
    [string]$TaskName = "StarGraphMemorySleep",
    [int]$Hour = 2
)

if (-not $PythonPath) {
    $PythonPath = "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python311\python.exe"
}

$Action = New-ScheduledTaskAction -Execute $PythonPath -Argument "`"$ScriptPath`" --mode scheduled --data-dir `"$env:USERPROFILE\.star_graph`""
$Trigger = New-ScheduledTaskTrigger -Daily -At "$($Hour.ToString().PadLeft(2,'0')):00"
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings -Description "Star Graph Memory nightly consolidation — replays, merges, prunes, and bridges memory anchors during sleep hours."

Write-Host "Scheduled task '$TaskName' installed (daily at $($Hour):00)" -ForegroundColor Green
Write-Host "Script: $ScriptPath" -ForegroundColor Gray
Write-Host "Data dir: $env:USERPROFILE\.star_graph" -ForegroundColor Gray
