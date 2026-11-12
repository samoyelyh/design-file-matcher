$ErrorActionPreference = "Stop"

$taskName = "DesignMatcher-VisionMatch-Service"
$supervisorPath = Join-Path $PSScriptRoot "run_visionmatch_supervisor.ps1"
$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$supervisorPath`""
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $currentUser
$settings = New-ScheduledTaskSettingsSet `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId $currentUser -LogonType Interactive -RunLevel Limited
Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "VisionMatch V2 Terra backend on port 8066; independent of Design Matcher 8088." `
    -Force | Out-Null

Start-ScheduledTask -TaskName $taskName
$healthy = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:8066/api/health" -TimeoutSec 2
        if ($health.ok -and $health.product -eq "VisionMatch V2" -and -not $health.validation_replay) {
            $healthy = $true
            break
        }
    }
    catch { }
    Start-Sleep -Seconds 1
}
if (-not $healthy) {
    throw "VisionMatch did not become healthy. Check $env:LOCALAPPDATA\DesignMatcher\visionmatch-service"
}
Write-Host "Installed and running: $taskName"
Write-Host "Starts when $currentUser logs in; restarts the backend after unexpected exits."
Write-Host "Backend: http://127.0.0.1:8066/"
Write-Host "Design Matcher (local): http://127.0.0.1:8088/"
$lanAddress = Get-NetIPAddress -AddressFamily IPv4 -AddressState Preferred -ErrorAction SilentlyContinue |
    Where-Object {
        $_.IPAddress -notlike "127.*" -and
        $_.IPAddress -notlike "169.254.*" -and
        $_.InterfaceAlias -notmatch "vEthernet|VMware|Loopback"
    } |
    Select-Object -ExpandProperty IPAddress -First 1
if ($lanAddress) {
    Write-Host "Design Matcher (LAN): http://${lanAddress}:8088/"
}
