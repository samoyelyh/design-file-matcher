$ErrorActionPreference = "Stop"

$runtimeRoot = Join-Path $env:LOCALAPPDATA "DesignMatcher\service"
$supervisorPidPath = Join-Path $runtimeRoot "supervisor.pid"
$serverPidPath = Join-Path $runtimeRoot "server.pid"
$stopFlagPath = Join-Path $runtimeRoot "stop.flag"

New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null
Set-Content -LiteralPath $stopFlagPath -Value "stop" -Encoding ascii

if (Test-Path -LiteralPath $serverPidPath) {
    $serverPid = [int](Get-Content -LiteralPath $serverPidPath -ErrorAction SilentlyContinue)
    $server = Get-CimInstance Win32_Process -Filter "ProcessId = $serverPid" -ErrorAction SilentlyContinue
    if ($server -and $server.CommandLine -like "*uvicorn*app.main:app*") {
        Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Where-Object {
                $_.ParentProcessId -eq $serverPid -and
                $_.CommandLine -like "*uvicorn*app.main:app*"
            } |
            ForEach-Object {
                Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
            }
        Stop-Process -Id $serverPid -Force -ErrorAction SilentlyContinue
    }
}

for ($attempt = 0; $attempt -lt 10; $attempt++) {
    if (-not (Test-Path -LiteralPath $supervisorPidPath)) {
        break
    }
    Start-Sleep -Milliseconds 300
}

if (Test-Path -LiteralPath $supervisorPidPath) {
    $supervisorPid = [int](Get-Content -LiteralPath $supervisorPidPath -ErrorAction SilentlyContinue)
    $supervisor = Get-CimInstance Win32_Process -Filter "ProcessId = $supervisorPid" -ErrorAction SilentlyContinue
    if ($supervisor -and $supervisor.CommandLine -like "*run_lan_supervisor.ps1*") {
        Stop-Process -Id $supervisorPid -Force -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath $supervisorPidPath -Force -ErrorAction SilentlyContinue
}

Remove-Item -LiteralPath $serverPidPath -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $stopFlagPath -Force -ErrorAction SilentlyContinue
Write-Host "Design Matcher background service stopped."
