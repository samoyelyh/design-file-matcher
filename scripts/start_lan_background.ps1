$ErrorActionPreference = "Stop"

$runtimeRoot = Join-Path $env:LOCALAPPDATA "DesignMatcher\service"
$supervisorPidPath = Join-Path $runtimeRoot "supervisor.pid"
$supervisorPath = Join-Path $PSScriptRoot "run_lan_supervisor.ps1"

New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null

$alreadyRunning = $false
if (Test-Path -LiteralPath $supervisorPidPath) {
    $existingPid = [int](Get-Content -LiteralPath $supervisorPidPath -ErrorAction SilentlyContinue)
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    $alreadyRunning = [bool]($existing -and $existing.CommandLine -like "*run_lan_supervisor.ps1*")
}

if (-not $alreadyRunning) {
    $listener = Get-NetTCPConnection -LocalPort 8088 -State Listen -ErrorAction SilentlyContinue
    if ($listener) {
        throw "Port 8088 is already occupied by process $($listener.OwningProcess)."
    }

    Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", ('"' + $supervisorPath + '"')
        ) `
        -WindowStyle Hidden | Out-Null
}

$healthy = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:8088/api/health" -TimeoutSec 2
        if ($response.status -eq "ok") {
            $healthy = $true
            break
        }
    }
    catch {
        Start-Sleep -Seconds 1
    }
}

if (-not $healthy) {
    $errorLog = Get-ChildItem -LiteralPath $runtimeRoot -Filter "server-*.err.log" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    $hint = if ($errorLog) { " Check log: $($errorLog.FullName)" } else { "" }
    throw "The LAN service did not become healthy.$hint"
}

$lanAddress = Get-NetIPAddress -AddressFamily IPv4 -AddressState Preferred -ErrorAction SilentlyContinue |
    Where-Object {
        $_.IPAddress -notlike "127.*" -and
        $_.IPAddress -notlike "169.254.*" -and
        $_.InterfaceAlias -notmatch "vEthernet|VMware|Loopback"
    } |
    Select-Object -ExpandProperty IPAddress -First 1

Write-Host "Design Matcher is running in the background."
Write-Host "Local: http://127.0.0.1:8088/"
if ($lanAddress) {
    Write-Host "LAN:   http://${lanAddress}:8088/"
}
Write-Host "Logs:  $runtimeRoot"
Write-Host "Stop:  stop-lan.cmd"
