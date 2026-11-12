param(
    [string]$BackendRoot
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $BackendRoot) {
    $existingBackend = Join-Path (Split-Path -Parent $projectRoot) "VisionMatch AI"
    $BackendRoot = if (Test-Path -LiteralPath (Join-Path $existingBackend "visionmatch\product\app.py")) {
        $existingBackend
    } else {
        Join-Path $projectRoot "visionmatch-backend"
    }
}
$pythonPath = if ($env:VISIONMATCH_PYTHON) {
    $env:VISIONMATCH_PYTHON
} elseif ($BackendRoot -eq $existingBackend -and (Test-Path -LiteralPath "G:\anaconda\python.exe")) {
    "G:\anaconda\python.exe"
} else {
    Join-Path $projectRoot ".venv\Scripts\python.exe"
}
$runtimeRoot = Join-Path $env:LOCALAPPDATA "DesignMatcher\visionmatch-service"
$supervisorPidPath = Join-Path $runtimeRoot "supervisor.pid"
$serverPidPath = Join-Path $runtimeRoot "server.pid"
$stopFlagPath = Join-Path $runtimeRoot "stop.flag"
$logPath = Join-Path $runtimeRoot "supervisor.log"
$mutex = New-Object System.Threading.Mutex($false, "Local\DesignMatcher-VisionMatch-8066-Supervisor")
$ownsMutex = $false
$server = $null

try {
    try { $ownsMutex = $mutex.WaitOne(0) }
    catch [System.Threading.AbandonedMutexException] { $ownsMutex = $true }
    if (-not $ownsMutex) { exit 0 }
    if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
        throw "Python environment not found: $pythonPath"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $BackendRoot "visionmatch\product\app.py") -PathType Leaf)) {
        throw "VisionMatch V2 source not found: $BackendRoot"
    }

    New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null
    Remove-Item -LiteralPath $stopFlagPath -Force -ErrorAction SilentlyContinue
    Set-Content -LiteralPath $supervisorPidPath -Value $PID -Encoding ascii
    $env:PYTHONUNBUFFERED = "1"
    $env:PYTHONIOENCODING = "utf-8"
    $env:VISIONMATCH_V2_DATA = Join-Path $BackendRoot "runtime\v2"

    while (-not (Test-Path -LiteralPath $stopFlagPath)) {
        # Leave existing listeners alone, including a manually started backend.
        $listener = Get-NetTCPConnection -LocalPort 8066 -State Listen -ErrorAction SilentlyContinue
        if ($listener) {
            Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) port 8066 already occupied; no process restarted"
            Start-Sleep -Seconds 10
            continue
        }

        $startedAt = Get-Date -Format "yyyyMMdd-HHmmss-fff"
        Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) starting VisionMatch V2 on 127.0.0.1:8066"
        $server = Start-Process `
            -FilePath $pythonPath `
            -ArgumentList @("-m", "uvicorn", "visionmatch.product.app:app", "--host", "127.0.0.1", "--port", "8066") `
            -WorkingDirectory $BackendRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $runtimeRoot "server-$startedAt.out.log") `
            -RedirectStandardError (Join-Path $runtimeRoot "server-$startedAt.err.log") `
            -PassThru
        Set-Content -LiteralPath $serverPidPath -Value $server.Id -Encoding ascii
        while (-not $server.WaitForExit(1000)) {
            if (Test-Path -LiteralPath $stopFlagPath) {
                Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
                $server.WaitForExit()
                break
            }
        }
        Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) backend exited code=$($server.ExitCode)"
        Remove-Item -LiteralPath $serverPidPath -Force -ErrorAction SilentlyContinue
        $server = $null
        if (-not (Test-Path -LiteralPath $stopFlagPath)) { Start-Sleep -Seconds 3 }
    }
}
finally {
    if ($ownsMutex) {
        if ($server -and -not $server.HasExited) { Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue }
        Remove-Item -LiteralPath $serverPidPath -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $supervisorPidPath -Force -ErrorAction SilentlyContinue
        $mutex.ReleaseMutex()
    }
    $mutex.Dispose()
}
