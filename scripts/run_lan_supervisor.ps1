$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$runtimeRoot = Join-Path $env:LOCALAPPDATA "DesignMatcher\service"
$supervisorPidPath = Join-Path $runtimeRoot "supervisor.pid"
$serverPidPath = Join-Path $runtimeRoot "server.pid"
$stopFlagPath = Join-Path $runtimeRoot "stop.flag"
$supervisorLogPath = Join-Path $runtimeRoot "supervisor.log"

New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python environment not found: $pythonPath"
}

if (Test-Path -LiteralPath $supervisorPidPath) {
    $existingPid = [int](Get-Content -LiteralPath $supervisorPidPath -ErrorAction SilentlyContinue)
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like "*run_lan_supervisor.ps1*") {
        exit 0
    }
}

Remove-Item -LiteralPath $stopFlagPath -Force -ErrorAction SilentlyContinue
Set-Content -LiteralPath $supervisorPidPath -Value $PID -Encoding ascii
$env:PYTHONUNBUFFERED = "1"

try {
    while (-not (Test-Path -LiteralPath $stopFlagPath)) {
        $startedAt = Get-Date -Format "yyyyMMdd-HHmmss"
        $stdoutPath = Join-Path $runtimeRoot "server-$startedAt.out.log"
        $stderrPath = Join-Path $runtimeRoot "server-$startedAt.err.log"
        Add-Content -LiteralPath $supervisorLogPath -Value "$(Get-Date -Format o) starting uvicorn"

        $server = Start-Process `
            -FilePath $pythonPath `
            -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8088") `
            -WorkingDirectory $projectRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath `
            -PassThru

        Set-Content -LiteralPath $serverPidPath -Value $server.Id -Encoding ascii
        $server.WaitForExit()
        Remove-Item -LiteralPath $serverPidPath -Force -ErrorAction SilentlyContinue
        Add-Content -LiteralPath $supervisorLogPath -Value "$(Get-Date -Format o) uvicorn exited code=$($server.ExitCode)"

        if (-not (Test-Path -LiteralPath $stopFlagPath)) {
            Start-Sleep -Seconds 2
        }
    }
}
finally {
    Remove-Item -LiteralPath $serverPidPath -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $supervisorPidPath -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stopFlagPath -Force -ErrorAction SilentlyContinue
    Add-Content -LiteralPath $supervisorLogPath -Value "$(Get-Date -Format o) supervisor stopped"
}
