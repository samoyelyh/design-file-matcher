param(
    [string]$Mirror = "https://hf-mirror.com",
    [string]$Repository = "facebook/dinov2-base",
    [string]$Destination = (Join-Path $PSScriptRoot "..\models\dinov2-base")
)

$ErrorActionPreference = "Stop"
$destinationPath = [System.IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Force -Path $destinationPath | Out-Null

$files = @(
    @{ Name = "config.json"; Size = 548 },
    @{ Name = "preprocessor_config.json"; Size = 436 },
    @{ Name = "model.safetensors"; Size = 346345912 }
)

foreach ($file in $files) {
    $filename = $file.Name
    $source = "$($Mirror.TrimEnd('/'))/$Repository/resolve/main/$filename"
    $target = Join-Path $destinationPath $filename
    if ((Test-Path -LiteralPath $target) -and
        (Get-Item -LiteralPath $target).Length -eq $file.Size) {
        Write-Host "Already complete: $filename"
        continue
    }

    $displayName = "DesignMatcher-DINOv2-$filename"
    $job = Get-BitsTransfer -ErrorAction SilentlyContinue |
        Where-Object DisplayName -eq $displayName |
        Select-Object -First 1
    if (-not $job) {
        $job = Start-BitsTransfer -Source $source -Destination $target `
            -DisplayName $displayName -Priority Foreground -Asynchronous
    } else {
        Set-BitsTransfer -BitsJob $job -Priority Foreground
        if ($job.JobState -in @("Suspended", "TransientError")) {
            Resume-BitsTransfer -BitsJob $job -Asynchronous
        }
    }

    while ($job.JobState -in @("Queued", "Connecting", "Transferring", "TransientError")) {
        $job = Get-BitsTransfer -Id $job.JobId
        if ($job.JobState -eq "TransientError") {
            Resume-BitsTransfer -BitsJob $job -Asynchronous
            $job = Get-BitsTransfer -Id $job.JobId
        }
        $percent = if ($job.BytesTotal -gt 0 -and $job.BytesTotal -lt [uint64]::MaxValue) {
            [math]::Round(100 * $job.BytesTransferred / $job.BytesTotal, 2)
        } else { 0 }
        Write-Progress -Activity "Downloading DINOv2 from $Mirror" `
            -Status "$filename - $percent%" -PercentComplete $percent
        Start-Sleep -Seconds 2
    }
    if ($job.JobState -ne "Transferred") {
        throw "Download failed: $filename - $($job.ErrorDescription)"
    }
    Complete-BitsTransfer -BitsJob $job
}

Write-Progress -Activity "Downloading DINOv2" -Completed
Write-Host "DINOv2 mirror download complete: $destinationPath"
