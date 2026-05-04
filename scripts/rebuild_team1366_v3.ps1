# Rebuild team1366:v3 without cache, repackage as tar.gz, and report size.
# PowerShell counterpart to scripts/rebuild_team1366_v3.sh (use this on Windows
# when bash/WSL is not available).
$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

Write-Host '=== docker build (--no-cache) ==='
docker build --no-cache -t team1366:v3 -f docker/Dockerfile .
if ($LASTEXITCODE -ne 0) { throw "docker build failed (exit $LASTEXITCODE)" }

Write-Host '=== docker save -> tar.gz ==='
$archive = 'artifacts/submissions/team1366_v3.tar.gz'
# Pipe through Linux gzip if available, otherwise fall back to .NET GZipStream.
$gzip = Get-Command gzip -ErrorAction SilentlyContinue
if ($gzip) {
    docker save team1366:v3 | & $gzip.Source | Out-File -FilePath $archive -Encoding Byte
} else {
    $tar = 'artifacts/submissions/team1366_v3.tar'
    docker save -o $tar team1366:v3
    if ($LASTEXITCODE -ne 0) { throw "docker save failed (exit $LASTEXITCODE)" }
    $in = [System.IO.File]::OpenRead($tar)
    $out = [System.IO.File]::Create($archive)
    $gz  = New-Object System.IO.Compression.GZipStream($out, [System.IO.Compression.CompressionMode]::Compress)
    try { $in.CopyTo($gz) } finally { $gz.Dispose(); $out.Dispose(); $in.Dispose() }
    Remove-Item $tar
}

Write-Host '=== size ==='
$bytes = (Get-Item $archive).Length
$mb = [math]::Round($bytes / 1MB, 2)
$gb = [math]::Round($bytes / 1GB, 3)
Write-Host "tar.gz size: $mb MB ($gb GB)"
