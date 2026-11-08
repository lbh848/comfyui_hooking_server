param([switch]$LauncherMode)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8:backslashreplace'
$sourceRoot = $PSScriptRoot
$workshopName = 'Comfy' + [char]0xACF5 + [char]0xBC29
$settingsUrl = 'http://127.0.0.1:8201/setting'
$mutex = New-Object System.Threading.Mutex($false, 'Local\ComfyWorkshopSettingsLauncher8201')
$ownsMutex = $false
$exitCode = 0

function Write-StartupStatus([string]$Stage, [string]$Message) {
    if ($LauncherMode) { Write-Output "COMFY_STARTUP|$Stage|$Message" }
    else { Write-Host $Message }
}

function Open-RunningSettings {
    try {
        $status = Invoke-RestMethod 'http://127.0.0.1:8201/api/setting/status' -TimeoutSec 2
    } catch {
        return $false
    }
    if ($status.name -ne $workshopName -or $status.mode -ne 'settings') {
        throw 'Port 8201 is used by another application. Close it and try again.'
    }
    if ($LauncherMode) { Write-StartupStatus 'ready' 'Connecting to the running workspace.' | Out-Host }
    else { Start-Process $settingsUrl }
    return $true
}

try {
    Set-Location -LiteralPath $sourceRoot
    if (-not $LauncherMode) { $Host.UI.RawUI.WindowTitle = 'Comfy Workshop' }
    Write-StartupStatus 'checking' 'Checking the startup environment.'
    if (Open-RunningSettings) { exit 0 }
    try {
        $ownsMutex = $mutex.WaitOne(0)
    } catch [System.Threading.AbandonedMutexException] {
        $ownsMutex = $true
    }
    if (-not $ownsMutex) {
        Write-StartupStatus 'waiting' 'Another launcher is preparing the workspace. Please wait.'
        exit 0
    }

    $client = New-Object Net.Sockets.TcpClient
    try {
        $portInUse = $client.ConnectAsync('127.0.0.1', 8201).Wait(500)
    } catch {
        $portInUse = $false
    } finally {
        $client.Dispose()
    }
    if ($portInUse) {
        throw 'Port 8201 is busy, but Comfy Workshop is unavailable. Check the running application.'
    }

    $python = Join-Path $sourceRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) {
        Write-StartupStatus 'runtime' 'Preparing the Python environment for the first launch.'
        $uv = Join-Path $sourceRoot '.tools\uv-0.11.8\uv.exe'
        if (-not (Test-Path -LiteralPath $uv)) {
            $command = Get-Command uv.exe -ErrorAction SilentlyContinue
            if (-not $command) { throw 'uv is required to prepare Python. Install uv and try again.' }
            $uv = $command.Source
        }
        & $uv sync
        if ($LASTEXITCODE -ne 0) { throw 'Python environment setup failed. See the log above.' }
    }

    Write-StartupStatus 'ui' 'Preparing the workspace UI.'
    $uiRoot = Join-Path $sourceRoot 'settings_ui'
    $index = Join-Path $uiRoot 'dist\index.html'
    $buildNeeded = -not (Test-Path -LiteralPath $index)
    if (-not $buildNeeded) {
        $builtAt = (Get-Item -LiteralPath $index).LastWriteTimeUtc
        $inputs = @(Get-ChildItem -LiteralPath (Join-Path $uiRoot 'src'), (Join-Path $uiRoot 'public') -Recurse -File)
        $inputs += @(Get-ChildItem -LiteralPath $uiRoot -File | Where-Object { $_.Extension -in '.json', '.ts', '.html' })
        $buildNeeded = @($inputs | Where-Object { $_.LastWriteTimeUtc -gt $builtAt }).Count -gt 0
    }
    if ($buildNeeded) {
        $npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
        if (-not $npm) { throw 'Node.js is required to build the UI. Install Node.js and try again.' }
        Push-Location -LiteralPath $uiRoot
        try {
            $installedLock = Join-Path $uiRoot 'node_modules\.package-lock.json'
            if (-not (Test-Path -LiteralPath $installedLock) -or
                (Get-Item -LiteralPath 'package-lock.json').LastWriteTimeUtc -gt (Get-Item -LiteralPath $installedLock).LastWriteTimeUtc) {
                Write-StartupStatus 'ui-dependencies' 'Installing UI dependencies.'
                & $npm.Source ci
                if ($LASTEXITCODE -ne 0) { throw 'UI dependency installation failed. See the log above.' }
            }
            Write-StartupStatus 'ui-build' 'Building the updated workspace UI.'
            & $npm.Source run build
            if ($LASTEXITCODE -ne 0) { throw 'UI build failed. See the log above.' }
        } finally {
            Pop-Location
        }
    }

    Write-Host ''
    Write-StartupStatus 'server' 'Connecting to the workspace server.'
    Write-Host "URL: $settingsUrl"
    if (-not $LauncherMode) { Write-Host 'Keep this window open while using the workspace. Use the shutdown button or Ctrl+C to stop.' }
    Write-Host ''
    if ($LauncherMode) { & $python -u (Join-Path $sourceRoot 'settings_server.py') --no-browser }
    else { & $python -u (Join-Path $sourceRoot 'settings_server.py') }
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) { throw "Comfy Workshop exited with code $exitCode. See the log above." }
} catch {
    Write-Host "`n[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    if ($LauncherMode) { Write-StartupStatus 'error' $_.Exception.Message }
    else { Read-Host 'Press Enter to close' | Out-Null }
    $exitCode = 1
} finally {
    if ($ownsMutex) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
exit $exitCode
