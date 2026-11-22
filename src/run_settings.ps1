param([switch]$LauncherMode)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8:backslashreplace'
$env:PYTHONNOUSERSITE = '1'
Remove-Item Env:PYTHONHOME, Env:PYTHONPATH, Env:VIRTUAL_ENV, Env:CONDA_PREFIX -ErrorAction SilentlyContinue
$sourceRoot = $PSScriptRoot
$workshopName = 'Comfy' + [char]0xACF5 + [char]0xBC29
$settingsUrl = 'http://127.0.0.1:8201/setting'
$uvVersion = '0.11.8'
$minimumUvVersion = [version]$uvVersion
$mutex = New-Object System.Threading.Mutex($false, 'Local\ComfyWorkshopSettingsLauncher8201')
$ownsMutex = $false
$exitCode = 0

function Write-StartupStatus([string]$Stage, [string]$Message) {
    if ($LauncherMode) { Write-Output "COMFY_STARTUP|$Stage|$Message" }
    else { Write-Host $Message }
}

function Add-PathDirectory([string]$Directory, [switch]$Prepend) {
    if ([string]::IsNullOrWhiteSpace($Directory) -or -not (Test-Path -LiteralPath $Directory -PathType Container)) {
        return
    }
    $normalized = $Directory.TrimEnd('\')
    $entries = @($env:PATH -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    $filtered = @($entries | Where-Object { $_.TrimEnd('\') -ine $normalized })
    if ($Prepend) {
        $env:PATH = (@($normalized) + $filtered) -join ';'
    } elseif ($filtered.Count -eq $entries.Count) {
        $env:PATH = ($entries + @($normalized)) -join ';'
    }
}

function Add-StandardToolPaths {
    if (-not (Get-Command git.exe -ErrorAction SilentlyContinue)) {
        $gitDirs = New-Object System.Collections.Generic.List[string]
        if ($env:LOCALAPPDATA) { $gitDirs.Add((Join-Path $env:LOCALAPPDATA 'Programs\Git\cmd')) }
        if ($env:ProgramFiles) { $gitDirs.Add((Join-Path $env:ProgramFiles 'Git\cmd')) }
        if ($env:ProgramW6432) { $gitDirs.Add((Join-Path $env:ProgramW6432 'Git\cmd')) }
        foreach ($dir in $gitDirs) {
            if (Test-Path -LiteralPath (Join-Path $dir 'git.exe') -PathType Leaf) {
                Add-PathDirectory $dir
                break
            }
        }
    }
}

function Test-UvCandidate([string]$Executable) {
    if ([string]::IsNullOrWhiteSpace($Executable) -or -not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
        return $false
    }
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = (& $Executable --version 2>$null | Out-String)
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($output)) {
            return $false
        }
        if ($output -match '\buv\s+(\d+)\.(\d+)\.(\d+)\b') {
            return ([version]"$($Matches[1]).$($Matches[2]).$($Matches[3])") -ge $minimumUvVersion
        }
        return $false
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $previous
    }
}

function Find-UsableUv {
    $candidates = New-Object System.Collections.Generic.List[string]
    $candidates.Add((Join-Path $sourceRoot ".tools\uv-$uvVersion\uv.exe"))
    $toolsDir = Join-Path $sourceRoot '.tools'
    if (Test-Path -LiteralPath $toolsDir -PathType Container) {
        $localTools = @(Get-ChildItem -LiteralPath $toolsDir -Directory -Filter 'uv-*' -ErrorAction SilentlyContinue | Sort-Object Name -Descending)
        foreach ($tool in $localTools) {
            $candidates.Add((Join-Path $tool.FullName 'uv.exe'))
        }
    }
    $command = Get-Command uv.exe -ErrorAction SilentlyContinue
    if ($command -and $command.Source) {
        $candidates.Add($command.Source)
    }
    if ($env:USERPROFILE) {
        $candidates.Add((Join-Path $env:USERPROFILE '.local\bin\uv.exe'))
    }
    if ($env:LOCALAPPDATA) {
        $candidates.Add((Join-Path $env:LOCALAPPDATA 'Programs\uv\uv.exe'))
    }
    $seen = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($candidate in $candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate) -or -not $seen.Add($candidate)) {
            continue
        }
        if (Test-UvCandidate $candidate) {
            Add-PathDirectory (Split-Path -Parent $candidate) -Prepend
            return $candidate
        }
    }
    return $null
}

function Install-ProjectUv {
    $uvToolDir = Join-Path $sourceRoot ".tools\uv-$uvVersion"
    $localUv = Join-Path $uvToolDir 'uv.exe'
    Write-StartupStatus 'runtime' "Installing project-local uv $uvVersion."
    $powershell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $installCommand = "[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12; irm 'https://astral.sh/uv/$uvVersion/install.ps1' | iex"
    $prevInstallDir = $env:UV_INSTALL_DIR
    $prevNoModify = $env:UV_NO_MODIFY_PATH
    try {
        $env:UV_INSTALL_DIR = $uvToolDir
        $env:UV_NO_MODIFY_PATH = '1'
        & $powershell -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command $installCommand
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install project-local uv $uvVersion. Check the network connection and try again."
        }
    } finally {
        if ($null -eq $prevInstallDir) { Remove-Item Env:UV_INSTALL_DIR -ErrorAction SilentlyContinue }
        else { $env:UV_INSTALL_DIR = $prevInstallDir }
        if ($null -eq $prevNoModify) { Remove-Item Env:UV_NO_MODIFY_PATH -ErrorAction SilentlyContinue }
        else { $env:UV_NO_MODIFY_PATH = $prevNoModify }
    }
    if (-not (Test-UvCandidate $localUv)) {
        throw "uv installer completed, but a usable uv.exe was not found at $localUv."
    }
    Add-PathDirectory $uvToolDir -Prepend
    return $localUv
}

function Get-PythonEnvStamp([string]$Root) {
    $pyproject = Join-Path $Root 'pyproject.toml'
    if (-not (Test-Path -LiteralPath $pyproject -PathType Leaf)) {
        return $null
    }
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        $parts = New-Object System.Collections.Generic.List[string]
        foreach ($name in @('pyproject.toml', 'uv.lock', '.python-version')) {
            $filePath = Join-Path $Root $name
            if (Test-Path -LiteralPath $filePath -PathType Leaf) {
                $bytes = [System.IO.File]::ReadAllBytes($filePath)
                $hash = ([System.BitConverter]::ToString($sha256.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant()
                $parts.Add("$name=$hash")
            }
        }
        return ($parts -join ';')
    } finally {
        $sha256.Dispose()
    }
}

function Test-VenvPython([string]$PythonPath, [bool]$CheckPackages) {
    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        return $false
    }
    $checkCode = if ($CheckPackages) {
        'import sys, aiohttp; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'
    } else {
        'import sys; raise SystemExit(0)'
    }
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $PythonPath -c $checkCode 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $previous
    }
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

    Add-StandardToolPaths
    $uv = Find-UsableUv
    $python = Join-Path $sourceRoot '.venv\Scripts\python.exe'
    $stampFile = Join-Path $sourceRoot '.venv\.comfy-sync-stamp'
    $expectedStamp = Get-PythonEnvStamp $sourceRoot

    $pythonExists = Test-Path -LiteralPath $python -PathType Leaf
    $pythonHealthy = $false
    if ($pythonExists) {
        $pythonHealthy = Test-VenvPython $python ([bool]$expectedStamp)
    }
    $stampMatches = $true
    if ($expectedStamp) {
        $savedStamp = ''
        if (Test-Path -LiteralPath $stampFile -PathType Leaf) {
            $rawStamp = Get-Content -LiteralPath $stampFile -Raw -ErrorAction SilentlyContinue
            if ($null -ne $rawStamp) { $savedStamp = $rawStamp.Trim() }
        }
        $stampMatches = ($savedStamp -eq $expectedStamp)
    }

    if (-not $pythonHealthy -or -not $stampMatches) {
        if ($pythonExists) {
            Write-StartupStatus 'runtime' 'Updating the Python environment.'
        } else {
            Write-StartupStatus 'runtime' 'Preparing the Python environment for the first launch.'
        }
        if (-not $uv) {
            $uv = Install-ProjectUv
        }
        $syncArgs = @('sync')
        if (Test-Path -LiteralPath (Join-Path $sourceRoot 'uv.lock') -PathType Leaf) {
            $syncArgs += '--frozen'
        }
        & $uv @syncArgs
        if ($LASTEXITCODE -ne 0) { throw 'Python environment setup failed. See the log above.' }
        if (-not (Test-VenvPython $python ([bool]$expectedStamp))) {
            Write-StartupStatus 'runtime' 'Repairing the Python environment.'
            & $uv @syncArgs --reinstall
            if ($LASTEXITCODE -ne 0 -or -not (Test-VenvPython $python ([bool]$expectedStamp))) {
                throw 'Python environment verification failed after setup. See the log above.'
            }
        }
        if ($expectedStamp) {
            $syncedStamp = Get-PythonEnvStamp $sourceRoot
            if ($syncedStamp) {
                [System.IO.File]::WriteAllText($stampFile, $syncedStamp, [System.Text.Encoding]::ASCII)
            }
        }
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
        if (-not $npm) {
            $nodeDirs = New-Object System.Collections.Generic.List[string]
            if ($env:ProgramFiles) { $nodeDirs.Add((Join-Path $env:ProgramFiles 'nodejs')) }
            if ($env:ProgramW6432) { $nodeDirs.Add((Join-Path $env:ProgramW6432 'nodejs')) }
            if ($env:LOCALAPPDATA) { $nodeDirs.Add((Join-Path $env:LOCALAPPDATA 'Programs\nodejs')) }
            foreach ($dir in $nodeDirs) {
                if (Test-Path -LiteralPath (Join-Path $dir 'npm.cmd') -PathType Leaf) {
                    Add-PathDirectory $dir
                    break
                }
            }
            $npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
        }
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
