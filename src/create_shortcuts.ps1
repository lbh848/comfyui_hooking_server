$ErrorActionPreference = 'Stop'
$installRoot = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $PSScriptRoot 'run_en.bat'
if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    throw "Launcher not found: $launcher"
}
foreach ($name in @('.git', '.github', '.gitignore', '.gitattributes')) {
    $metadata = Join-Path $installRoot $name
    if (Test-Path -LiteralPath $metadata) {
        $item = Get-Item -LiteralPath $metadata -Force
        $item.Attributes = $item.Attributes -bor [System.IO.FileAttributes]::Hidden
    }
}
foreach ($directory in @('config', 'data')) {
    [System.IO.Directory]::CreateDirectory((Join-Path $installRoot $directory)) | Out-Null
}
$shell = New-Object -ComObject WScript.Shell
try {
    $applications = [ordered]@{
        '삽화' = 'illustration'
        '에셋' = 'asset'
        '영상' = 'video'
    }
    foreach ($application in $applications.GetEnumerator()) {
        $shortcut = $shell.CreateShortcut((Join-Path $installRoot ($application.Key + '.lnk')))
        $shortcut.TargetPath = $launcher
        $shortcut.Arguments = $application.Value
        $shortcut.WorkingDirectory = $PSScriptRoot
        $shortcut.Description = 'SoyaComfy - ' + $application.Key
        $shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,71"
        $shortcut.Save()
        [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($shortcut) | Out-Null
    }
} finally {
    [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($shell) | Out-Null
}
Write-Output "Created 3 application shortcuts in: $installRoot"
