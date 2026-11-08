$ErrorActionPreference = 'Stop'
$installRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
$compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if (-not (Test-Path -LiteralPath $compiler)) {
    $compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework\v4.0.30319\csc.exe'
}
if (-not (Test-Path -LiteralPath $compiler)) {
    throw 'The .NET Framework C# compiler is required to build the Windows launcher.'
}
# Keep the script ASCII-only, including the localized executable name.
$launcherName = 'Comfy' + [char]0xACF5 + [char]0xBC29
$output = Join-Path $installRoot ($launcherName + '.exe')
$buildRoot = Join-Path $installRoot 'output\launcher-build'
[IO.Directory]::CreateDirectory($buildRoot) | Out-Null
$staged = Join-Path $buildRoot ($launcherName + '.exe')
$previous = Join-Path $buildRoot ("$launcherName.previous-{0}.exe" -f [DateTime]::UtcNow.Ticks)
& $compiler /nologo /target:winexe /platform:anycpu /optimize+ /codepage:65001 /reference:System.Windows.Forms.dll /reference:System.Drawing.dll /reference:System.Web.Extensions.dll "/win32manifest:$(Join-Path $PSScriptRoot 'launcher.manifest')" "/out:$staged" (Join-Path $PSScriptRoot 'SettingsLauncher.cs') (Join-Path $PSScriptRoot 'StartupWindow.cs')
if ($LASTEXITCODE -ne 0) { throw 'Windows launcher build failed.' }
foreach ($target in @($output, $staged, $previous)) {
    if (-not [IO.Path]::GetFullPath($target).StartsWith($installRoot.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Launcher build destination is outside the installation.'
    }
}
# Windows can rename a running executable; keep that version until it exits.
if (Test-Path -LiteralPath $output) { Move-Item -LiteralPath $output -Destination $previous }
try { Move-Item -LiteralPath $staged -Destination $output }
catch {
    if ((Test-Path -LiteralPath $previous) -and -not (Test-Path -LiteralPath $output)) {
        Move-Item -LiteralPath $previous -Destination $output
    }
    throw
}
Write-Output "Built: $output"
