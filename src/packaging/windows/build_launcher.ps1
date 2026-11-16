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
$buildRoot = Join-Path $installRoot 'data\backups\launcher-build'
[IO.Directory]::CreateDirectory($buildRoot) | Out-Null
$staged = Join-Path $buildRoot ($launcherName + '.exe')
$previous = Join-Path $buildRoot ("$launcherName.previous-{0}.exe" -f [DateTime]::UtcNow.Ticks)
& $compiler /nologo /target:winexe /platform:anycpu /optimize+ /codepage:65001 /reference:System.Windows.Forms.dll /reference:System.Drawing.dll /reference:System.Web.Extensions.dll "/win32manifest:$(Join-Path $PSScriptRoot 'launcher.manifest')" "/win32icon:$(Join-Path $PSScriptRoot 'launcher.ico')" "/out:$staged" (Join-Path $PSScriptRoot 'SettingsLauncher.cs') (Join-Path $PSScriptRoot 'StartupWindow.cs')
if ($LASTEXITCODE -ne 0) { throw 'Windows launcher build failed.' }
if (-not (Test-Path -LiteralPath $staged -PathType Leaf)) {
    throw 'The compiled launcher is missing. Check your antivirus protection history.'
}
foreach ($target in @($output, $staged, $previous)) {
    if (-not [IO.Path]::GetFullPath($target).StartsWith($installRoot.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Launcher build destination is outside the installation.'
    }
}
# Windows can rename a running executable; keep that version until it exits.
if (Test-Path -LiteralPath $output) { Move-Item -LiteralPath $output -Destination $previous }
try {
    Move-Item -LiteralPath $staged -Destination $output
    if (-not (Test-Path -LiteralPath $output -PathType Leaf)) {
        throw 'The built launcher is missing. Check your antivirus protection history.'
    }
}
catch {
    if ((Test-Path -LiteralPath $previous) -and -not (Test-Path -LiteralPath $output)) {
        Move-Item -LiteralPath $previous -Destination $output
    }
    throw
}
# Invalidate the Shell's cached icon after replacing an EXE at the same path.
# https://learn.microsoft.com/windows/win32/api/shlobj_core/nf-shlobj_core-shchangenotify
try {
    if (-not ('ComfyLauncherShell' -as [type])) {
        Add-Type -TypeDefinition @'
using System.Runtime.InteropServices;
public static class ComfyLauncherShell {
    [DllImport("shell32.dll", CharSet = CharSet.Unicode, ExactSpelling = true)]
    public static extern void SHChangeNotify(int eventId, uint flags, string item1, string item2);
}
'@
    }
    # SHCNE_ASSOCCHANGED + SHCNF_IDLIST + SHCNF_FLUSH clears cached icons.
    [ComfyLauncherShell]::SHChangeNotify(0x08000000, 0x1000, $null, $null)
    # SHCNE_UPDATEITEM + SHCNF_PATHW + SHCNF_FLUSH refreshes this file.
    [ComfyLauncherShell]::SHChangeNotify(0x2000, 0x1005, $output, $null)
}
catch { Write-Warning "The EXE was built, but its Explorer icon could not be refreshed: $_" }
Write-Output "Built: $output"
