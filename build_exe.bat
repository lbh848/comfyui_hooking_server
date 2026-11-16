@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "BUILD_SCRIPT=%~dp0src\packaging\windows\build_launcher.ps1"
set "BUILD_EXIT_CODE=1"

echo Building the Comfy Workshop launcher...
echo.
if not exist "%BUILD_SCRIPT%" (
    echo [ERROR] Build script not found: "%BUILD_SCRIPT%"
    echo Keep this BAT file in the project folder, next to src.
    goto :finish
)

"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%BUILD_SCRIPT%"
set "BUILD_EXIT_CODE=%ERRORLEVEL%"
echo.
if "%BUILD_EXIT_CODE%"=="0" (
    echo [OK] Build complete. The EXE is in this project folder.
) else (
    echo [ERROR] Build failed. See the error above.
)

:finish
echo.
if /i not "%~1"=="--no-pause" pause
exit /b %BUILD_EXIT_CODE%
