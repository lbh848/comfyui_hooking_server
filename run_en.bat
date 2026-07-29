@echo off
setlocal
cd /d %~dp0
chcp 65001 >nul 2>&1

set "UV_VERSION=0.11.8"
set "UV_TOOL_DIR=%CD%\.tools\uv-%UV_VERSION%"
set "UV_EXE=%UV_TOOL_DIR%\uv.exe"

echo [1/6] Checking project-local uv %UV_VERSION%...
if not exist "%UV_EXE%" (
    echo       Project-local uv not found. Installing pinned version...
    set "UV_INSTALL_DIR=%UV_TOOL_DIR%"
    set "UV_NO_MODIFY_PATH=1"
    powershell -NoProfile -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/%UV_VERSION%/install.ps1 | iex"
    if errorlevel 1 (
        echo [ERROR] Failed to install project-local uv %UV_VERSION%.
        echo         Check the network connection and PowerShell policy.
        goto :end
    )
    if not exist "%UV_EXE%" (
        echo [ERROR] uv installer completed but uv.exe is missing:
        echo         %UV_EXE%
        goto :end
    )
    echo       Project-local uv installed.
) else (
    echo       Project-local uv OK.
)
set "UV_INSTALL_DIR="
set "UV_NO_MODIFY_PATH="
set "PATH=%UV_TOOL_DIR%;%PATH%"

echo [2/6] Checking system Git...
call :ensure_git
if errorlevel 1 goto :end

:: Cleanup legacy venv (one-time)
if exist "venv" (
    echo       Cleaning up legacy venv folder...
    rmdir /s /q "venv" 2>nul
)

:: Python 3.12 + packages
echo [3/6] Setting up Python 3.12 environment...
"%UV_EXE%" sync
if errorlevel 1 (
    echo [ERROR] Failed to set up environment. Check network connection.
    goto :end
)

:: Windows GPU 가속: onnxruntime-directml 은 pyproject.toml 에 선언되어
:: uv sync 로 함께 설치됨(DirectML = NVIDIA/AMD/Intel 공통 GPU, 없으면 자동 CPU 폴백).
:: 별도 runtime 설치 분기는 의존성 충돌(onnxruntime vs -directml)을 유발해 제거.

echo [4/6] Checking model files...
"%UV_EXE%" run --no-sync python ensure_models.py
if errorlevel 1 (
    echo [ERROR] Failed to prepare model files. Check the messages above and network connection.
    goto :end
)

echo [5/6] Packages and models ready.

:: Create required folders
echo [6/6] Creating folders...
if not exist "workflow" mkdir workflow
if not exist "current_work" mkdir current_work
if not exist "workflow_backup" mkdir workflow_backup
if not exist "frontend" mkdir frontend
if not exist "logs" mkdir logs
if not exist "mode_workflow" mkdir mode_workflow
if not exist "asset_data" mkdir asset_data
if not exist "auto_complete" mkdir auto_complete
if not exist "pose_data" mkdir pose_data
if not exist "chain_presets" mkdir chain_presets
if not exist "key" mkdir key

echo.
echo ============================================
echo   ComfyUI Proxy Server Start (port 8189)
echo   Frontend: http://127.0.0.1:8189/
echo ============================================
echo.
"%UV_EXE%" run --no-sync python server.py

:end
pause
endlocal
goto :eof

:ensure_git
where git.exe >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%G in ('git --version 2^>nul') do echo       %%G
    exit /b 0
)
call :add_standard_git_to_path
where git.exe >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%G in ('git --version 2^>nul') do echo       %%G
    exit /b 0
)

echo       Git not found. Installing Git for Windows with winget...
where winget.exe >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git is missing and Windows Package Manager ^(winget^) is unavailable.
    echo         Install or repair Microsoft App Installer, then run this manager again.
    exit /b 1
)

winget install --id Git.Git --exact --source winget --silent --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo [ERROR] Git for Windows installation failed.
    echo         Review the winget message above and run this manager again.
    exit /b 1
)

:: winget updates the persistent PATH, but this batch needs Git immediately.
call :add_standard_git_to_path

where git.exe >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git was installed but git.exe was not found in a standard location.
    echo         Restart Windows or add Git\cmd to PATH, then run this manager again.
    exit /b 1
)
for /f "delims=" %%G in ('git --version 2^>nul') do echo       Installed %%G
exit /b 0

:add_standard_git_to_path
if exist "%LOCALAPPDATA%\Programs\Git\cmd\git.exe" set "PATH=%LOCALAPPDATA%\Programs\Git\cmd;%PATH%"
if exist "%ProgramFiles%\Git\cmd\git.exe" set "PATH=%ProgramFiles%\Git\cmd;%PATH%"
if defined ProgramW6432 if exist "%ProgramW6432%\Git\cmd\git.exe" set "PATH=%ProgramW6432%\Git\cmd;%PATH%"
exit /b 0
