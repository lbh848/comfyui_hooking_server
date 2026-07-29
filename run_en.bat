@echo off
setlocal
cd /d %~dp0
chcp 65001 >nul 2>&1

set "UV_VERSION=0.11.8"
set "UV_TOOL_DIR=%CD%\.tools\uv-%UV_VERSION%"
set "UV_EXE=%UV_TOOL_DIR%\uv.exe"

echo [1/5] Checking project-local uv %UV_VERSION%...
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

:: Cleanup legacy venv (one-time)
if exist "venv" (
    echo       Cleaning up legacy venv folder...
    rmdir /s /q "venv" 2>nul
)

:: Python 3.12 + packages
echo [2/5] Setting up Python 3.12 environment...
"%UV_EXE%" sync
if errorlevel 1 (
    echo [ERROR] Failed to set up environment. Check network connection.
    goto :end
)

:: Windows GPU 가속: onnxruntime-directml 은 pyproject.toml 에 선언되어
:: uv sync 로 함께 설치됨(DirectML = NVIDIA/AMD/Intel 공통 GPU, 없으면 자동 CPU 폴백).
:: 별도 runtime 설치 분기는 의존성 충돌(onnxruntime vs -directml)을 유발해 제거.

echo [3/5] Checking model files...
"%UV_EXE%" run --no-sync python ensure_models.py
if errorlevel 1 (
    echo [ERROR] Failed to prepare model files. Check the messages above and network connection.
    goto :end
)

echo [4/5] Packages and models ready.

:: Create required folders
echo [5/5] Creating folders...
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
