@echo off
setlocal
cd /d %~dp0
chcp 65001 >nul 2>&1

echo [1/4] Python Checking...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo         Install Python 3.10+: https://www.python.org/downloads/
    echo         Check "Add Python to PATH" during installation.
    goto :end
)

if not exist "venv\Scripts\activate.bat" (
    echo [2/4] Creating venv...
    if exist "venv" (
        rmdir /s /q "venv" 2>nul
    )
    python -m venv venv --clear
    if not exist "venv\Scripts\activate.bat" (
        echo [ERROR] Failed to create venv.
        echo         Delete venv folder manually and retry.
        goto :end
    )
) else (
    echo [2/4] venv OK.
)

echo [3/4] Installing packages...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install packages. Check network connection.
    goto :end
)

echo [4/4] Creating folders...
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
python server.py

:end
pause
