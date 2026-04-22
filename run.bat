@echo off
setlocal
cd /d %~dp0

:: venv가 없으면 생성
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

:: 패키지 설치/업데이트
echo [INFO] Installing requirements...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

:: 필요한 폴더 생성
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

:: 서버 실행
echo [INFO] Starting ComfyUI Proxy Server on port 8189...
echo [INFO] Frontend: http://127.0.0.1:8189/
python server.py

pause
