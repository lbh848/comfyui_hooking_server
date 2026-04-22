@echo off
setlocal
cd /d %~dp0
chcp 65001 >nul 2>&1

:: Python 설치 확인
echo [1/4] Python 확인 중...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python이 설치되어 있지 않거나 PATH에 등록되지 않았습니다.
    echo         Python 3.10+ 을 설치해주세요: https://www.python.org/downloads/
    echo         설치 시 "Add Python to PATH"를 체크하세요.
    pause
    exit /b 1
)

:: venv가 없으면 생성
if not exist "venv\Scripts\activate.bat" (
    echo [2/4] 가상환경 생성 중... (최초 실행 시 1~3분 소요)
    python -m venv venv
    if not exist "venv\Scripts\activate.bat" (
        echo [ERROR] 가상환경 생성에 실패했습니다.
        echo         Python 버전을 확인하고 venv 폴더를 삭제 후 다시 시도하세요.
        pause
        exit /b 1
    )
    echo       가상환경 생성 완료.
) else (
    echo [2/4] 가상환경 확인 완료.
)

:: 패키지 설치/업데이트
echo [3/4] 패키지 설치 중... (최초 실행 시 수 분 소요될 수 있습니다)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] 패키지 설치에 실패했습니다. 네트워크 연결을 확인하세요.
    pause
    exit /b 1
)

:: 필요한 폴더 생성
echo [4/4] 폴더 정리 중...
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
echo.
echo ============================================
echo   ComfyUI Proxy Server 시작 (port 8189)
echo   Frontend: http://127.0.0.1:8189/
echo ============================================
echo.
python server.py

pause
