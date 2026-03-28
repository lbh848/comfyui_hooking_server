@echo off
chcp 65001 >nul
echo ========================================
echo 8188 포트 프로세스 종료
echo ========================================
echo.

REM 8188 포트를 사용하는 프로세스 찾기
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8188 ^| find "LISTENING"') do (
    echo [정보] 8188 포트 사용 중인 프로세스 PID: %%a
    taskkill /F /PID %%a
    if errorlevel 1 (
        echo [실패] 프로세스(PID %%a) 종료에 실패했습니다. 관리자 권한이 필요할 수 있습니다.
    ) else (
        echo [성공] 프로세스(PID %%a)가 성공적으로 종료되었습니다.
    )
)

echo.
echo 작업 완료.
pause
