@echo off
cd /d "%~dp0"
echo Building Git Auto Updater...
pip install pyinstaller
pyinstaller --onefile --noconsole --name "GitUpdater" --paths .. --distpath .. updater.py
echo.
if exist "..\GitUpdater.exe" (
    echo Build complete! Output: ..\GitUpdater.exe
) else (
    echo Build failed!
)
pause
