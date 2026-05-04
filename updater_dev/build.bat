@echo off
echo Building Git Auto Updater...
pip install pyinstaller
pyinstaller --onefile --noconsole --name "GitUpdater" updater.py
echo.
if exist "dist\GitUpdater.exe" (
    echo Build complete! Output: dist\GitUpdater.exe
) else (
    echo Build failed!
)
pause
