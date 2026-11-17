@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================
echo  Flask backend  http://127.0.0.1:5000
echo  keep this window open. Ctrl+C to stop.
echo ========================================
D:\python\python.exe app.py
echo.
echo [backend stopped] press any key to close
pause >nul
