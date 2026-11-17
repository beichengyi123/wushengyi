@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================
echo  sign + voice main program (1.py)
echo  camera window will open. press ESC / q in it.
echo ========================================
D:\python\python.exe -u 1.py
echo.
echo [main program stopped] press any key to close
pause >nul
