@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================
echo  MQTT bridge  (needs 1.py + backend running)
echo  keep this window open. Ctrl+C to stop.
echo ========================================
D:\python\python.exe -u bridge_mqtt.py
echo.
echo [bridge stopped] press any key to close
pause >nul
