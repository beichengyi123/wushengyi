@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem ==========================================================
rem  一键启动: Flask 中控后端(5000) + 智能体服务(8080) + 识别主程序(手语+语音) + 打开网页
rem  三个黑窗口都不要关, 关掉对应服务就停了
rem  注意: 语音要"出字到屏幕", 必须有识别主程序(1.py), 它会弹摄像头窗口
rem ==========================================================

if "%JAVA_HOME%"=="" set "JAVA_HOME=D:\dev\jdk17"

echo ==========================================================
echo   WuShengYi  -  start all services
echo ----------------------------------------------------------
echo   [1/4] Flask backend     http://127.0.0.1:5000
echo   [2/4] Agent service     http://127.0.0.1:8080
echo   [3/4] Recognizer        camera window (sign + voice)
echo   [4/4] open browser
echo ==========================================================

netstat -ano | findstr ":5000 " | findstr LISTENING >nul
if errorlevel 1 (
    start "WSY-backend" /d "%~dp0web" "D:\python\python.exe" app.py
    echo   backend starting...
) else (
    echo   [skip] port 5000 already running
)

netstat -ano | findstr ":8080 " | findstr LISTENING >nul
if errorlevel 1 (
    start "WSY-agent" /d "%~dp0agent" "%JAVA_HOME%\bin\java.exe" -jar target\wusheng-agent.jar
    echo   agent starting... ^(takes about 8s^)
) else (
    echo   [skip] port 8080 already running
)

rem [3/4] 识别主程序: 手语识别 + 语音转文字, 屏幕出字靠它
start "WSY-recognize" /d "%~dp0" "D:\python\python.exe" -u 1.py
echo   recognizer starting... ^(camera window^)

echo.
echo   waiting for services to be ready...
timeout /t 8 /nobreak >nul
start "" http://127.0.0.1:5000
echo   browser opened. This window can be closed.
timeout /t 3 /nobreak >nul
