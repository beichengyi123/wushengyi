@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem ==========================================================
rem  停止全部后台服务(后端 5000 + 智能体 8080)
rem  桌面端软件关闭后服务仍在后台跑, 想彻底停掉就双击本文件
rem ==========================================================

echo ==========================================================
echo   stop services on port 5000 / 8080
echo ==========================================================

for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":5000 " ^| findstr LISTENING') do (
    echo   [backend] kill pid %%p
    taskkill /PID %%p /F >nul 2>&1
)
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8080 " ^| findstr LISTENING') do (
    echo   [agent]   kill pid %%p
    taskkill /PID %%p /F >nul 2>&1
)

echo.
echo   done. press any key to close
pause >nul
