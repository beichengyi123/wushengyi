@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem 环境变量兜底(正常情况下系统里已配好 JAVA_HOME)
if "%JAVA_HOME%"=="" set "JAVA_HOME=D:\dev\jdk17"
set "PATH=%JAVA_HOME%\bin;%PATH%"

echo ==========================================================
echo   无声有译 · 智能体服务 (SpringBoot + LangChain4j)
echo   接口: http://127.0.0.1:8080/api/agent/compose
echo   探活: http://127.0.0.1:8080/api/agent/health
echo   日志窗口别关, 关掉程序就停了 (Ctrl+C 停止)
echo ==========================================================

if exist "target\wusheng-agent.jar" (
    "%JAVA_HOME%\bin\java.exe" -jar "target\wusheng-agent.jar"
) else (
    echo 未找到 target\wusheng-agent.jar, 先执行打包:
    echo    mvn -DskipTests package
    echo 或双击 build_agent.bat
)
echo.
echo [智能体已停止] 按任意键关闭
pause >nul
