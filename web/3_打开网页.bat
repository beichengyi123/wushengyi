@echo off
chcp 65001 >nul
rem open the dashboard in default browser (backend must be running first)
start "" http://127.0.0.1:5000
