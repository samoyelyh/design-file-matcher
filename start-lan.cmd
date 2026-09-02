@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_lan_background.ps1"
exit /b %errorlevel%
