@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install_lan_autostart.ps1"
exit /b %errorlevel%
