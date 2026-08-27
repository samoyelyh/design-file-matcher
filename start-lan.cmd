@echo off
setlocal
set "DESIGN_MATCHER_LAN_PORT=8088"
if not defined HF_ENDPOINT set "HF_ENDPOINT=https://hf-mirror.com"
echo.
echo Design File Matcher - LAN mode
echo Local:   http://127.0.0.1:%DESIGN_MATCHER_LAN_PORT%
echo LAN:     http://192.168.0.31:%DESIGN_MATCHER_LAN_PORT%
echo Model:   %HF_ENDPOINT% ^(Nanjing HF mirror^)
echo.
echo Keep this window open while other computers are using the service.
echo Press Ctrl+C to stop.
echo.
"%~dp0.venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port %DESIGN_MATCHER_LAN_PORT%
