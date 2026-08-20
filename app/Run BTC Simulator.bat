@echo off
setlocal
title BTC Execution Simulator

cd /d "%~dp0.."

echo ============================================================
echo  BTC Execution Simulator
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH.
    echo Install Python 3.10 or newer, then double-click this file again.
    echo.
    pause
    exit /b 1
)

python -c "import fastapi, uvicorn, yaml, numpy" >nul 2>nul
if errorlevel 1 (
    echo Installing missing project dependencies...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Dependency installation failed. Check your internet connection
        echo and the error messages above.
        pause
        exit /b 1
    )
)

echo Starting the application at http://127.0.0.1:8000
echo Your browser will open automatically.
echo Press Ctrl+C in this window to stop the application.
echo.

start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:8000'"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

echo.
echo The application has stopped.
pause

