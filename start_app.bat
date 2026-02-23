@echo off
title ManTacAi Launcher
color 0A

echo ===================================================
echo       MANIAC TACTIC DETECTOR (ManTacAi)
echo           System Initialization...
echo ===================================================
echo.

echo [1/2] Launching Backend Neural Engine (Port 8000)...
start "ManTacAi Backend" cmd /k "python backend/main.py"

echo [2/2] Launching Holographic Frontend (Port 5173)...
cd frontend
start "ManTacAi Interface" cmd /k "npm run dev"

echo.
echo ===================================================
echo       SYSTEM ONLINE. ACCESS VIA BROWSER.
echo       http://localhost:5173
echo ===================================================
echo.
pause
