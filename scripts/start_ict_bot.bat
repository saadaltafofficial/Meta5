@echo off
REM ICT Forex Trading Bot Startup Script
REM This script starts the MT5 platform and the ICT trading bot

echo ===============================================
echo ICT Forex Trading Bot - VPS Startup Script
echo ===============================================
echo.

REM Set paths
set ROOT_DIR=%~dp0..
set PYTHON_PATH=python
set MT5_PATH="C:\Program Files\MetaTrader 5\terminal64.exe"
set MAIN_SCRIPT=%ROOT_DIR%\main.py

echo Starting MetaTrader 5...
start "" %MT5_PATH%

REM Wait for MT5 to initialize
echo Waiting for MT5 to initialize (30 seconds)...
timeout /t 30 /nobreak > nul

echo Starting ICT Trading Bot...
cd %ROOT_DIR%

:start_bot
echo [%date% %time%] Starting ICT Trading Bot...
%PYTHON_PATH% %MAIN_SCRIPT%

REM If the bot crashes, restart it
echo [%date% %time%] Bot exited. Restarting in 10 seconds...
timeout /t 10 /nobreak > nul
goto start_bot
