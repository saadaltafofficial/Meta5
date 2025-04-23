@echo off
echo ===============================================
echo ICT Trading Bot VPS Startup Script
echo ===============================================

echo Starting MetaTrader 5...
start "" "C:\Program Files\MetaTrader 5\terminal64.exe" /portable

echo Waiting for MT5 to initialize (30 seconds)...
timeout /t 30 /nobreak

echo Starting ICT Trading Bot...
cd /d %~dp0
python standalone_trader.py

echo If the script exits, it will restart in 60 seconds...
timeout /t 60 /nobreak

echo Restarting script...
start "" %0
exit
