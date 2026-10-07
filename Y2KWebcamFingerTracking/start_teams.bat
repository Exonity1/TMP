@echo off
rem Vorschaufenster plus Ausgabe an die OBS Virtual Camera (fuer MS Teams).
cd /d "%~dp0"
".venv\Scripts\python.exe" finger_tracking.py --virtual-cam %*
if errorlevel 1 pause
