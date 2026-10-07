@echo off
rem Nur Vorschaufenster, keine virtuelle Kamera.
cd /d "%~dp0"
".venv\Scripts\python.exe" finger_tracking.py %*
if errorlevel 1 pause
