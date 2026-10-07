@echo off
rem Einmalige Einrichtung: legt die Python-Umgebung an und installiert die Pakete.
cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 (
    echo Python wurde nicht gefunden. Bitte Python 3.11 von python.org
    echo nur fuer den eigenen Benutzer installieren und "Add python.exe to PATH" anhaken.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" python -m venv .venv
if errorlevel 1 goto :failed

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo.
echo Fertig. Starten mit start.bat (nur Vorschau) oder start_teams.bat (mit virtueller Kamera).
pause
exit /b 0

:failed
echo.
echo Die Einrichtung ist fehlgeschlagen, siehe Meldung oben.
pause
exit /b 1
