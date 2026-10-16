@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo Python Launcher was not found. Install Python 3.10 or newer and enable the py launcher.
    exit /b 1
)

py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo Python 3.10 or newer is required.
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3 -m venv .venv
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        exit /b 1
    )
)

echo Installing dependencies into .venv...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency installation failed.
    exit /b 1
)

if not exist "config.json" (
    copy "config.example.json" "config.json" >nul
    if errorlevel 1 (
        echo Failed to create config.json from config.example.json.
        exit /b 1
    )
)

echo.
echo Setup complete.
echo Add your Dhan client_id and access_token to config.json, then run:
echo   run.bat --once
echo   run.bat
exit /b 0
