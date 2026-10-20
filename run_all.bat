@echo off
setlocal
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Virtual environment not found. Run setup.bat first.
    exit /b 1
)

"%~dp0.venv\Scripts\python.exe" "%~dp0strategy_runner.py" %*
exit /b %errorlevel%
