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

if not exist ".env" (
    copy ".env.example" ".env" >nul
    if errorlevel 1 (
        echo Failed to create .env from .env.example.
        exit /b 1
    )
)

echo.
echo Setup complete.
echo Add your Dhan credentials to .env, then run:
echo   run.bat --once
echo   run.bat
echo   run.bat --ignore-entry-cutoff  (extend entry cutoff to 15:25 IST)
echo Optional Telegram alerts: create a bot with @BotFather, then set TELEGRAM_BOT_TOKEN and
echo TELEGRAM_CHAT_ID in .env. Alerts are sent for new BUY recommendations only.
echo Live orders are off by default; set ENABLE_LIVE_ORDERS=true in .env and pass --enable-live-orders.
echo Dhan order placement requires a static IP whitelisted with Dhan.
echo To deploy on Linux, see the UNIX / CRON instructions at the top of breakout_scanner.py.
echo   backtest.bat --start YYYY-MM-DD --end YYYY-MM-DD --symbols RELIANCE,HDFCBANK --disable breakout_volume
echo     Historical backtest is an underlying-price proxy; option OI/liquidity and option P^&L are not tested.
exit /b 0
