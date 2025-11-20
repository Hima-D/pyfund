@echo off
:: make.bat - Professional build/script runner for pyfundlib
:: Windows-friendly, colorful, and feels like magic

:: Colors
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set CYAN=[96m
set MAGENTA=[95m
set RESET=[0m

:: Helper
if "%~1"=="" goto help

:: Commands
if /I "%~1"=="help" goto help
if /I "%~1"=="install" goto install
if /I "%~1"=="dev" goto dev
if /I "%~1"=="test" goto test
if /I "%~1"=="lint" goto lint
if /I "%~1"=="format" goto format
if /I "%~1"=="backtest" goto backtest
if /I "%~1"=="train" goto train
if /I "%~1"=="live" goto live
if /I "%~1"=="paper" goto paper
if /I "%~1"=="automate" goto automate
if /I "%~1"=="dashboard" goto dashboard
if /I "%~1"=="clean" goto clean
goto invalid

:help
echo.
echo %CYAN%PyFundLib Make Script%RESET%
echo.
echo %GREEN%Usage:%RESET% make ^<command%GREEN%^>%RESET%
echo.
echo %YELLOW%Commands:%RESET%
echo   %MAGENTA%install%RESET%     Install package + dependencies
echo   %MAGENTA%dev%RESET%         Install in editable mode with dev tools
echo   %MAGENTA%test%RESET%        Run tests
echo   %MAGENTA%lint%RESET%        Run ruff + mypy
echo   %MAGENTA%format%RESET%      Format code (black + ruff)
echo   %MAGENTA%backtest%RESET%    Run quick backtest (AAPL + RSI)
echo   %MAGENTA%train%RESET%       Train ML models
echo   %MAGENTA%live%RESET%        Start live trading (DANGER!)
echo   %MAGENTA%paper%RESET%       Start paper trading
echo   %MAGENTA%automate%RESET%    Start full automation
echo   %MAGENTA%dashboard%RESET%   Launch Streamlit dashboard
echo   %MAGENTA%clean%RESET%       Clean cache, logs, builds
echo.
goto end

:install
echo %GREEN%Installing pyfundlib...%RESET%
pip install .
echo %GREEN%Done!%RESET%
goto end

:dev
echo %GREEN%Installing in dev mode...%RESET%
pip install -e .[dev,broker]
echo %GREEN%Dev environment ready!%RESET%
goto end

:test
echo %YELLOW%Running tests...%RESET%
pytest -v
goto end

:lint
echo %YELLOW%Linting code...%RESET%
ruff check .
mypy src
goto end

:format
echo %CYAN%Formatting code...%RESET%
black .
ruff check --fix .
echo %GREEN%Code is beautiful now!%RESET%
goto end

:backtest
echo %MAGENTA%Running quick backtest (AAPL + RSI)%RESET%
python -c "from pyfundlib.backtester.engine import Backtester; from pyfundlib.strategies.rsi_mean_reversion import RSIMeanReversionStrategy; bt=Backtester(strategy=RSIMeanReversionStrategy(), ticker='AAPL'); bt.report()"
goto end

:train
echo %MAGENTA%Starting ML training...%RESET%
pyfundlib train
goto end

:live
echo %RED%LIVE TRADING MODE%RESET%
choice /C YN /M "Are you SURE you want to trade real money?"
if errorlevel 2 goto end
pyfundlib automate --mode live --confirm-live
goto end

:paper
echo %GREEN%Starting paper trading...%RESET%
pyfundlib automate
goto end

:automate
echo %CYAN%Launching full automation...%RESET%
pyfundlib automate
goto end

:dashboard
echo %CYAN%Launching dashboard...%RESET%
streamlit run examples/05_dashboard.py
goto end

:clean
echo %YELLOW%Cleaning project...%RESET%
rmdir /S /Q build dist *.egg-info __pycache__ .pytest_cache .mypy_cache cache logs mlruns 2>nul || echo Some dirs already clean
echo %GREEN%Clean complete!%RESET%
goto end

:invalid
echo %RED%Unknown command: %1%RESET%
echo Type %MAGENTA%make help%RESET% for available commands
goto end

:end
echo.