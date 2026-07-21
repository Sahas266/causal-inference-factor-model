@echo off
setlocal

set "REPO_ROOT=%~dp0..\.."
pushd "%REPO_ROOT%" || exit /b 1

set "PYTHON_EXE=%REPO_ROOT%\venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

if not exist "causal_portfolio\execution\logs" mkdir "causal_portfolio\execution\logs"

rem Long-running: sends a PnL update every 30 minutes until stopped.
rem Start it once (e.g. via Task Scheduler "at log on", no repeat trigger
rem needed) and leave it running; %* forwards overrides like --mainnet.
"%PYTHON_EXE%" -m causal_portfolio.execution.notify --pnl-loop --interval-minutes 30 %* >> "causal_portfolio\execution\logs\pnl_notifier.log" 2>&1

set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
