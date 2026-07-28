@echo off
setlocal

set "REPO_ROOT=%~dp0..\.."
pushd "%REPO_ROOT%" || exit /b 1

set "PYTHON_EXE=%REPO_ROOT%\venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo Missing repository Python: "%PYTHON_EXE%" 1>&2
    popd
    exit /b 1
)

if not exist "causal_portfolio\execution\logs" mkdir "causal_portfolio\execution\logs"

rem One-shot. Windows Task Scheduler owns the 30-minute recurrence.
rem %* forwards overrides like --mainnet.
"%PYTHON_EXE%" -m causal_portfolio.execution.notify --pnl %* >> "causal_portfolio\execution\logs\pnl_notifier.log" 2>&1

set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
