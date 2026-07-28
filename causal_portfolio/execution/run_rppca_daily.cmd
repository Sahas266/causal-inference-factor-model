@echo off
setlocal

set "REPO_ROOT=%~dp0..\.."
pushd "%REPO_ROOT%" || exit /b 1

set "CPCM_LOCAL_DB=%REPO_ROOT%\causal_portfolio\data\cpcm_local.duckdb"
set "PYTHON_EXE=%REPO_ROOT%\venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

if "%~1"=="" (
    if not exist "causal_portfolio\execution\logs" mkdir "causal_portfolio\execution\logs"
    "%PYTHON_EXE%" -m causal_portfolio.models.rppca_daily --refresh-prices --target-out tmp\rppca_daily_target.json --target-gross 0.3 --max-transaction-cost-bps 15 --estimated-taker-fee-bps 4.5 --execute >> "causal_portfolio\execution\logs\rppca_daily_task.log" 2>&1
) else (
    "%PYTHON_EXE%" -m causal_portfolio.models.rppca_daily %*
)

set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
