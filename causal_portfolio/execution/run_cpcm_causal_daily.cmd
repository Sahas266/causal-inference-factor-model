@echo off
setlocal

set "REPO_ROOT=%~dp0..\.."
pushd "%REPO_ROOT%" || exit /b 1

set "CPCM_LOCAL_DB=%REPO_ROOT%\causal_portfolio\data\cpcm_local.duckdb"
set "PYTHON_EXE=%REPO_ROOT%\venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo Missing repository Python: "%PYTHON_EXE%" 1>&2
    popd
    exit /b 1
)

if "%~1"=="" (
    if not exist "causal_portfolio\execution\logs" mkdir "causal_portfolio\execution\logs"
    "%PYTHON_EXE%" -m causal_portfolio.models.causal_daily --retire-rppca-only --max-transaction-cost-bps 15 --estimated-taker-fee-bps 4.5 >> "causal_portfolio\execution\logs\cpcm_causal_daily_task.log" 2>&1
    if errorlevel 1 goto :failed
    "%PYTHON_EXE%" -m causal_portfolio.models.causal_daily --refresh-data --target-out tmp\cpcm_causal_daily_target.json --base-weight 0.05 --max-transaction-cost-bps 15 --estimated-taker-fee-bps 4.5 --min-position-change-pct 0.10 --min-rebalance-completeness 0.90 --execute >> "causal_portfolio\execution\logs\cpcm_causal_daily_task.log" 2>&1
) else (
    "%PYTHON_EXE%" -m causal_portfolio.models.causal_daily %*
)

set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%

:failed
set "EXIT_CODE=%ERRORLEVEL%"
:finish
popd
exit /b %EXIT_CODE%
