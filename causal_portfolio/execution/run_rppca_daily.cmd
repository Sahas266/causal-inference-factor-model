@echo off
rem Compatibility entrypoint for the already-registered Windows task.
rem RP-PCA was only an execution placeholder; all calls now use CPCM causal.
call "%~dp0run_cpcm_causal_daily.cmd" %*
exit /b %ERRORLEVEL%
