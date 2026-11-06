@echo off
rem ---------------------------------------------------------------------------
rem update_from_usb.bat -- copy updated ASH code from this flash drive onto the
rem air-gapped machine, without touching anything ASH has learned there.
rem
rem   Run it from the flash drive:   E:\A.S.H\offline\update_from_usb.bat
rem   Different destination:         update_from_usb.bat "D:\path\to\A.S.H"
rem   Skip the confirmation prompt:  update_from_usb.bat "D:\path\to\A.S.H" /y
rem
rem Default destination is C:\Users\ash\Desktop\A.S.H.
rem
rem NEVER overwritten on the destination (they belong to that machine):
rem   entities.json, episodic_memory.jsonl   what ASH learned about its user
rem   state\                                 brain weights, concept graph, journal
rem   logs\                                  daemon logs
rem   daemon.json                            that machine's settings
rem   .env                                   that machine's LLM address/keys
rem   data\embeddings_intents.pkl            local cache, rebuilt on startup
rem Not copied at all: .git, .claude, __pycache__
rem
rem Uses robocopy (built into Windows), so no git or Python is needed. Only
rem changed files are copied. Files deleted from the code are NOT deleted on the
rem destination.
rem ---------------------------------------------------------------------------
setlocal

for %%I in ("%~dp0..") do set "SRC=%%~fI"
set "DST=%~1"
if "%DST%"=="" set "DST=C:\Users\ash\Desktop\A.S.H"
set "ASSUME_YES=0"
if /i "%~2"=="/y" set "ASSUME_YES=1"

if not exist "%SRC%\ashd.py" (
    echo ERROR: %SRC% does not look like an ASH folder ^(no ashd.py^).
    goto :fail
)
if not exist "%DST%\ashd.py" (
    echo ERROR: %DST% does not look like an ASH folder ^(no ashd.py^).
    echo        Pass the right folder as the first argument.
    goto :fail
)

echo.
echo   From: %SRC%
echo   To:   %DST%
echo.
echo   Kept as-is on the destination: learned memory, state, logs, daemon.json, .env
echo   Make sure ASH is stopped on that machine before continuing.
echo.

if "%ASSUME_YES%"=="1" goto :copy

:ask
choice /c YNP /m "Copy now? Y = yes, N = cancel, P = preview only"
if errorlevel 3 goto :preview
if errorlevel 2 goto :cancel
goto :copy

:preview
echo.
echo --- PREVIEW (nothing is copied) ---
call :run /L
echo --- end of preview ---
echo.
goto :ask

:copy
echo.
call :run
if errorlevel 8 goto :fail
echo.
echo Done. Start ASH as usual:  python ashd.py
goto :end

:run
robocopy "%SRC%" "%DST%" /E /FFT /R:1 /W:1 /NP /NDL /NJH %1 ^
  /XD "%SRC%\.git" "%SRC%\.claude" "%SRC%\state" "%SRC%\logs" "__pycache__" ^
  /XF entities.json episodic_memory.jsonl daemon.json .env embeddings_intents.pkl ashd.pid *.pyc
exit /b %errorlevel%

:cancel
echo Cancelled. Nothing was copied.
goto :end

:fail
echo.
echo Update did NOT complete.
endlocal
exit /b 1

:end
endlocal
exit /b 0
