@echo off
setlocal
cd /d "%~dp0"
set "PIP_CACHE_DIR=G:\Program Files\codexcache\pip"
set "TEMP=G:\Program Files\codexcache\tmp"
set "TMP=%TEMP%"
if not exist "%TEMP%" mkdir "%TEMP%"
if exist ".venv\Scripts\python.exe" (
  set "BUILD_PYTHON=.venv\Scripts\python.exe"
) else (
  set "BUILD_PYTHON=python"
)
"%BUILD_PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
"%BUILD_PYTHON%" scripts\build_release.py %*
exit /b %errorlevel%
