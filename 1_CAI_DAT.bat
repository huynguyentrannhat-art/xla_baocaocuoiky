@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY=py -3.14"
%PY% --version >nul 2>&1
if errorlevel 1 (
  set "PY=py -3.11"
  %PY% --version >nul 2>&1
)
if errorlevel 1 (
  set "PY=python"
  %PY% --version
  if errorlevel 1 (
    echo KHONG TIM THAY PYTHON. Hay cai Python 3.11+ va them vao PATH.
    pause
    exit /b 1
  )
)
echo Dang tao moi truong bang: %PY%
%PY% -m venv .venv
if errorlevel 1 (
 echo KHONG TAO DUOC .venv. Kiem tra ban Python va quyen thu muc.
 pause
 exit /b 1
)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
 echo LOI CAI GOI. Xem log o tren.
 pause
 exit /b 1
)
echo [OK] CAI DAT HOAN TAT
pause
