@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Chay 1_CAI_DAT.bat truoc
 pause
 exit /b 1
)
.venv\Scripts\python.exe main.py --source 0
pause
