@echo off
REM 공구 마모 진단 GUI 실행 (더블클릭용)
REM  - venv 파이썬으로 실행해야 cv2/torch 가 잡힌다.
REM  - pythonw.exe 를 쓰면 검은 콘솔 창 없이 GUI 만 뜬다.
setlocal
cd /d "%~dp0"

set "VENV=C:\MAENG-Lab\venv\Scripts"
set "PYW=%VENV%\pythonw.exe"
set "PY=%VENV%\python.exe"

if not exist "%PY%" (
    echo [오류] venv 파이썬을 찾지 못했습니다: %PY%
    echo        venv 위치가 다르면 이 파일의 VENV 경로를 고쳐주세요.
    pause
    exit /b 1
)

if exist "%PYW%" (
    start "" "%PYW%" "%~dp0monitor_gui_v5.py"
) else (
    start "" "%PY%" "%~dp0monitor_gui_v5.py"
)
endlocal
