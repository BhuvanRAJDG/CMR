@echo off
title VitalDrive AI — Integrated Dashboard
echo ================================================
echo   VitalDrive AI  (SafeDriveAI + Sleep Monitor)
echo ================================================
echo.

REM Use the SafeDriveAI venv
set PYTHON="%~dp0SafeDriveAI\.venv\Scripts\python.exe"
set STREAMLIT="%~dp0SafeDriveAI\.venv\Scripts\streamlit.exe"

REM Verify packages are present
%PYTHON% -c "import streamlit, cv2, mediapipe" 2>nul
if errorlevel 1 (
    echo [SETUP] Installing required packages...
    %PYTHON% -m pip install streamlit opencv-python mediapipe pyserial twilio scipy requests Pillow plotly streamlit-autorefresh pyttsx3 pandas numpy
    echo [SETUP] Done.
    echo.
)

echo [RUN] Starting VitalDrive AI dashboard...
echo [RUN] Open your browser at http://localhost:8501
echo.

%STREAMLIT% run "%~dp0SafeDriveAI\main.py" --server.port 8501 --server.headless false --browser.gatherUsageStats false

pause
