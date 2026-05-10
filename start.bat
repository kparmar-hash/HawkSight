@echo off
REM Navigate to project root if not already there (optional, depending on where you run this)
cd /d "%~dp0"

REM Activate the virtual environment
call .venv\Scripts\activate.bat

REM The prompt is now modified to show (.venv). 
REM You can add your python command here, e.g.:
REM python main.py

REM Keep the window open if run by double-clicking
REM cmd /k