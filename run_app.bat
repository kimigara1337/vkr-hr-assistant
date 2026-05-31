@echo off
cd /d "%~dp0"

echo ========================================
echo HR Candidate Ranking Assistant
echo ========================================

if not exist ".venv" (
    echo Virtual environment not found.
    echo Run setup first:
    echo python -m venv .venv
    echo .venv\Scripts\activate
    echo pip install -r requirements.txt
    pause
    exit /b
)

call .venv\Scripts\activate

echo Starting Streamlit app...
python -m streamlit run app\app.py

pause
