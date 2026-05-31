@echo off
cd /d "%~dp0"

echo ========================================
echo Setup HR Candidate Ranking Assistant
echo ========================================

python -m venv .venv

call .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python -m py_compile app\app.py src\inference.py

echo ========================================
echo Setup complete.
echo Starting Streamlit app...
echo ========================================

python -m streamlit run app\app.py

pause
