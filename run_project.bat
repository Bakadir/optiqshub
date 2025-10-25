@echo off
echo ================================
echo  Django Project Starter
echo ================================

REM Check for virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install dependencies
if exist "requirements.txt" (
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    echo No requirements.txt found. Skipping install.
)

REM Run Django migrations
echo Running migrations...
python manage.py migrate

REM Run the server
echo Starting Django development server...
start http://127.0.0.1:8000
python manage.py runserver
