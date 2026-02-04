@echo off
pip install -r requirements_local.txt
pause
python netty_local_server.py
echo Starting Server...
echo.

findstr /C:"LLM_MODEL" .env
findstr /C:"LLM_PROVIDER" .env
echo Current Provider settings:
echo Please ensure you have configured your LLM provider in the .env file.
echo.
echo Installing dependencies from requirements_local.txt...
echo Starting Netty Chat Local Server...

