@echo off
cd /d "%~dp0"
if not exist .venv (
  python -m venv .venv
  call .venv\Scripts\activate.bat
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate.bat
)
if not exist .env (
  copy .env.example .env
  echo 请编辑 .env 填入 DEEPSEEK_API_KEY 后重新运行。
  pause
  exit /b 1
)
python main.py %*
