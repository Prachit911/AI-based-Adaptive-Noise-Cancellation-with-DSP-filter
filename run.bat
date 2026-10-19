@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"
title AI-based Adaptive Noise Cancellation

echo ========================================================
echo   AI-based Adaptive Noise Cancellation with DSP Filters
echo ========================================================
echo.

:: 1. Check or set up Virtual Environment
if not exist ".venv\Scripts\python.exe" (
    echo [.venv not detected] Creating virtual environment...
    py -m venv .venv 2>nul || python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Python not found or failed to create venv. Please make sure Python 3.10+ is installed.
        pause
        exit /b 1
    )
    echo Installing dependencies...
    call .\.venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    pip install gradio
) else (
    call .\.venv\Scripts\activate.bat
)

:: 2. Set PYTHONPATH to include src
set PYTHONPATH=%cd%\src

:: 3. Check CLI arguments
if "%~1"=="app" goto run_app
if "%~1"=="web" goto run_app
if "%~1"=="enhance" goto run_enhance
if "%~1"=="evaluate" goto run_eval
if "%~1"=="test" goto run_test
if "%~1"=="generate" goto run_gen

:menu
echo.
echo Select an option:
echo  1) Launch Interactive Web App (Gradio UI)
echo  2) Enhance 'noisy.wav' using Supervised Selector
echo  3) Enhance 'noisy.wav' using Q-learning Agent
echo  4) Generate synthetic audio samples
echo  5) Run held-out evaluation benchmark
echo  6) Train Supervised Selector
echo  7) Train Q-learning Agent
echo  8) Run test suite (pytest)
echo  9) Exit
echo.
set /p choice="Enter choice [1-9]: "

if "%choice%"=="1" goto run_app
if "%choice%"=="2" goto run_enhance_selector
if "%choice%"=="3" goto run_enhance_rl
if "%choice%"=="4" goto run_gen
if "%choice%"=="5" goto run_eval
if "%choice%"=="6" goto run_train_selector
if "%choice%"=="7" goto run_train_agent
if "%choice%"=="8" goto run_test
if "%choice%"=="9" goto end
echo Invalid selection. Please try again.
goto menu

:run_app
echo.
echo Launching Gradio Web App...
python -m noise_suppression.app
goto end

:run_enhance_selector
echo.
echo Enhancing noisy.wav with Supervised Selector...
python -m noise_suppression.enhance --input noisy.wav --output outputs/enhanced_selector.wav --method selector
echo Finished! Check outputs/enhanced_selector.wav
pause
goto menu

:run_enhance_rl
echo.
echo Enhancing noisy.wav with Q-learning Agent...
python -m noise_suppression.enhance --input noisy.wav --output outputs/enhanced_rl.wav --method auto
echo Finished! Check outputs/enhanced_rl.wav
pause
goto menu

:run_gen
echo.
echo Generating synthetic clean and noisy samples...
python -m noise_suppression.generate_synthetic_samples
echo Finished! Samples generated in data/ folder.
pause
goto menu

:run_eval
echo.
echo Running evaluation benchmark...
python -m noise_suppression.evaluate --n 30
echo Benchmark complete. See outputs/evaluation_report.md
pause
goto menu

:run_train_selector
echo.
echo [1/4] Making zero-leakage manifest...
python -m noise_suppression.make_manifest --check-leakage
echo [2/4] Preprocessing audio mixtures...
python -m noise_suppression.preprocess
echo [3/4] Sweeping filter bank...
python -m noise_suppression.sweep_filters --n-jobs 4
echo [4/4] Training GBRT selector...
python -m noise_suppression.train_selector
echo Trained model saved to models/selector.joblib
pause
goto menu

:run_train_agent
echo.
echo Training Q-learning Agent...
python -m noise_suppression.train_agent --episodes 2500
echo Q-table saved to models/q_table.json
pause
goto menu

:run_test
echo.
echo Running pytest...
pytest
pause
goto menu

:end
echo Exiting.
