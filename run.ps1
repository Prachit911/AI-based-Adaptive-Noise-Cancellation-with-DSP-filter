<#
.SYNOPSIS
    Convenience runner script for AI-Based Adaptive Noise Cancellation with DSP Filters.
.DESCRIPTION
    Sets up the virtual environment, configures PYTHONPATH, and runs desired project actions.
#>

[CmdletBinding()]
param (
    [Parameter(Position = 0)]
    [ValidateSet("app", "web", "enhance", "enhance-rl", "generate", "evaluate", "train-selector", "train-agent", "test")]
    [string]$Action
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  AI-based Adaptive Noise Cancellation with DSP Filters" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Setup / Activate Virtual Environment
$venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$venvActivate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"

if (-not (Test-Path $venvPython)) {
    Write-Host "[.venv not detected] Creating virtual environment..." -ForegroundColor Yellow
    py -m venv .venv 2>$null
    if (-not (Test-Path $venvPython)) {
        python -m venv .venv
    }
    
    if (-not (Test-Path $venvPython)) {
        Write-Error "Failed to create .venv. Please make sure Python 3.10+ is installed and on PATH."
        exit 1
    }

    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
    & $venvPython -m pip install gradio
}

# Set PYTHONPATH
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

function Show-Menu {
    Write-Host ""
    Write-Host "Select an option to run:" -ForegroundColor Green
    Write-Host " 1) Launch Interactive Web App (Gradio UI)"
    Write-Host " 2) Enhance 'noisy.wav' using Supervised Selector"
    Write-Host " 3) Enhance 'noisy.wav' using Q-learning Agent"
    Write-Host " 4) Generate synthetic audio samples"
    Write-Host " 5) Run held-out evaluation benchmark"
    Write-Host " 6) Train Supervised Selector"
    Write-Host " 7) Train Q-learning Agent"
    Write-Host " 8) Run test suite (pytest)"
    Write-Host " 9) Exit"
    Write-Host ""
    return (Read-Host "Enter choice [1-9]")
}

function Run-App {
    Write-Host "`nLaunching Gradio Web App..." -ForegroundColor Cyan
    & $venvPython -m noise_suppression.app
}

function Run-Enhance-Selector {
    Write-Host "`nEnhancing noisy.wav with Supervised Selector..." -ForegroundColor Cyan
    & $venvPython -m noise_suppression.enhance --input noisy.wav --output outputs/enhanced_selector.wav --method selector
    Write-Host "Saved output to outputs/enhanced_selector.wav" -ForegroundColor Green
}

function Run-Enhance-RL {
    Write-Host "`nEnhancing noisy.wav with Q-learning Agent..." -ForegroundColor Cyan
    & $venvPython -m noise_suppression.enhance --input noisy.wav --output outputs/enhanced_rl.wav --method auto
    Write-Host "Saved output to outputs/enhanced_rl.wav" -ForegroundColor Green
}

function Run-Generate {
    Write-Host "`nGenerating synthetic clean and noisy samples..." -ForegroundColor Cyan
    & $venvPython -m noise_suppression.generate_synthetic_samples
    Write-Host "Samples generated in data/ folder." -ForegroundColor Green
}

function Run-Evaluate {
    Write-Host "`nRunning evaluation benchmark..." -ForegroundColor Cyan
    & $venvPython -m noise_suppression.evaluate --n 30
    Write-Host "Benchmark complete. Results in outputs/evaluation_report.md" -ForegroundColor Green
}

function Run-Train-Selector {
    Write-Host "`n[1/4] Generating manifest..." -ForegroundColor Cyan
    & $venvPython -m noise_suppression.make_manifest --check-leakage
    Write-Host "`n[2/4] Preprocessing audio mixtures..." -ForegroundColor Cyan
    & $venvPython -m noise_suppression.preprocess
    Write-Host "`n[3/4] Sweeping filter bank..." -ForegroundColor Cyan
    & $venvPython -m noise_suppression.sweep_filters --n-jobs 4
    Write-Host "`n[4/4] Training GBRT selector..." -ForegroundColor Cyan
    & $venvPython -m noise_suppression.train_selector
    Write-Host "Trained model saved to models/selector.joblib" -ForegroundColor Green
}

function Run-Train-Agent {
    Write-Host "`nTraining Q-learning agent..." -ForegroundColor Cyan
    & $venvPython -m noise_suppression.train_agent --episodes 2500
    Write-Host "Q-table saved to models/q_table.json" -ForegroundColor Green
}

function Run-Tests {
    Write-Host "`nRunning pytest..." -ForegroundColor Cyan
    & (Join-Path $ProjectRoot ".venv\Scripts\pytest.exe")
}

# If action parameter supplied via command line
switch ($Action) {
    "app"            { Run-App; exit 0 }
    "web"            { Run-App; exit 0 }
    "enhance"        { Run-Enhance-Selector; exit 0 }
    "enhance-rl"     { Run-Enhance-RL; exit 0 }
    "generate"       { Run-Generate; exit 0 }
    "evaluate"       { Run-Evaluate; exit 0 }
    "train-selector" { Run-Train-Selector; exit 0 }
    "train-agent"    { Run-Train-Agent; exit 0 }
    "test"           { Run-Tests; exit 0 }
}

# Otherwise interactive loop
do {
    $choice = Show-Menu
    switch ($choice) {
        "1" { Run-App }
        "2" { Run-Enhance-Selector }
        "3" { Run-Enhance-RL }
        "4" { Run-Generate }
        "5" { Run-Evaluate }
        "6" { Run-Train-Selector }
        "7" { Run-Train-Agent }
        "8" { Run-Tests }
        "9" { Write-Host "Exiting."; exit 0 }
        default { Write-Host "Invalid selection. Please choose 1-9." -ForegroundColor Red }
    }
} while ($true)
