# AI-Based Adaptive Noise Cancellation with DSP Filters

A beginner-friendly Python project where a small Q-learning agent selects a DSP filter for a noisy speech clip. The first version includes a simple filter bank and an end-to-end training/inference path.

## What the prototype does

1. Extracts simple acoustic features from the noisy clip.
2. Converts the feature values into a discrete state.
3. Uses a tabular Q-learning policy to select one DSP action for the whole clip.
4. Supports spectral subtraction, Wiener filtering, a speech-band bandpass, 50 Hz hum notches, and leaving audio unchanged.
5. Trains by making noisy mixtures from clean speech plus separate noise recordings. During training, the clean reference lets the agent calculate an SI-SDR-based reward.

**Scope note:** This is a one-step/episodic MDP (contextual action-selection) prototype, not a long-horizon RL system. The first version selects one filter per file rather than switching frame-by-frame. These are deliberate simplifications to make the project runnable and explainable.

## 1. Setup (Windows PowerShell)

Install Python 3.10 or later. Open PowerShell in this project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
```

Run the commands below in the **same PowerShell window** so `PYTHONPATH` stays set. If PowerShell blocks virtual-environment activation, open a new terminal and use `.\.venv\Scripts\python.exe` in place of `python` in the commands below.

## 2. Quick start with synthetic audio (or prepare real datasets)

To immediately try all models without downloading large datasets, run the synthetic audio generator:

```powershell
python -m noise_suppression.generate_synthetic_samples
```

This populates `data/clean/`, `data/noise/`, `data/eval_clean/`, and `data/eval_noise/`, and creates `noisy.wav` for quick CLI testing.

### Alternatively, use real recordings or public datasets:
- Put clean speech in `data/clean/`.
- Put environmental-noise-only recordings in `data/noise/`.
- Put **different** clean speech recordings for evaluation in `data/eval_clean/`.
- Put **different** noise recordings for evaluation in `data/eval_noise/`.

For public corpora (LibriSpeech + ESC-50), run:
```powershell
python prepare_public_datasets_fixed.py --reset
```

## 3. Try individual DSP filters

Apply any of the 12 filters in the extended filter bank on a noisy recording:

```powershell
python -m noise_suppression.enhance --input noisy.wav --output outputs/wiener.wav --method wiener
python -m noise_suppression.enhance --input noisy.wav --output outputs/subtraction.wav --method spectral_subtraction
python -m noise_suppression.enhance --input noisy.wav --output outputs/bandpass.wav --method speech_bandpass
python -m noise_suppression.enhance --input noisy.wav --output outputs/notch50.wav --method notch_50hz
python -m noise_suppression.enhance --input noisy.wav --output outputs/log_mmse.wav --method log_mmse
python -m noise_suppression.enhance --input noisy.wav --output outputs/declicker.wav --method median_declicker
```

**Full action list:** `keep_original`, `spectral_subtraction`, `wiener`, `wiener_mild`, `wiener_aggressive`, `speech_bandpass`, `notch_50hz`, `notch_60hz`, `highpass_80hz`, `log_mmse`, `median_declicker`, `bandpass_then_wiener`.

## 4. Train the AI selectors

### Option A: Supervised Filter Selector (Recommended)

The Supervised Selector uses 35 continuous acoustic features (MFCCs, spectral flux/flatness/entropy/centroid/rolloff, HNR, zero-crossing rate, band energies, hum power ratios) and a Gradient Boosted Decision Tree (GBRT) bank to predict the expected SI-SDR gain of every filter.

```powershell
# 1. Build zero-leakage manifest
python -m noise_suppression.make_manifest --check-leakage

# 2. Preprocess normalized audio mixtures
python -m noise_suppression.preprocess

# 3. Parallel filter sweep (full-information feedback matrix)
python -m noise_suppression.sweep_filters --n-jobs 4

# 4. Train the GBRT selector
python -m noise_suppression.train_selector
```

Saves the trained model to `models/selector.joblib`.

### Option B: Tabular Q-learning Agent (Baseline)

```powershell
python -m noise_suppression.train_agent --episodes 2500
```

Saves the Q-table policy to `models/q_table.json`.

## 5. Enhance audio with automated AI selection

### Using the Supervised Selector (Default):
```powershell
python -m noise_suppression.enhance --input noisy.wav --output outputs/enhanced_selector.wav --method selector
```

### Using the Q-learning policy:
```powershell
python -m noise_suppression.enhance --input noisy.wav --output outputs/enhanced_rl.wav --method auto
```

## 6. Comprehensive benchmark evaluation

Run held-out evaluation across all filters, AI selectors, and the oracle upper bound:

```powershell
python -m noise_suppression.evaluate --n 30
```

Outputs:
- `outputs/evaluation.csv` & `outputs/evaluation_results.csv`: per-sample metrics.
- `outputs/evaluation_summary.csv`: aggregated results with 95% bootstrap confidence intervals.
- `outputs/evaluation_report.md`: formatted Markdown report with SNR and noise category breakdowns.

Metrics computed:
- **SI-SDR improvement (dB)**
- **STOI improvement**
- **PESQ-WB improvement** (if `pesq` is installed)
- **95% bootstrap confidence intervals**

## 7. Interactive web interface

Launch the Gradio web dashboard:

```powershell
python -m noise_suppression.app
```

Open the local URL displayed in the terminal to upload noisy clips, select enhancement methods, inspect predicted filter gains, and listen to before-and-after audio side by side.

## 8. Running test suite

Run all 37 unit and integration tests:

```powershell
pytest
```
