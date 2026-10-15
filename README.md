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

## 2. Prepare audio files

For the very first run, keep it simple: record 3-5 short clean speech clips in a quiet room and 3-5 separate noise clips (fan, traffic, keyboard typing, clapping, etc.). Save them as WAV files.

- Put clean speech in `data/clean/`.
- Put environmental-noise-only recordings in `data/noise/`.
- Put **different** clean speech recordings for final evaluation in `data/eval_clean/`.
- Put **different** noise recordings for final evaluation in `data/eval_noise/`.

Accepted formats are WAV, FLAC, OGG, and AIFF. Stereo is converted to mono automatically. Keep clean speech and noise as separate files; the training script creates noisy mixtures at different SNR values automatically. Do not put already-mixed speech+noise into `data/noise/`.

For larger public datasets, these are useful starting points:

- [Mini LibriSpeech (clean English speech; CC BY 4.0)](https://www.openslr.org/31/). The page lists a 126 MB `dev-clean-2` archive and a 332 MB `train-clean-5` archive.
- [ESC-50 environmental sounds](https://github.com/karolpiczak/ESC-50). It includes 5-second WAV clips and categories such as clapping and keyboard typing. The dataset is CC BY-NC, so check attribution and non-commercial restrictions before using or redistributing it.

Extract selected recordings into the matching folders. For the final project, use a reasonable variety of recordings and maintain a held-out evaluation set.

## 3. Try an individual DSP filter first

Put a noisy recording at `noisy.wav`, then run:

```powershell
python -m noise_suppression.enhance --input noisy.wav --output outputs/wiener.wav --method wiener
python -m noise_suppression.enhance --input noisy.wav --output outputs/subtraction.wav --method spectral_subtraction
python -m noise_suppression.enhance --input noisy.wav --output outputs/bandpass.wav --method speech_bandpass
```

Other methods: `keep_original`, `notch_50hz`.

## 4. Train the RL selector

After adding clean and noise files, run:

```powershell
python -m noise_suppression.train_agent --episodes 2500
```

A policy is saved to `models/q_table.json`. For a quick smoke test, use `--episodes 100`; it is not enough to claim robust performance.

## 5. Let RL choose the filter

```powershell
python -m noise_suppression.enhance --input noisy.wav --output outputs/rl_enhanced.wav --method auto
```

The console prints the extracted state and chosen filter.

## 6. Evaluate filters (held-out data)

Fill `data/eval_clean/` and `data/eval_noise/` with recordings not used in training, then run:

```powershell
python -m noise_suppression.evaluate --n 30
```

Results are saved to `outputs/evaluation.csv` with the SI-SDR improvement for each method. Higher SI-SDR improvement is generally better, but listening is still necessary; no single metric fully describes perceived quality.

## 7. Optional web interface

```powershell
pip install gradio
$env:PYTHONPATH = "$PWD\src"
python -m noise_suppression.app
```

Open the local URL printed in the terminal.

## Initial limitations to state honestly in the report

- The noise profile is estimated from quiet frames and can be wrong when speech/noise overlap strongly.
- A fixed speech bandpass and fixed 50 Hz notches only help for some noise types; they are not universal denoisers.
- The reward is based on SI-SDR improvement against a clean reference during training. At inference time, the clean reference is not needed.
- A whole-clip filter choice is a starting point. Frame-wise policy switching can be investigated later if time permits.
- The RL selector is only as good as the acoustic states, training samples, filters, and reward. Compare it against each individual filter on held-out audio before claiming it improves performance.
