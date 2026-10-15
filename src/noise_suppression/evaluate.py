"""Compare DSP actions on synthetic mixtures from held-out clean/noise folders.

Expected folders: data/eval_clean and data/eval_noise. If empty, use separate
held-out files from data/clean and data/noise, and pass those folders explicitly.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import csv
import random
import numpy as np
from .audio_utils import read_audio, resample_audio, mix_at_snr, si_sdr
from .features import extract_state
from .filters import ACTIONS, apply_filter
from .agent import QLearningFilterSelector

EXTS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


def list_audio(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate filters with clean references and held-out noises.")
    parser.add_argument("--clean-dir", default="data/eval_clean")
    parser.add_argument("--noise-dir", default="data/eval_noise")
    parser.add_argument("--q-table", default="models/q_table.json")
    parser.add_argument("--n", type=int, default=30, help="Number of random mixtures")
    parser.add_argument("--output", default="outputs/evaluation.csv")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    clean_files, noise_files = list_audio(Path(args.clean_dir)), list_audio(Path(args.noise_dir))
    if not clean_files or not noise_files:
        raise SystemExit("Evaluation data missing. Put held-out clean speech in data/eval_clean and noise in data/eval_noise.")
    agent = QLearningFilterSelector.load(args.q_table) if Path(args.q_table).exists() else None
    py_rng = random.Random(args.seed)
    rng = np.random.default_rng(args.seed)
    actions_to_test = [*ACTIONS, "rl_auto"]
    rows = []

    for idx in range(args.n):
        clean_path, noise_path = py_rng.choice(clean_files), py_rng.choice(noise_files)
        clean, sr = read_audio(clean_path)
        noise, noise_sr = read_audio(noise_path)
        noise = resample_audio(noise, noise_sr, sr)
        if clean.size > sr * 12:
            start = int(rng.integers(0, clean.size - sr * 12 + 1))
            clean = clean[start:start + sr * 12]
        snr_db = py_rng.choice([-5.0, 0.0, 5.0, 10.0])
        noisy = mix_at_snr(clean, noise, snr_db, rng)
        baseline = si_sdr(clean, noisy)
        state = extract_state(noisy, sr)
        tested = []
        for action in ACTIONS:
            out = apply_filter(action, noisy, sr)
            tested.append((action, si_sdr(clean, out) - baseline))
        if agent is not None:
            selected = agent.select(state)
            out = apply_filter(selected, noisy, sr)
            tested.append(("rl_auto", si_sdr(clean, out) - baseline))
        for method, improvement in tested:
            rows.append({"sample": idx + 1, "clean_file": clean_path.name, "noise_file": noise_path.name,
                         "snr_db": snr_db, "state": state, "method": method,
                         "si_sdr_improvement_db": round(float(improvement), 4)})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved per-sample results to {output}")
    for method in sorted({r["method"] for r in rows}):
        vals = [r["si_sdr_improvement_db"] for r in rows if r["method"] == method]
        print(f"{method:24s}: mean SI-SDR improvement = {np.mean(vals):7.3f} dB")


if __name__ == "__main__":
    main()
