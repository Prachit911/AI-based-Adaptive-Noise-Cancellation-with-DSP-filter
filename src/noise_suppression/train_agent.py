"""Train the Q-learning filter selector using synthetic noisy/clean pairs.

Expected data layout:
  data/clean/*.wav  -- clean speech recordings
  data/noise/*.wav  -- environmental noise recordings
"""
from __future__ import annotations
import argparse
from pathlib import Path
import random
import time
import numpy as np
from .audio_utils import read_audio, resample_audio, mix_at_snr, si_sdr
from .features import extract_state
from .filters import ACTIONS, apply_filter
from .agent import QLearningFilterSelector

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


def list_audio(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RL-based DSP filter selector.")
    parser.add_argument("--clean-dir", default="data/clean", help="Directory containing clean speech WAV/FLAC files")
    parser.add_argument("--noise-dir", default="data/noise", help="Directory containing environmental-noise recordings")
    parser.add_argument("--episodes", type=int, default=2500, help="Number of Q-learning episodes")
    parser.add_argument("--output", default="models/q_table.json", help="Where to save the trained policy")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    clean_files = list_audio(Path(args.clean_dir))
    noise_files = list_audio(Path(args.noise_dir))
    if not clean_files:
        raise SystemExit(f"No clean audio found in {args.clean_dir}. Add clean speech WAV/FLAC files first.")
    if not noise_files:
        raise SystemExit(f"No noise audio found in {args.noise_dir}. Add fan/traffic/clapping/keyboard/noise WAV/FLAC files first.")
    if args.episodes < 1:
        raise SystemExit("--episodes must be at least 1.")

    rng = np.random.default_rng(args.seed)
    py_rng = random.Random(args.seed)
    agent = QLearningFilterSelector(alpha=0.15, gamma=0.0, seed=args.seed)
    snr_choices = [-5.0, 0.0, 5.0, 10.0, 15.0]
    rewards: list[float] = []
    started = time.time()

    for episode in range(args.episodes):
        clean_path = py_rng.choice(clean_files)
        noise_path = py_rng.choice(noise_files)
        clean, clean_sr = read_audio(clean_path)
        noise, noise_sr = read_audio(noise_path)
        noise = resample_audio(noise, noise_sr, clean_sr)
        if clean.size > clean_sr * 12:  # Keep training responsive on very long recordings.
            start = int(rng.integers(0, clean.size - clean_sr * 12 + 1))
            clean = clean[start:start + clean_sr * 12]
        snr_db = float(py_rng.choice(snr_choices))
        noisy = mix_at_snr(clean, noise, snr_db, rng)
        state = extract_state(noisy, clean_sr)

        action_index = agent.choose_action_index(state, epsilon=max(0.05, 0.35 * (1.0 - episode / args.episodes)))
        action = ACTIONS[action_index]
        enhanced = apply_filter(action, noisy, clean_sr)
        reward = si_sdr(clean, enhanced) - si_sdr(clean, noisy)
        # A very small complexity penalty breaks ties in favor of cheaper actions.
        if action != "keep_original":
            reward -= 0.01
        reward = float(np.clip(reward, -15.0, 15.0))
        agent.update(state, action_index, reward)
        rewards.append(reward)

        if (episode + 1) % max(1, args.episodes // 10) == 0 or episode == 0:
            recent = float(np.mean(rewards[-min(100, len(rewards)):]))
            print(f"Episode {episode + 1:>5}/{args.episodes} | recent mean reward: {recent:7.3f} dB | states: {len(agent.q_table)}")

    agent.save(args.output, metadata={
        "episodes": args.episodes,
        "clean_file_count": len(clean_files),
        "noise_file_count": len(noise_files),
        "seed": args.seed,
        "mean_training_reward_db": float(np.mean(rewards)),
        "training_seconds": round(time.time() - started, 2),
        "reward": "SI-SDR(output, clean) - SI-SDR(noisy, clean), minus 0.01 for non-identity actions",
        "note": "One-step episodic MDP/contextual action selection; reward requires clean reference during training only."
    })
    print(f"\nSaved policy to: {args.output}")
    print("Note: this prototype chooses one filter for each whole clip. It does not yet switch filters frame-by-frame.")


if __name__ == "__main__":
    main()
