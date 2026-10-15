"""Apply one manually chosen DSP filter or the trained RL-selected filter."""
from __future__ import annotations
import argparse
from pathlib import Path
from .audio_utils import read_audio, write_audio
from .features import extract_state
from .filters import ACTIONS, apply_filter
from .agent import QLearningFilterSelector


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhance noisy speech with a DSP filter or learned RL selector.")
    parser.add_argument("--input", required=True, help="Input noisy audio file")
    parser.add_argument("--output", default="outputs/enhanced.wav", help="Output WAV file")
    parser.add_argument("--method", choices=["auto", *ACTIONS], default="auto", help="auto uses the trained Q-table")
    parser.add_argument("--q-table", default="models/q_table.json", help="Trained policy path used by --method auto")
    args = parser.parse_args()

    audio, sample_rate = read_audio(args.input)
    if args.method == "auto":
        q_path = Path(args.q_table)
        if not q_path.exists():
            raise SystemExit(f"No trained Q-table found at {q_path}. Run the training command from README.md first.")
        agent = QLearningFilterSelector.load(q_path)
        state = extract_state(audio, sample_rate)
        method = agent.select(state)
        print(f"Detected state: {state}; RL-selected filter: {method}")
    else:
        method = args.method
        print(f"Using manual filter: {method}")

    enhanced = apply_filter(method, audio, sample_rate)
    write_audio(args.output, enhanced, sample_rate)
    print(f"Saved enhanced audio to: {args.output}")


if __name__ == "__main__":
    main()
