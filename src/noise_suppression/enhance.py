"""Apply one manually chosen DSP filter, the tabular Q-agent, or the Supervised Filter Selector."""
from __future__ import annotations

import argparse
from pathlib import Path

from .agent import QLearningFilterSelector
from .audio_utils import read_audio, write_audio
from .features import extract_features, extract_state
from .filters import ACTIONS, apply_filter
from .selector import SupervisedFilterSelector


def enhance_audio(
    input_path: Path | str,
    output_path: Path | str,
    method: str = "selector",
    selector_path: Path | str = "models/selector.joblib",
    q_table_path: Path | str = "models/q_table.json",
) -> tuple[str, str]:
    """Enhance a single audio file and return (chosen_action, details_message)."""
    audio, sample_rate = read_audio(input_path)

    if method == "selector":
        sel_file = Path(selector_path)
        if not sel_file.exists():
            raise FileNotFoundError(
                f"Selector model not found at {sel_file}. "
                "Train it first using: python -m noise_suppression.train_selector"
            )
        selector = SupervisedFilterSelector.load(sel_file)
        features = extract_features(audio, sample_rate)
        gains = selector.predict_gains(features)
        chosen_filter = selector.select(features)

        # Build top 3 gain predictions
        sorted_gains = sorted(gains.items(), key=lambda x: -x[1])
        top_str = ", ".join(f"{act}: {gain:+.2f} dB" for act, gain in sorted_gains[:3])
        details = (
            f"Supervised Selector chosen filter: '{chosen_filter}' "
            f"(predicted delta SI-SDR: {gains.get(chosen_filter, 0.0):+.2f} dB | Top: {top_str})"
        )

    elif method == "auto":
        q_file = Path(q_table_path)
        if not q_file.exists():
            raise FileNotFoundError(
                f"No trained Q-table found at {q_file}. "
                "Train it first using: python -m noise_suppression.train_agent"
            )
        agent = QLearningFilterSelector.load(q_file)
        state = extract_state(audio, sample_rate)
        chosen_filter = agent.select(state)
        details = f"Q-Learning chosen filter: '{chosen_filter}' for state: '{state}'"

    elif method in ACTIONS:
        chosen_filter = method
        details = f"Manual filter: '{chosen_filter}'"

    else:
        raise ValueError(f"Unknown enhancement method: {method}")

    enhanced = apply_filter(chosen_filter, audio, sample_rate)
    write_audio(output_path, enhanced, sample_rate)
    return chosen_filter, details


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enhance noisy speech with a DSP filter, Q-learning policy, or Supervised Selector."
    )
    parser.add_argument("--input", required=True, help="Input noisy audio file")
    parser.add_argument("--output", default="outputs/enhanced.wav", help="Output WAV file")
    parser.add_argument(
        "--method",
        choices=["selector", "auto", *ACTIONS],
        default="selector",
        help="Enhancement method ('selector' = GBRT selector, 'auto' = Q-table, or a specific filter name)",
    )
    parser.add_argument("--selector", default="models/selector.joblib", help="Trained selector path")
    parser.add_argument("--q-table", default="models/q_table.json", help="Trained Q-table path")
    args = parser.parse_args()

    chosen, details = enhance_audio(
        input_path=args.input,
        output_path=args.output,
        method=args.method,
        selector_path=args.selector,
        q_table_path=args.q_table,
    )
    print(details)
    print(f"Saved enhanced audio to: {args.output}")


if __name__ == "__main__":
    main()
