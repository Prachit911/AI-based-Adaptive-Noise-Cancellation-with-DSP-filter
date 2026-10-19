"""Optional Gradio web interface. Requires: pip install -e ".[app]" """
from __future__ import annotations

from pathlib import Path

try:
    import gradio as gr
except ImportError:
    gr = None

from .agent import QLearningFilterSelector
from .audio_utils import read_audio
from .features import extract_features, extract_state
from .filters import ACTIONS, apply_filter
from .selector import SupervisedFilterSelector

# Resolve models relative to project root so the app works from any cwd.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SELECTOR_PATH = _PROJECT_ROOT / "models" / "selector.joblib"
Q_PATH = _PROJECT_ROOT / "models" / "q_table.json"


def process_audio(filepath: str | None, method: str):
    if not filepath:
        raise gr.Error("Please upload an audio file first.")

    audio, sr = read_audio(filepath)

    if method == "Supervised Selector (Recommended)":
        if not SELECTOR_PATH.exists():
            raise gr.Error(
                f"Selector model not found at {SELECTOR_PATH}. Train it first using:\n"
                "python -m noise_suppression.train_selector"
            )
        selector = SupervisedFilterSelector.load(SELECTOR_PATH)
        features = extract_features(audio, sr)
        gains = selector.predict_gains(features)
        chosen = selector.select(features)
        result = apply_filter(chosen, audio, sr)

        sorted_gains = sorted(gains.items(), key=lambda x: -x[1])
        top_gains_str = "\n".join(f"  - {act:24s}: {g:+6.2f} dB" for act, g in sorted_gains)
        status = (
            f"[Selected] Supervised Selector picked: '{chosen}'\n"
            f"Predicted SI-SDR improvements:\n{top_gains_str}"
        )

    elif method == "Q-learning Agent":
        if not Q_PATH.exists():
            raise gr.Error(
                f"Q-table not found at {Q_PATH}. Train it first using:\n"
                "python -m noise_suppression.train_agent"
            )
        agent = QLearningFilterSelector.load(Q_PATH)
        state = extract_state(audio, sr)
        chosen = agent.select(state)
        result = apply_filter(chosen, audio, sr)
        status = f"[RL Agent] Q-learning selected: '{chosen}' (state: {state} | {sr} Hz)"

    else:
        chosen = method
        result = apply_filter(chosen, audio, sr)
        status = f"[Manual] Filter: '{chosen}' ({sr} Hz)"

    return (sr, result), status


def main():
    if gr is None:
        raise SystemExit(
            "Gradio is required to run the web interface.\n"
            "Please install it using:\n"
            "    pip install gradio\n"
            "or:\n"
            "    pip install -e \".[app]\""
        )
    choices = [
        "Supervised Selector (Recommended)",
        "Q-learning Agent",
        *ACTIONS,
    ]
    with gr.Blocks(title="AI-Based Adaptive Noise Cancellation") as demo:
        gr.Markdown("# 🎧 AI-Based Adaptive Speech Noise Cancellation with DSP Filters")
        gr.Markdown(
            "Upload a noisy speech recording, choose an automated selector or an individual "
            "DSP filter, and inspect the enhanced speech with predicted gain breakdown."
        )

        with gr.Row():
            input_audio = gr.Audio(label="Noisy speech input", type="filepath")
            output_audio = gr.Audio(label="Enhanced speech output", type="numpy")

        with gr.Row():
            method = gr.Dropdown(
                choices=choices,
                value="Supervised Selector (Recommended)",
                label="Enhancement Method",
            )
            run_btn = gr.Button("Suppress Noise", variant="primary")

        status = gr.Textbox(label="Processing Details & Predicted Gains", lines=7, interactive=False)
        run_btn.click(process_audio, inputs=[input_audio, method], outputs=[output_audio, status])

        gr.Markdown(
            "**Architecture:** Continuous acoustic features (MFCCs, spectral flux/flatness/entropy, "
            "stationarity, hum ratios) feed a Gradient Boosted Decision Tree bank that predicts the "
            "expected SI-SDR gain of every classical DSP filter, selecting the optimal treatment per clip."
        )

    demo.launch()


if __name__ == "__main__":
    main()
