"""Optional Gradio web interface. Requires: pip install -e ".[app]"""
from __future__ import annotations
from pathlib import Path
import gradio as gr
from .audio_utils import read_audio
from .features import extract_state
from .filters import ACTIONS, apply_filter
from .agent import QLearningFilterSelector

Q_PATH = Path("models/q_table.json")


def process_audio(filepath: str | None, method: str):
    if not filepath:
        raise gr.Error("Upload an audio file first.")
    audio, sr = read_audio(filepath)
    if method == "RL auto-select":
        if not Q_PATH.exists():
            raise gr.Error("The RL policy is not trained yet. Run the training command from README.md first.")
        agent = QLearningFilterSelector.load(Q_PATH)
        state = extract_state(audio, sr)
        selected = agent.select(state)
        result = apply_filter(selected, audio, sr)
        status = f"RL selected: {selected} | state: {state} | sample rate: {sr} Hz"
    else:
        selected = method
        result = apply_filter(selected, audio, sr)
        status = f"Filter: {selected} | sample rate: {sr} Hz"
    return (sr, result), status


def main():
    choices = ["RL auto-select", *ACTIONS]
    with gr.Blocks(title="AI-Based Intelligent Noise Suppression") as demo:
        gr.Markdown("# AI-Based Intelligent Noise Suppression\nUpload a noisy speech recording, select a DSP filter manually or let the Q-learning policy choose one, then compare the result.")
        with gr.Row():
            input_audio = gr.Audio(label="Noisy speech input", type="filepath")
            output_audio = gr.Audio(label="Enhanced speech output", type="numpy")
        method = gr.Dropdown(choices=choices, value="RL auto-select", label="Filter selection")
        run = gr.Button("Suppress noise", variant="primary")
        status = gr.Textbox(label="Processing details", interactive=False)
        run.click(process_audio, inputs=[input_audio, method], outputs=[output_audio, status])
        gr.Markdown("**Note:** Classical filters have different strengths and limitations. The prototype chooses one filter for the whole clip; it does not yet switch filters frame-by-frame.")
    demo.launch()


if __name__ == "__main__":
    main()
