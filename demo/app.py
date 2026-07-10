"""TacK-JEPA interactive demo — Gradio Space.

Shows real per-taxel contact-force data from held-out (object-disjoint)
Stage C episodes, computed by the actual project pipeline
(sim/taxel_force_synthesis.py + sim/forward_kinematics.py), for three
grasp objects the model never saw during training.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

ASSETS = Path(__file__).parent / "assets" / "heatmaps"

EPISODES = {
    "Object A — sphere (held out)": {
        "image": str(ASSETS / "object_A.png"),
        "peak_force": "4.3 N",
        "active_taxels": "74 / 2,244",
        "contact_step": "60",
    },
    "Object B — box (held out)": {
        "image": str(ASSETS / "object_B.png"),
        "peak_force": "25.4 N",
        "active_taxels": "97 / 2,244",
        "contact_step": "87",
    },
    "Object C — box, larger grasp (held out)": {
        "image": str(ASSETS / "object_C.png"),
        "peak_force": "80.4 N",
        "active_taxels": "139 / 2,244",
        "contact_step": "151",
    },
}

CSS = """
.gradio-container {background: #ffffff !important; font-family: 'IBM Plex Mono', ui-monospace, monospace;}
.gradio-container * {color: #111 !important;}
.prose, .prose p, .prose h3 {color: #111 !important; opacity: 1 !important;}
#panel {border: 1px solid #111; border-radius: 4px; padding: 16px;}
.stat {border-left: 2px solid #111; padding-left: 10px;}
label span {font-weight:600 !important;}
"""

with gr.Blocks(css=CSS, theme=gr.themes.Monochrome(), title="TacK-JEPA — live probe demo") as demo:
    with gr.Row():
        picker = gr.Dropdown(
            choices=list(EPISODES.keys()),
            value=list(EPISODES.keys())[0],
            label="Held-out grasp episode",
        )
    with gr.Row():
        with gr.Column(scale=3):
            img = gr.Image(value=EPISODES[list(EPISODES.keys())[0]]["image"], show_label=False)
        with gr.Column(scale=1, elem_id="panel"):
            peak = gr.Textbox(label="Peak grasp force", value=EPISODES[list(EPISODES.keys())[0]]["peak_force"], interactive=False)
            taxels = gr.Textbox(label="Active taxels at peak contact", value=EPISODES[list(EPISODES.keys())[0]]["active_taxels"], interactive=False)
            step = gr.Textbox(label="Contact step reached", value=EPISODES[list(EPISODES.keys())[0]]["contact_step"], interactive=False)

    def update(name: str):
        e = EPISODES[name]
        return e["image"], e["peak_force"], e["active_taxels"], e["contact_step"]

    picker.change(update, inputs=picker, outputs=[img, peak, taxels, step])

if __name__ == "__main__":
    demo.launch()
