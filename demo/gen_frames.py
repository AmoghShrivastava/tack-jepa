"""Generate scrubbable frame sequences for the demo site.

For each chosen episode, renders N frames spanning the real contact window
(from first meaningful contact to just past peak grasp force), so the site
can offer an actual step-through of the simulated grasp rather than one
static peak frame.
"""

from __future__ import annotations

import time
from pathlib import Path

BUILD_VERSION = int(time.time())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim.episode_processing import load_episode, synthesize_episode, taxel_world_positions_at_step
from sim.taxel_layout import TaxelLayout

OUT = Path(__file__).parent / "site" / "assets" / "frames"
N_BUILD = 14   # contact-begins -> peak
N_RELEASE = 5  # peak -> release

EPISODES = {
    "1": (2, 13),
    "2": (7, 19),
    "3": (12, 17),
    "4": (2, 17),
    "5": (12, 19),
}


def render_frame(layout, data, dump, step, out_path, vmax):
    mag = data.magnitude
    wp = taxel_world_positions_at_step(layout, data, step)
    m = mag[step]
    order = np.argsort(m)

    fig = plt.figure(figsize=(7, 6.2))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    sc = ax.scatter(wp[order, 0], wp[order, 1], wp[order, 2], c=m[order], cmap="inferno", s=10, vmin=0.0, vmax=vmax)
    if "obj_pos" in dump:
        op = dump["obj_pos"][step]
        ax.scatter([op[0]], [op[1]], [op[2]], c="tab:cyan", s=110, marker="o")
    ax.set_title(f"step {step}", fontsize=14)
    ax.set_box_aspect((1, 1, 1))
    lims = np.array([wp.min(0), wp.max(0)])
    ctr, half = lims.mean(0), (lims[1] - lims[0]).max() / 2
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(ctr[2] - half, ctr[2] + half)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.08)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    layout = TaxelLayout.load()
    manifest = {}
    for key, (variant, idx) in EPISODES.items():
        p = Path(f"datasets/stage_c/ep_{variant}_{idx:04d}.npz")
        dump = load_episode(p)
        data = synthesize_episode(dump, layout)
        mag = data.magnitude
        total = mag.sum(axis=1)
        seg = total[50:] if len(total) > 50 else total
        peak_step = int(seg.argmax()) + (50 if len(total) > 50 else 0)
        peak = total[peak_step]
        vmax = float(mag[peak_step].max())

        # window centered on the true peak (a narrow spike, so build up to it
        # explicitly rather than evenly sampling the whole episode and risking
        # missing it), then a separate finer release window after
        window_start = max(0, peak_step - 48)
        build_steps = np.linspace(window_start, peak_step, N_BUILD).astype(int)
        release_end = min(len(total) - 1, peak_step + 18)
        release_steps = np.linspace(peak_step, release_end, N_RELEASE + 1)[1:].astype(int)
        steps = sorted(set(build_steps.tolist() + release_steps.tolist()))
        peak_idx = steps.index(peak_step)

        frames = []
        for i, s in enumerate(steps):
            out_path = OUT / f"ep{key}_f{i}.png"
            render_frame(layout, data, dump, s, out_path, vmax)
            active = int((mag[s] > 0.01).sum())
            if i < peak_idx:
                stage = "contact forming" if i < peak_idx * 0.6 else "grasp tightening"
            elif i == peak_idx:
                stage = "peak grasp"
            else:
                stage = "release"
            frames.append({
                "step": int(s),
                "force": round(float(total[s]), 2),
                "active": active,
                "stage": stage,
                "img": f"assets/frames/ep{key}_f{i}.png?v={BUILD_VERSION}",
            })
        manifest[key] = frames
        print(key, variant, idx, "peak", round(peak, 2), "frames", len(frames))

    import json
    (Path(__file__).parent / "site" / "assets" / "frames.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
