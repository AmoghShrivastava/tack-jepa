"""Taxel force heatmap for an episode dump (Phase 2 exit criterion, PRD §9).

    python -m sim.visualize --episode episodes/phase1_smoke.npz --out docs/figures/phase2_heatmap.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim.episode_processing import (
    load_episode,
    synthesize_episode,
    taxel_world_positions_at_step,
)
from sim.taxel_layout import TaxelLayout


def plot_heatmap(episode_path: Path, out_path: Path) -> dict:
    layout = TaxelLayout.load()
    dump = load_episode(episode_path)
    data = synthesize_episode(dump, layout)
    mag = data.magnitude  # (S, T)

    total = mag.sum(axis=1)
    peak_step = int(total[50:].argmax()) + 50 if len(total) > 50 else int(total.argmax())

    fig = plt.figure(figsize=(15, 6))

    # 3D scatter of taxels at the peak-contact step, colored by force
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    wp = taxel_world_positions_at_step(layout, data, peak_step)
    m = mag[peak_step]
    order = np.argsort(m)  # draw loaded taxels on top
    sc = ax.scatter(
        wp[order, 0], wp[order, 1], wp[order, 2],
        c=m[order], cmap="inferno", s=6, vmin=0.0,
    )
    if "obj_pos" in dump:
        op = dump["obj_pos"][peak_step]
        ax.scatter([op[0]], [op[1]], [op[2]], c="tab:cyan", s=80, marker="o", label="object")
        ax.legend(loc="upper left")
    ax.set_title(f"per-taxel |force| at step {peak_step} (N)")
    ax.set_box_aspect((1, 1, 1))
    lims = np.array([wp.min(0), wp.max(0)])
    ctr, half = lims.mean(0), (lims[1] - lims[0]).max() / 2
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(ctr[2] - half, ctr[2] + half)
    fig.colorbar(sc, ax=ax, shrink=0.6)

    # force over time: total + per-link breakdown for the loaded links
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(total, label="total |force| across taxels", lw=2, c="k")
    per_link = np.stack(
        [mag[:, layout.link_index == li].sum(1) for li in range(len(layout.link_names))]
    )
    for li in np.argsort(per_link.max(1))[::-1][:4]:
        ax2.plot(per_link[li], label=layout.link_names[li], alpha=0.8)
    ax2.axvline(peak_step, ls="--", c="gray", lw=1)
    ax2.set_xlabel("step")
    ax2.set_ylabel("summed taxel |force| (N)")
    ax2.set_title("taxel force over the episode")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    active = int((mag[peak_step] > 0.01).sum())
    return {
        "peak_step": peak_step,
        "peak_total_force": float(total[peak_step]),
        "active_taxels_at_peak": active,
        "out": str(out_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=Path("episodes/phase1_smoke.npz"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures/phase2_heatmap.png"))
    args = parser.parse_args()
    info = plot_heatmap(args.episode, args.out)
    for k, v in info.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
