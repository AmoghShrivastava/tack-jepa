"""Plot loss + collapse-canary curves for one or more runs (Phase 4 evidence).

    python -m eval.plot_training --runs runs/phase4_* --out docs/figures/phase4_training.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(run_dir: Path) -> list[dict]:
    lines = (run_dir / "metrics.jsonl").read_text().strip().splitlines()
    return [json.loads(ln) for ln in lines]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("docs/figures/phase4_training.png"))
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for run in args.runs:
        m = load_metrics(run)
        steps = [r["step"] for r in m]
        name = run.name
        axes[0].plot(steps, [r["loss_pred"] for r in m], label=name)
        axes[1].plot(steps, [r["canary_cosine"] for r in m], label=name)
        axes[2].plot(steps, [r.get("vicreg_var", float("nan")) for r in m], label=name)
    axes[0].set_title("prediction loss")
    axes[0].set_yscale("log")
    axes[1].set_title("collapse canary (pairwise cosine, lower=healthier)")
    axes[1].set_ylim(0, 1.05)
    axes[2].set_title("VICReg variance hinge (1 = fully collapsed dims)")
    for ax in axes:
        ax.set_xlabel("step")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
