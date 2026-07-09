"""Standalone collapse check for a trained checkpoint (PRD §6.5).

Encodes a fixed set of validation windows and reports the canary (mean
pairwise cosine similarity across windows) plus per-dimension std — the two
numbers that expose a collapsed representation.

    python -m eval.collapse_canary --run runs/phase4_baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.dataset import make_loader, shard_urls
from models.jepa_loss import collapse_canary
from sim.taxel_layout import TaxelLayout
from training.train import build_models, encode_batch


def load_run(run_dir: Path, device: str = "cpu"):
    import yaml

    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    layout = TaxelLayout.load()
    models = build_models(cfg, layout)
    ckpt = torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=True)
    # checkpoints save {"models": {...}, "optimizer": ..., "step": ...} to
    # support resuming preempted runs (see training/train.py)
    state = ckpt["models"] if "models" in ckpt else ckpt
    for k, m in models.items():
        m.load_state_dict(state[k])
        m.eval().to(device)
    return cfg, models


@torch.no_grad()
def canary_report(
    run_dir: Path, n_batches: int = 4, split: str = "val", shuffle: int = 0, seed: int = 0
) -> dict:
    cfg, models = load_run(run_dir)
    loader = make_loader(
        shard_urls(cfg["data"]["shard_dir"], split),
        batch_size=cfg["data"]["batch_size"],
        context_len=cfg["data"]["context_len"],
        horizon=cfg["data"]["horizon"],
        stride=cfg["data"]["stride"] * 4,  # spread windows out
        shuffle=shuffle,
        seed=seed,
    )
    latents = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        B, N = batch["B"], batch["N"]
        _, glob = encode_batch(models["encoder"], batch["context_batch"], B, N)
        last_idx = torch.arange(B) * N + (N - 1)
        latents.append(glob[last_idx])
    z = torch.cat(latents)
    return {
        "run": str(run_dir),
        "n_windows": int(z.shape[0]),
        "canary_cosine": collapse_canary(z),
        "mean_dim_std": float(z.std(dim=0).mean()),
        "min_dim_std": float(z.std(dim=0).min()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--shuffle", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(canary_report(args.run, split=args.split, shuffle=args.shuffle), indent=2))


if __name__ == "__main__":
    main()
