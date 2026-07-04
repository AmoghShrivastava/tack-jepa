"""Frozen-encoder physics probing (PRD §7.3) — the primary intrinsic eval.

Freezes a trained encoder, trains fresh probe heads on train-split latents,
and reports on the object-disjoint val split:
  - per-taxel force magnitude regression: MAE, R^2
  - per-taxel slip classification: accuracy, F1, AUROC
  - contact area regression: MAE, R^2
plus the collapse canary. Identical procedure for the baseline and every
ablation checkpoint, which is what makes §7.2 comparisons meaningful.

    python -m eval.physics_probes_eval --run runs/phase4_baseline --steps 150
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.dataset import make_loader, shard_urls
from eval.collapse_canary import load_run
from eval.metrics import binary_metrics, mae, r2
from models.probes import PhysicsProbes


def _iter_latents(models, loader, device, max_batches=None):
    """Yield (node_latent, global_latent, labels) for each window's last step."""
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        B, N = batch["B"], batch["N"]
        ctx = batch["context_batch"].to(device)
        with torch.no_grad():
            from training.train import encode_batch

            node, glob = encode_batch(models["encoder"], ctx, B, N)
        last_idx = torch.arange(B, device=device) * N + (N - 1)
        node_mask = torch.isin(ctx.batch, last_idx)
        yield (
            node[node_mask],
            glob[last_idx],
            {
                "force_mag": ctx.force_mag[node_mask],
                "slip": ctx.slip[node_mask],
                "contact_area": ctx.contact_area.view(B, N)[:, -1],
            },
        )


def probe_eval(
    run_dir: Path,
    train_steps: int = 150,
    eval_batches: int = 30,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    cfg, models = load_run(run_dir, device)
    m = cfg["model"]

    def loader_for(split, stride_mult=1):
        return make_loader(
            shard_urls(cfg["data"]["shard_dir"], split),
            batch_size=cfg["data"]["batch_size"],
            context_len=cfg["data"]["context_len"],
            horizon=cfg["data"]["horizon"],
            stride=cfg["data"]["stride"] * stride_mult,
            shuffle=1,
            seed=seed,
        )

    probes = PhysicsProbes(node_dim=m["node_out"], global_dim=m["global_dim"]).to(device)
    opt = torch.optim.AdamW(probes.parameters(), lr=lr)

    # ---- train fresh probes on frozen train-split latents
    step = 0
    while step < train_steps:
        for node, glob, labels in _iter_latents(models, loader_for("train"), device):
            if step >= train_steps:
                break
            out = probes(node, glob)
            losses = PhysicsProbes.losses(out, **labels)
            opt.zero_grad(set_to_none=True)
            sum(losses.values()).backward()
            opt.step()
            step += 1

    # ---- evaluate on the object-disjoint val split
    preds: dict[str, list] = {k: [] for k in ("force_mag", "slip_logit", "contact_area")}
    trues: dict[str, list] = {k: [] for k in ("force_mag", "slip", "contact_area")}
    probes.eval()
    with torch.no_grad():
        for node, glob, labels in _iter_latents(
            models, loader_for("val", 2), device, max_batches=eval_batches
        ):
            out = probes(node, glob)
            for k in preds:
                preds[k].append(out[k].cpu().numpy())
            for k in trues:
                trues[k].append(labels[k].cpu().numpy())
    p = {k: np.concatenate(v) for k, v in preds.items()}
    t = {k: np.concatenate(v) for k, v in trues.items()}

    report = {
        "run": str(run_dir),
        "variant": cfg["model"]["variant"],
        "probe_train_steps": train_steps,
        "force_mag": {
            "mae": mae(p["force_mag"], t["force_mag"]),
            "r2": r2(p["force_mag"], t["force_mag"]),
        },
        "slip": binary_metrics(p["slip_logit"], t["slip"]),
        "contact_area": {
            "mae": mae(p["contact_area"], t["contact_area"]),
            "r2": r2(p["contact_area"], t["contact_area"]),
        },
    }
    # canary over val latents (fresh pass, cheap)
    from eval.collapse_canary import canary_report

    report["canary"] = canary_report(run_dir, n_batches=4, split="val")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--eval-batches", type=int, default=30)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = probe_eval(args.run, args.steps, args.eval_batches)
    text = json.dumps(report, indent=2)
    print(text)
    out = args.out or (args.run / "probe_eval.json")
    out.write_text(text)


if __name__ == "__main__":
    main()
