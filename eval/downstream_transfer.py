"""Downstream task transfer + sample efficiency (PRD §7.4).

Tasks:
  slip_onset      — from the latent at t, predict slip occurring at t+H that is
                    NOT yet present at t (exercises the predictive property).
                    Uses existing Stage A shards.
  grasp_stability — will the grasp survive an object nudge? Needs perturbation
                    episodes from `python -m sim.episode_generator --mode perturb`.

Encoders compared (§7.4): a pretrained frozen encoder (any run dir), the
image-native ablation's frozen encoder, and a no-pretraining baseline (same
task head on raw taxel input). Sample-efficiency curves = task metric vs
number of labeled episodes.

    python -m eval.downstream_transfer --task slip_onset \
        --runs runs/phase4_baseline runs/phase4_image_native --episodes 8 16 32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data.dataset import TaxelSequenceDataset, collate_sequences, shard_urls
from eval.collapse_canary import load_run
from eval.metrics import binary_metrics
from training.train import encode_batch

H_ONSET = 5  # predict slip H steps ahead


class RawInputBaseline(nn.Module):
    """No-pretraining control: the same MLP head on pooled raw taxel input."""

    def __init__(self, out_dim: int = 512):
        super().__init__()
        # per-taxel raw features: force (3); pooled mean+max over taxels
        self.net = nn.Sequential(nn.Linear(6, 256), nn.GELU(), nn.Linear(256, out_dim))

    def forward(self, batch, B, N):
        force = batch.force.view(B * N, -1, 3)
        pooled = torch.cat([force.mean(1), force.amax(1)], dim=1)
        return None, self.net(pooled)


def _windows(shard_dir, split, context_len, horizon, stride, limit_episodes=None):
    """Window samples grouped per episode (for episode-count subsetting)."""
    ds = TaxelSequenceDataset(
        shard_urls(shard_dir, split),
        context_len=context_len,
        horizon=horizon,
        stride=stride,
    )
    grouped: dict[str, list[dict]] = {}
    for w in ds:
        key = w["episode"]
        if limit_episodes and key not in grouped and len(grouped) >= limit_episodes:
            break
        grouped.setdefault(key, []).append(w)
    return list(grouped.values())


def _slip_onset_label(sample) -> float:
    """Positive when the target step slips somewhere the context end does not."""
    now = float(sample["context"][-1].slip.sum())
    future = float(sample["target"].slip.sum())
    return 1.0 if (future > 0 and now == 0) else 0.0


def eval_slip_onset(
    run_dir: Path | None,
    shard_dir: str = "datasets/shards",
    episode_budgets: tuple = (8, 16, 32),
    train_steps_per_budget: int = 120,
    batch_size: int = 8,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """One encoder (or raw baseline when run_dir is None) x sample budgets."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    if run_dir is not None:
        cfg, models = load_run(run_dir, device)
        encoder = models["encoder"]
        for p in encoder.parameters():
            p.requires_grad_(False)
        global_dim = cfg["model"]["global_dim"]
        N = cfg["data"]["context_len"]
        name = f"{cfg['model']['variant']}:{Path(run_dir).name}"
        raw = None
    else:
        N = 8
        global_dim = 512
        raw = RawInputBaseline(global_dim).to(device)
        name = "raw_no_pretrain"

    def encode(ctx_batch, B):
        if raw is None:
            with torch.no_grad():
                _, glob = encode_batch(encoder, ctx_batch, B, N)
        else:
            _, glob = raw(ctx_batch, B, N)
        return glob.view(B, N, -1)[:, -1]

    train_eps = _windows(shard_dir, "train", N, H_ONSET, 4, limit_episodes=max(episode_budgets))
    val_eps = _windows(shard_dir, "val", N, H_ONSET, 8, limit_episodes=24)
    val_windows = [w for ep in val_eps for w in ep]

    results = {}
    for budget in episode_budgets:
        sel = rng.permutation(len(train_eps))[:budget]
        pool = [w for i in sel for w in train_eps[i]]
        labels = np.array([_slip_onset_label(w) for w in pool])
        head = nn.Sequential(
            nn.Linear(global_dim, 128), nn.GELU(), nn.Linear(128, 1)
        ).to(device)
        params = list(head.parameters()) + ([] if raw is None else list(raw.parameters()))
        opt = torch.optim.AdamW(params, lr=1e-3)
        pos_frac = labels.mean()
        pw = torch.tensor((1 - pos_frac) / max(pos_frac, 1e-4)).clamp(1.0, 200.0)

        for _step in range(train_steps_per_budget):
            idx = rng.integers(0, len(pool), size=min(batch_size, len(pool)))
            batch = collate_sequences([pool[i] for i in idx])
            z = encode(batch["context_batch"].to(device), batch["B"])
            y = torch.tensor(labels[idx], dtype=torch.float32, device=device)
            loss = nn.functional.binary_cross_entropy_with_logits(
                head(z).squeeze(-1), y, pos_weight=pw
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # evaluate
        logits, ys = [], []
        with torch.no_grad():
            for i in range(0, len(val_windows), batch_size):
                chunk = val_windows[i : i + batch_size]
                batch = collate_sequences(chunk)
                z = encode(batch["context_batch"].to(device), batch["B"])
                logits.append(head(z).squeeze(-1).cpu().numpy())
                ys.append([_slip_onset_label(w) for w in chunk])
        results[budget] = binary_metrics(np.concatenate(logits), np.concatenate(ys))
    return {"encoder": name, "task": "slip_onset", "curve": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["slip_onset"], default="slip_onset")
    parser.add_argument("--runs", nargs="*", type=Path, default=[])
    parser.add_argument("--include-raw-baseline", action="store_true", default=True)
    parser.add_argument("--episodes", nargs="*", type=int, default=[8, 16, 32])
    parser.add_argument("--out", type=Path, default=Path("runs/downstream_transfer.json"))
    args = parser.parse_args()

    reports = []
    for run in args.runs:
        reports.append(eval_slip_onset(run, episode_budgets=tuple(args.episodes)))
    if args.include_raw_baseline:
        reports.append(eval_slip_onset(None, episode_budgets=tuple(args.episodes)))
    text = json.dumps(reports, indent=2)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)


if __name__ == "__main__":
    main()
