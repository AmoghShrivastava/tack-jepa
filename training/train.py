"""TacK-JEPA training loop (PRD §5, §6) — single-device; DDP comes at Phase 6.

Runs the baseline and every §7.2 ablation from one entry point:

    python -m training.train --variant baseline train.steps=300
    python -m training.train --variant no_fk ...

Each run writes runs/<name>/resolved_config.yaml, metrics.jsonl, and
checkpoint.pt. The collapse canary (§6.5) is logged from step one.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from pathlib import Path

import numpy as np
import torch

from data.dataset import make_loader, shard_urls
from models.ablations.image_native import TactileImageEncoder
from models.ablations.reconstruction import RawForceDecoder
from models.encoder import TaxelGraphEncoder
from models.jepa_loss import collapse_canary, jepa_latent_loss, vicreg_regularizer
from models.predictor import ActionConditionedPredictor
from models.probes import PhysicsProbes
from sim.taxel_layout import TaxelLayout
from training.config import dump_config, load_config
from training.curriculum import make_horizon_schedule
from training.ema import ema_momentum, ema_update, make_target

JEPA_VARIANTS = ("baseline", "no_fk", "no_vicreg", "image_native")


def build_models(cfg: dict, layout: TaxelLayout) -> dict[str, torch.nn.Module]:
    m = cfg["model"]
    variant = m["variant"]
    if variant == "image_native":
        enc = TactileImageEncoder(
            layout=layout,
            dim=m["hidden"],
            n_layers=m["n_layers"],
            heads=m["heads"],
            node_out=m["node_out"],
            global_dim=m["global_dim"],
        )
    else:
        enc = TaxelGraphEncoder(
            n_links=m["n_links"],
            link_emb_dim=m["link_emb_dim"],
            hidden=m["hidden"],
            n_layers=m["n_layers"],
            heads=m["heads"],
            node_out=m["node_out"],
            global_dim=m["global_dim"],
            use_geometry=(variant != "no_fk"),
        )
    models: dict[str, torch.nn.Module] = {
        "encoder": enc,
        "predictor": ActionConditionedPredictor(
            dim=m["global_dim"],
            n_layers=cfg["predictor"]["n_layers"],
            heads=cfg["predictor"]["heads"],
            context_len=cfg["data"]["context_len"],
            max_horizon=cfg["predictor"]["max_horizon"],
        ),
        "probes": PhysicsProbes(node_dim=m["node_out"], global_dim=m["global_dim"]),
    }
    if variant in JEPA_VARIANTS:
        models["target_encoder"] = make_target(enc)
    else:  # reconstruction: no target encoder, decode raw future forces
        models["decoder"] = RawForceDecoder(
            dim=m["global_dim"], n_taxels=layout.n_taxels
        )
    return models


def encode_batch(encoder, batch, B, N):
    node, glob = encoder(
        force=batch.force,
        link_index=batch.link_id,
        edge_index=batch.edge_index,
        batch=batch.batch,
        pos=batch.pos,
        normal=batch.normal,
        qpos=batch.qpos,
    )
    return node, glob


def lr_at(step, cfg_train):
    warmup, total = cfg_train["warmup_steps"], cfg_train["steps"]
    base = cfg_train["lr"]
    if step < warmup:
        return base * (step + 1) / warmup
    frac = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0)))


_AMP_DTYPES = {"fp32": None, "bf16": torch.bfloat16}


def amp_context(device: torch.device, precision: str):
    """PRD §6.2/§8: bf16 mixed precision (Nebius H100-class GPUs support this
    natively). No GradScaler needed — bf16's exponent range doesn't require
    loss scaling the way fp16 does. Works (functionally, not for speed) on
    CPU too, so the code path is exercised and verified before Phase 6."""
    dtype = _AMP_DTYPES[precision]
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None)


def train(cfg: dict) -> dict:
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device(cfg["train"]["device"])
    layout = TaxelLayout.load()
    variant = cfg["model"]["variant"]
    is_jepa = variant in JEPA_VARIANTS

    out_dir = Path(cfg["out_dir"]) / cfg["run_name"]
    dump_config(cfg, out_dir)
    metrics_path = out_dir / "metrics.jsonl"

    import wandb

    wandb.init(
        mode=cfg["wandb"]["mode"],
        project=cfg["wandb"]["project"],
        name=cfg["run_name"],
        config=cfg,
        dir=str(out_dir),
    )

    urls = shard_urls(cfg["data"]["shard_dir"], "train")
    if not urls:
        raise FileNotFoundError(f"no train shards under {cfg['data']['shard_dir']}")

    # Horizon curriculum (PRD §5.7/§6.2): k=1 -> max_horizon as training
    # stabilizes. Each distinct horizon needs its own loader (windows are
    # built at a fixed k), so build one per curriculum stage and switch
    # between them by step according to the schedule.
    if cfg["data"].get("horizon_curriculum", True):
        horizon_at, stage_ks = make_horizon_schedule(
            cfg["train"]["steps"], cfg["predictor"]["max_horizon"]
        )
    else:
        stage_ks = [cfg["data"]["horizon"]]
        horizon_at = lambda step: stage_ks[0]  # noqa: E731

    # Gradient accumulation: the full-size model's per-graph memory (GATv2
    # edge-level attention over ~2200 taxels x context_len graphs) OOMs well
    # below PRD §6.2's target batch of 32-64 on a single GPU (empirically: a
    # 46GB L40S fits micro-batch 4-5, not 32) — verified during the Phase 6
    # validation run, see ROADMAP.md. micro_batch_size is the actual
    # memory-bound per-forward-pass size; data.batch_size stays the semantic
    # "effective batch" from the PRD, reached by accumulating gradients.
    micro_bs = cfg["train"].get("micro_batch_size", cfg["data"]["batch_size"])
    accum_steps = max(1, math.ceil(cfg["data"]["batch_size"] / micro_bs))

    def make_iter(k):
        loader = make_loader(
            urls,
            batch_size=micro_bs,
            context_len=cfg["data"]["context_len"],
            horizon=k,
            stride=cfg["data"]["stride"],
            shuffle=1,
            seed=cfg["seed"],
        )

        def infinite():
            while True:
                yield from loader

        return infinite()

    iters = {k: make_iter(k) for k in stage_ks}

    models = build_models(cfg, layout)
    for mod in models.values():
        mod.to(device)
    trainable = [p for k, m in models.items() if k != "target_encoder" for p in m.parameters()]
    opt = torch.optim.AdamW(
        trainable, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )

    tr = cfg["train"]
    step = 0
    t0 = time.time()
    history: list[dict] = []
    precision = tr.get("precision", "fp32")
    checkpoint_every = tr.get("checkpoint_every", 500)
    ckpt_path = out_dir / "checkpoint.pt"

    # Resume support: a preemptible instance can be stopped mid-run (Nebius
    # sends SIGTERM 60s before) — periodic checkpoints below + this resume
    # path mean re-running the same run_name picks up where it left off
    # instead of losing everything.
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        for k, m in models.items():
            m.load_state_dict(ckpt["models"][k])
        opt.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        print(f"[{cfg['run_name']}] resumed from checkpoint at step {step}", flush=True)

    def save_checkpoint(at_step: int):
        torch.save(
            {
                "models": {k: m.state_dict() for k, m in models.items()},
                "optimizer": opt.state_dict(),
                "step": at_step,
            },
            ckpt_path,
        )

    # Nebius preemptible instances get SIGTERM ~60s before the VM is actually
    # killed. checkpoint_every alone can still lose up to that many steps; the
    # handler just sets a flag (signal handlers must stay minimal/async-safe)
    # that the loop checks after each completed step and reacts to by saving
    # immediately and exiting, so a preemption costs ~0 steps instead of up to
    # checkpoint_every.
    _preempted = {"flag": False}

    def _on_sigterm(signum, frame):
        _preempted["flag"] = True

    signal.signal(signal.SIGTERM, _on_sigterm)

    while step < tr["steps"]:
        k = horizon_at(step)
        opt.zero_grad(set_to_none=True)
        accum: dict[str, float] = {}
        last_canary_z = None

        # Gradient accumulation over accum_steps micro-batches (see above) —
        # each micro-batch's loss is scaled by 1/accum_steps before backward,
        # so the accumulated gradient matches one true effective-batch step.
        for _micro in range(accum_steps):
            batch = next(iters[k])
            B, N = batch["B"], batch["N"]
            assert batch["horizon"] == k
            ctx = batch["context_batch"].to(device)
            tgt = batch["target_batch"].to(device)
            actions = batch["actions"].to(device)

            # Everything through the loss (encode/predict/VICReg/probes) runs
            # under one autocast context, matching standard AMP usage — mixing
            # bf16 activations with fp32-only ops outside the context would
            # otherwise error the moment precision="bf16" is actually used.
            with amp_context(device, precision):
                node_ctx, glob_ctx = encode_batch(models["encoder"], ctx, B, N)
                ctx_seq = glob_ctx.view(B, N, -1)
                pred = models["predictor"](ctx_seq, actions, horizon=k)

                if is_jepa:
                    with torch.no_grad():
                        _, glob_tgt = encode_batch(models["target_encoder"], tgt, B, 1)
                    loss_pred = jepa_latent_loss(pred, glob_tgt)
                else:
                    decoded = models["decoder"](pred)
                    target_force = tgt.force.view(B, layout.n_taxels, 3)
                    loss_pred = RawForceDecoder.loss(decoded, target_force)
                loss = loss_pred
                micro_logs = {"loss_pred": float(loss_pred.detach())}

                if tr["vicreg_var_weight"] or tr["vicreg_cov_weight"]:
                    var_l, cov_l = vicreg_regularizer(glob_ctx)
                    loss = loss + tr["vicreg_var_weight"] * var_l + tr["vicreg_cov_weight"] * cov_l
                    micro_logs["vicreg_var"] = float(var_l)
                    micro_logs["vicreg_cov"] = float(cov_l)

                # probes on the LAST context step's latents (detached by default, §5.10)
                last_idx = (
                    torch.arange(B, device=device) * N + (N - 1)
                )  # graph ids of last ctx steps
                node_mask = torch.isin(ctx.batch, last_idx)
                nl = node_ctx[node_mask]
                gl = glob_ctx[last_idx]
                if not tr["probe_grad_to_encoder"]:
                    nl, gl = nl.detach(), gl.detach()
                probe_out = models["probes"](nl, gl)
                probe_losses = PhysicsProbes.losses(
                    probe_out,
                    force_mag=ctx.force_mag[node_mask],
                    slip=ctx.slip[node_mask],
                    contact_area=ctx.contact_area.view(B, N)[:, -1],
                    contact_area_scale=layout.n_taxels,
                )
                probe_total = sum(probe_losses.values())
                loss = loss + tr["probe_weight"] * probe_total
                micro_logs.update({f"probe_{k_}": float(v) for k_, v in probe_losses.items()})
                micro_logs["loss_total"] = float(loss.detach())

            (loss / accum_steps).backward()
            for name, val in micro_logs.items():
                accum[name] = accum.get(name, 0.0) + val / accum_steps
            last_canary_z = glob_ctx[last_idx].detach()

        torch.nn.utils.clip_grad_norm_(trainable, tr["grad_clip"])
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tr)
        opt.step()

        if is_jepa:
            m = ema_momentum(step, tr["steps"], tr["ema_start"], tr["ema_end"])
            ema_update(models["target_encoder"], models["encoder"], m)
            accum["ema_momentum"] = m

        logs = {"horizon": k, **accum}
        # canary from the last micro-batch's windows — within-window steps of
        # a static press are near-identical by nature and would mask collapse
        # detection, so this is measured across different windows already
        logs["canary_cosine"] = collapse_canary(last_canary_z)
        logs["lr"] = opt.param_groups[0]["lr"]
        logs["step"] = step
        logs["effective_batch"] = micro_bs * accum_steps
        history.append(logs)

        if step % tr["log_every"] == 0 or step == tr["steps"] - 1:
            wandb.log(logs, step=step)
            with open(metrics_path, "a") as f:
                f.write(json.dumps(logs) + "\n")
            print(
                f"[{cfg['run_name']}] step {step:4d} "
                f"loss {logs['loss_total']:.4f} pred {logs['loss_pred']:.4f} "
                f"canary {logs['canary_cosine']:.3f} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
        step += 1
        if step % checkpoint_every == 0:
            save_checkpoint(step)
        if _preempted["flag"]:
            print(
                f"[{cfg['run_name']}] SIGTERM received at step {step} — "
                "saving checkpoint immediately, exiting",
                flush=True,
            )
            save_checkpoint(step)
            wandb.finish()
            return {
                "run_name": cfg["run_name"], "variant": variant, "steps": step, "preempted": True
            }

    save_checkpoint(step)
    wandb.finish()

    return {
        "run_name": cfg["run_name"],
        "variant": variant,
        "steps": step,
        "first_loss": history[0]["loss_pred"],
        "last_loss": history[-1]["loss_pred"],
        "first_canary": history[0]["canary_cosine"],
        "last_canary": history[-1]["canary_cosine"],
        "out_dir": str(out_dir),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default=None, help="config name under training/configs/")
    parser.add_argument("overrides", nargs="*", help="key.sub=value overrides")
    args = parser.parse_args()
    cfg = load_config(args.variant, args.overrides)
    summary = train(cfg)
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
