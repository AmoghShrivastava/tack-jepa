#!/usr/bin/env bash
# TacK-JEPA training launch template — Phase 6, GATED (see nebius/README.md).
# DO NOT RUN before the Phase 5 go-ahead. Nothing in this file provisions
# hardware; it documents the intended procedure on an already-provisioned VM.
set -euo pipefail

# ---------------------------------------------------------------- environment
# On the GPU VM (once): clone repo, then
#   python -m venv .venv && source .venv/bin/activate
#   pip install -e .[dev,sim]
#   pip install torch --index-url https://download.pytorch.org/whl/cu126  # match VM CUDA
#   pip install torch_geometric==2.8.0 webdataset==1.0.2 wandb
# Data: generate on a CPU instance (sim is CPU-bound, do not burn GPU time):
#   python -m sim.episode_generator --out datasets/stage_a --per-variant 35
#   python -m data.shard_writer --episodes datasets/stage_a --out datasets/shards
# then sync shards to the GPU VM (or object storage).

export WANDB_MODE=${WANDB_MODE:-online}   # requires WANDB_API_KEY

STEPS=${STEPS:-20000}

# ------------------------------------------------------- single-GPU (first!)
# Keep a single-GPU run working before any multi-GPU attempt (PRD §6.3).
run_single() {
  local variant=$1
  python -m training.train --variant "$variant" \
    run_name="${variant}_s0" seed=0 \
    train.device=cuda train.steps="$STEPS" data.batch_size=32
}

# --------------------------------------------------- full §7.2 ablation suite
# Every ablation is required — they ARE the research contribution (PRD §7.2).
all_variants=(baseline no_fk image_native reconstruction no_vicreg)

for v in "${all_variants[@]}"; do
  run_single "$v"
done

# Post-run evals (frozen-encoder probes + canary, per run):
for v in "${all_variants[@]}"; do
  python -m eval.physics_probes_eval --run "runs/${v}_s0" --steps 500
done
python -m eval.downstream_transfer --runs runs/baseline_s0 runs/image_native_s0 \
    --episodes 8 16 32 64

# ------------------------------------------------------------------ REMINDER
# STOP THE INSTANCE when done (fill in the exact CLI command at provisioning
# time and log it in nebius/README.md next to the launch command):
#   nebius compute instance stop <INSTANCE_ID>   # verify exact syntax live
echo "Training suite done — NOW STOP THE GPU INSTANCE (see nebius/README.md)."
