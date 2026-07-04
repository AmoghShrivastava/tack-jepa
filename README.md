# TacK-JEPA

**Tactile Kinematic Joint-Embedding Predictive Architecture** — a force-native,
kinematically-grounded, action-conditioned world model for articulated multi-taxel
tactile sensing. 100% simulation-sourced (Genesis); no physical hardware required.

- **Full design doc:** [PRD.md](PRD.md) — read this first; it is authoritative.
- **Build status & phase plan:** [ROADMAP.md](ROADMAP.md)
- **Literature grounding:** [docs/literature.md](docs/literature.md)

Status: Phases 0–4 complete on CPU only (no GPU billing yet). Phase 5 is a hard
stop-and-review gate before any Nebius provisioning — see ROADMAP.md.

## Quickstart

```bash
python -m venv .venv            # Python 3.10–3.13
.venv/Scripts/activate          # Windows; use .venv/bin/activate on Linux/macOS
pip install -e .[dev,sim,ml]
pip install torch --index-url https://download.pytorch.org/whl/cpu  # or a CUDA build
pytest
ruff check .
```

Or via Docker (PRD §8): `docker/sim.Dockerfile` (CPU, simulation/data-gen) and
`docker/train.Dockerfile` (GPU, training — reconcile the CUDA tag against the
actual Nebius VM before building; see the file's header comment).

## End-to-end pipeline

```bash
# 1. Generate episodes (CPU-bound, no GPU needed)
python -m sim.episode_generator --stage a --out datasets/stage_a --per-variant 35
python -m sim.episode_generator --stage b --out datasets/stage_b --per-variant 35

# 2. Shard into WebDataset tars (object-disjoint train/val split)
python -m data.shard_writer --episodes datasets/stage_a --out datasets/shards_a
python -m data.shard_writer --episodes datasets/stage_b --out datasets/shards_b

# 3. Train (any §7.2 ablation variant; CPU by default, see training/configs/)
python -m training.train --variant baseline train.steps=300
python -m training.train --variant no_fk,phase4_cpu train.steps=120

# 4. Evaluate a checkpoint
python -m eval.physics_probes_eval --run runs/baseline
python -m eval.collapse_canary --run runs/baseline
python -m eval.downstream_transfer --runs runs/baseline --episodes 8 16 32
```

Stage A = static press-only (fixed object, floating wrist approaches + holds,
constant finger pose — no motion during contact). Stage B = dynamic grasp
(free object, wrist settles into engagement pose while fingers actively
close — real action-conditioned dynamics + slip). See PRD §6.1 and
ROADMAP.md's "Audit and correction" section for why this distinction matters.

## Repository layout

See PRD §10. In brief: `sim/` (Genesis env, taxel synthesis, FK, episode
generators), `data/` (graph construction, WebDataset sharding), `models/`
(encoders, predictor, losses, §7.2 ablations), `training/` (training loop,
horizon curriculum, YAML configs), `eval/` (physics probes, downstream
transfer, collapse canary), `docker/` (CPU sim image, GPU train image),
`nebius/` (compute provisioning — **gated behind Phase 5**), `tests/`, `docs/`.
