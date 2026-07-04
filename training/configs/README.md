# Training configs

Plain YAML + argparse (decided Phase 0 — see ROADMAP.md decisions log; PRD §8 left
this open). Convention, effective from Phase 4:

- `base.yaml` — full default config (the §5 baseline model + §6.2 hyperparameters).
- One YAML per curriculum stage (`stage_a.yaml`, …) and per §7.2 ablation
  (`no_fk.yaml`, `image_native.yaml`, `reconstruction.yaml`, `no_vicreg.yaml`),
  each containing ONLY the keys it overrides relative to `base.yaml`.
- A tiny loader in `training/` deep-merges base ← stage/ablation ← CLI overrides,
  and dumps the fully-resolved config alongside every run's outputs for
  reproducibility (PRD §7.6.3).
