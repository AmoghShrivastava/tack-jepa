# TacK-JEPA

**Tactile Kinematic Joint-Embedding Predictive Architecture** — a force-native,
kinematically-grounded, action-conditioned world model for articulated multi-taxel
tactile sensing. 100% simulation-sourced (Genesis); no physical hardware required.

- **Full design doc:** [PRD.md](PRD.md) — read this first; it is authoritative.
- **Build status & phase plan:** [ROADMAP.md](ROADMAP.md)
- **Literature grounding:** [docs/literature.md](docs/literature.md)

## Quickstart (Phase 0 — scaffold only)

```bash
python -m venv .venv            # Python 3.10–3.13
.venv/Scripts/activate          # Windows; use .venv/bin/activate on Linux/macOS
pip install -e .[dev]
pytest
ruff check .
```

Runtime dependencies (genesis-world, torch, torch_geometric, …) are added per phase —
see the decisions log in [ROADMAP.md](ROADMAP.md).

## Repository layout

See PRD §10. In brief: `sim/` (Genesis env, taxel synthesis, FK), `data/` (graph
construction, WebDataset sharding), `models/` (encoders, predictor, losses, ablations),
`training/`, `eval/`, `nebius/` (compute provisioning — **gated behind Phase 5**),
`tests/`, `docs/`.
