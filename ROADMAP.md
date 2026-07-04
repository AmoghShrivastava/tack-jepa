# ROADMAP — TacK-JEPA

Living document tracking the phased build order from [PRD.md](PRD.md) §9.
**Hard rule: no Nebius GPU billing before Phase 5 is reached and explicitly approved by the user.**

| Phase | Deliverable | Exit criteria | GPU billing? | Status |
|---|---|---|---|---|
| 0 | Repo scaffold, `pyproject.toml`, CI skeleton, PRD committed, this file | `pytest` runs (even with zero tests), CI green | No | **Done** (2026-07-04; pytest 6 passing, CI run green) |
| 1 | Genesis environment: Allegro-class hand URDF loaded, single object, basic press episode runs headless | A single episode's raw contact-solver output can be dumped to disk and inspected | No (CPU only) | **Done** (2026-07-04; genesis 1.2.1 on Windows CPU, press episode with sustained ball-in-palm contact, dump inspected) |
| 2 | Taxel layout generation (FPS per link) + taxel force synthesis (§5.3) + FK module, with unit tests | Unit tests pass; taxel force heatmap for one episode looks physically sensible | No | **Done** (2026-07-04; FK matches Genesis to 1e-5, force conservation tested, heatmap in docs/figures/phase2_heatmap.png shows load under the object) |
| 3 | Graph construction + WebDataset sharding + Stage A data generation at small scale | A PyTorch `Dataset`/`DataLoader` yields correctly-shaped graph batches | No | Not started |
| 4 | Full model: online + EMA target encoder, predictor, JEPA loss, VICReg, probe heads — tiny-scale training run to confirm the loop works | Loss decreases, no immediate collapse, all §7.2 ablation code paths runnable | Optional, minimal (confirm with user first) | Not started |
| 5 | **Scale-up decision point** — review Phases 0–4 with the user before provisioning any Nebius training cluster | Explicit human go-ahead | **Yes, from here on** | Not started — hard gate |
| 6 | Stage B/C data generation at scale + full pretraining incl. all §7.2 ablations | Ablation results reported per §7.6 | Yes | Not started |
| 7 | Downstream task transfer evaluation (§7.4) | Sample-efficiency numbers reported | Yes | Not started |
| 8+ | Stretch: soft-body coupling (§5.4), real-data zero-shot validation (§7.5), Stage D | — | Yes | Not started |

## Decisions log

- **2026-07-04 (Phase 1):** Allegro URDF from **dexsuite/dex-urdf** (MIT; underlying
  SimLab BSD) rather than raw `allegro_hand_ros` — simulation-optimized collision
  primitives and clean visual meshes. Loaded with `recompute_inertia=True` (upstream
  inertia tensors violate A+B>=C) and `links_to_keep=` the 4 fingertip links (Genesis
  otherwise merges fixed links, breaking per-link contact attribution).
- **2026-07-04 (Phase 1):** Genesis 1.2.1 does not declare torch as a dependency;
  CPU torch installed separately. `hand.get_contacts()` also returns `normal` and
  `penetration` — the contact normal gives exact slip ground truth for §5.9 labels.
- **2026-07-04 (Phase 2):** Taxels sampled on links' *visual* meshes (actual surface),
  not collision primitives. Committed artifact: 2,244 taxels / 21 links (96–160 per
  link, 4–12 mm spacing). Standalone numpy FK cross-validated against Genesis link
  poses at 1e-5 tolerance.

- **2026-07-04 (Phase 0):** Config management = **plain YAML + argparse** (PRD §8 left this open). Rationale: the ablation suite is 5 enumerable variants, not a large sweep space; simple per-ablation YAML files overriding a base config are fully debuggable with no framework indirection. Revisit if sweep needs grow.
- **2026-07-04 (Phase 0):** Package management = **pip + venv, Python 3.10** (user's choice). Genesis (`genesis-world`) supports Python 3.10–3.13 as of its April 2026 PyPI release; 3.10.10 is what's installed locally. PRD's "3.11" was a placeholder pending this verification.
- **2026-07-04 (Phase 0):** Runtime dependencies are added and pinned in `pyproject.toml` at the start of the phase that first needs them, not all up front — keeps each phase's environment change small and independently verifiable.

## Phase 8+ flagged items (per PRD)

- Soft-body/MPM taxel substrate coupling to replace the rigid-contact + Gaussian-kernel force distribution approximation (PRD §5.4).
- ART-Glove / OSMO real-data zero-shot validation, only if such a dataset is publicly released (PRD §7.5).
- Stage D longer-horizon manipulation sequences (PRD §6.1).
