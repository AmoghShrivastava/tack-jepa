# ROADMAP — TacK-JEPA

Living document tracking the phased build order from [PRD.md](PRD.md) §9.
**Hard rule: no Nebius GPU billing before Phase 5 is reached and explicitly approved by the user.**

| Phase | Deliverable | Exit criteria | GPU billing? | Status |
|---|---|---|---|---|
| 0 | Repo scaffold, `pyproject.toml`, CI skeleton, PRD committed, this file | `pytest` runs (even with zero tests), CI green | No | **Done** (2026-07-04; pytest 6 passing, CI run green) |
| 1 | Genesis environment: Allegro-class hand URDF loaded, single object, basic press episode runs headless | A single episode's raw contact-solver output can be dumped to disk and inspected | No (CPU only) | **Done** (2026-07-04; genesis 1.2.1 on Windows CPU, press episode with sustained ball-in-palm contact, dump inspected) |
| 2 | Taxel layout generation (FPS per link) + taxel force synthesis (§5.3) + FK module, with unit tests | Unit tests pass; taxel force heatmap for one episode looks physically sensible | No | **Done** (2026-07-04; FK matches Genesis to 1e-5, force conservation tested, heatmap in docs/figures/phase2_heatmap.png shows load under the object) |
| 3 | Graph construction + WebDataset sharding + Stage A data generation at small scale | A PyTorch `Dataset`/`DataLoader` yields correctly-shaped graph batches | No | **Done** (2026-07-04; 210 episodes → 140 train / 70 val object-disjoint shards; loader verified on real shards, ~1.1 s/batch B=4) |
| 4 | Full model: online + EMA target encoder, predictor, JEPA loss, VICReg, probe heads — tiny-scale training run to confirm the loop works | Loss decreases, no immediate collapse, all §7.2 ablation code paths runnable | Optional, minimal (confirm with user first) | **Done** (2026-07-04; all 5 variants trained on CPU — zero GPU billed; loss ↓ in every run; no point-collapse (per-dim std 0.27–0.31); probe/canary/downstream eval harnesses validated end-to-end; curves in docs/figures/phase4_training.png) |
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

- **2026-07-04 (Phase 3):** Stage A = 210 randomized press episodes (6 object variants:
  3 sphere sizes, 3 boxes; jittered drop pose, close magnitude). Val split holds out
  whole variants 2 & 4 (object-disjoint, PRD §7.3). Slip ground truth from contact
  tangential speeds; labels are rare in Stage A (~6e-5 taxel-level — static presses
  slip only during landing transients), so the slip probe uses pos_weight; Stage B
  dynamic grasps are the real slip data. Radius graphs built at load time (cheap for
  9-step windows); precompute into shards if the loader bottlenecks at cluster scale.
- **2026-07-04 (Phase 3):** webdataset on Windows: bare `C:\` paths and `file:///C:/`
  URLs both fail (different code paths); scheme-less relative forward-slash paths
  work everywhere (`data.shard_writer.local_wds_url`).

- **2026-07-04 (Phase 4):** CPU validation used a width-reduced overlay
  (`training/configs/phase4_cpu.yaml`: hidden 96, 3 layers — architecture otherwise
  identical) after measuring the full-size model at 60–100 s/step on laptop CPU;
  full-size training is a Phase 6 GPU matter. All five §7.2 variants trained without
  crashes; prediction loss decreased in every run. Collapse picture (honest): latents
  are NOT point-collapsed (per-dim std ≈ 0.27–0.31 on val) but share a large common
  component (pairwise cosine 0.87–0.93 for baseline); VICReg visibly injects variance
  (canary dips to ~0.55 when the hinge activates), and the no_vicreg run's canary sits
  at 0.9999999 — the anti-collapse machinery demonstrably matters. Toy-scale runs are
  loop validation ONLY; no H1/H2/H3 conclusions may be drawn from them (PRD §7.6).

- **2026-07-04 (Phase 4, eval-harness findings for Phase 6 planning):** (a) §7.3 probe
  eval and §7.4 downstream transfer both run end-to-end, but Stage A data contains
  almost no slip/slip-onset positives on the val split (3 positives / 432 windows) —
  meaningful slip metrics need Stage B dynamic-grasp data and slip-aware window
  sampling in the eval loaders. (b) Toy-encoder probe R² is negative (worse than
  predicting the mean) — expected at 300 CPU steps; these runs validate plumbing, not
  representations. (c) Full-size probe/downstream numbers must come from Phase 6 runs.

## Phase 8+ flagged items (per PRD)

- Soft-body/MPM taxel substrate coupling to replace the rigid-contact + Gaussian-kernel force distribution approximation (PRD §5.4).
- ART-Glove / OSMO real-data zero-shot validation, only if such a dataset is publicly released (PRD §7.5).
- Stage D longer-horizon manipulation sequences (PRD §6.1).
