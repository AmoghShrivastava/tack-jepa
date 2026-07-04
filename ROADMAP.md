# ROADMAP — TacK-JEPA

Living document tracking the phased build order from [PRD.md](PRD.md) §9.
**Hard rule: no Nebius GPU billing before Phase 5 is reached and explicitly approved by the user.**

| Phase | Deliverable | Exit criteria | GPU billing? | Status |
|---|---|---|---|---|
| 0 | Repo scaffold, `pyproject.toml`, CI skeleton, PRD committed, this file | `pytest` runs (even with zero tests), CI green | No | **Done** (2026-07-04; pytest 6 passing, CI run green) |
| 1 | Genesis environment: Allegro-class hand URDF loaded, single object, basic press episode runs headless | A single episode's raw contact-solver output can be dumped to disk and inspected | No (CPU only) | **Done** (2026-07-04; genesis 1.2.1 on Windows CPU, press episode with sustained ball-in-palm contact, dump inspected) |
| 2 | Taxel layout generation (FPS per link) + taxel force synthesis (§5.3) + FK module, with unit tests | Unit tests pass; taxel force heatmap for one episode looks physically sensible | No | **Done** (2026-07-04; FK matches Genesis to 1e-5, force conservation tested, heatmap in docs/figures/phase2_heatmap.png shows load under the object) |
| 3 | Graph construction + WebDataset sharding + Stage A/B data generation at small scale | A PyTorch `Dataset`/`DataLoader` yields correctly-shaped graph batches | No | **Done, corrected 2026-07-04** — see "Audit and correction" below. 210 Stage A + 210 Stage B episodes → 140/70 object-disjoint shards each; loader verified on real shards. |
| 4 | Full model: online + EMA target encoder, predictor, JEPA loss, VICReg, probe heads — tiny-scale training run to confirm the loop works | Loss decreases, no immediate collapse, all §7.2 ablation code paths runnable | Optional, minimal (confirm with user first) | **Done, rerun on corrected pipeline 2026-07-04** — all 5 variants trained on CPU on Stage B data with the floating wrist, horizon curriculum, and bf16-capable loop; zero GPU billed. See "Audit and correction" below. |
| 5 | **Scale-up decision point** — review Phases 0–4 with the user before provisioning any Nebius training cluster | Explicit human go-ahead | **Yes, from here on** | Not started — hard gate |
| 6 | Stage B/C data generation at scale + full pretraining incl. all §7.2 ablations | Ablation results reported per §7.6 | Yes | Not started |
| 7 | Downstream task transfer evaluation (§7.4) | Sample-efficiency numbers reported | Yes | Not started |
| 8+ | Stretch: soft-body coupling (§5.4), real-data zero-shot validation (§7.5), Stage D | — | Yes | Not started |

## Audit and correction (2026-07-04)

A full re-audit of Phases 0–4 against PRD.md (re-reading §5/§6/§8/§10 line by
line against the actual code, not against prior summaries) found several
real discrepancies. Per the user's direction, all were fixed rather than
left as accepted deviations. Recorded here for the historical record —
**earlier entries in this log that describe "static presses" or a fixed hand
base are describing the pre-correction implementation and are superseded.**

**Not better alternatives — fixed:**

1. **Fixed hand base + falling object, instead of PRD §5.2's floating 6-DoF
   wrist + stationary object.** The original `HandEnv` defaulted
   `hand_fixed=True` with no code path ever setting it otherwise, so the 6
   "wrist" dims of the 22-dim action vector were hardcoded constants in
   every episode — only 16 of 22 action dims carried information, and the
   "22-DoF, ART-Glove-matching action space" claim was only nominally true.
   **Fix:** `HandEnv` now defaults to a genuine free-floating wrist (Genesis
   `FREE` joint, verified empirically: dofs `[0:3]`=world position,
   `[3:6]`=orientation as a rotation vector — matches this codebase's
   existing `PALM_UP_ROTVEC` convention exactly). PD gains for the wrist
   dofs default to zero in Genesis and had to be set explicitly (tuned:
   kp=1500, kv=100 — converges in ~50 steps at dt=0.01, no oscillation,
   ~5–10mm steady-state gravity sag). `sim/hand_env.py` also adds
   `genesis_to_prd_order`/`prd_to_genesis_order` to convert between
   Genesis's native dof layout (`[wrist_pos3, wrist_rot3, finger16]`) and
   the PRD §5.2 action/state convention (`[finger16, wrist_pos3,
   wrist_rot3]`) used in shards/model inputs. Verified: all 6 wrist
   dimensions now show real per-episode variance in generated data (was
   exactly zero before).

2. **What was generated and labeled "Stage A" was actually Stage B's
   dynamics.** §6.1 assigns "closing fingers" as Stage B's defining new
   feature over Stage A ("no wrist/finger motion during contact, just
   approach-contact-hold"); the original generator actively interpolated
   fingers open→close for 120 of 220 steps while already in contact — true
   static Stage A was never built, and this log's Phase-3 entry below
   incorrectly called that data "static presses."
   **Fix:** `sim/episode_generator.py` now implements both stages properly:
   - **Stage A** (`rollout_stage_a_episode`): object is `fixed=True`
     (truly stationary, no gravity/dynamics on it at all); the floating
     wrist executes a randomized approach trajectory (jittered start pose,
     tighter-jittered final/held pose) and holds; **finger pose is constant
     for the entire episode** (a per-episode randomized curl in [0.85, 1.2]
     of the reference close amount, never interpolated mid-episode) — zero
     finger motion, matching §6.1 literally. Tuned object placement
     (nominal height 0.308m under a wrist held at 0.25m) and jitter bands
     empirically so ~70% of episodes register real contact (some
     zero-contact episodes are kept deliberately — realistic negative
     examples, not every real approach should succeed).
   - **Stage B** (`rollout_stage_b_episode`): object is free/dynamic (drops
     under gravity into the hand, reusing the Phase 1–3-validated
     engagement geometry); the wrist **also** moves (randomized start offset
     settling into the nominal engagement pose over the settle+close
     phases) **while** fingers interpolate open→close — genuine
     multi-timestep, action-conditioned dynamics with "varying grasp
     poses," matching §6.1. 210/210 episodes register contact; 197/210
     show slip events >0.01 m/s (vs. Stage A's near-zero slip rate observed
     in Phase 4 — Stage B is now the real source of slip training signal,
     as intended).
   - Full arbitrary-direction approach (grasping from the side/below) is
     explicitly **not** attempted — that diversity is Stage C's stated scope
     per §6.1 ("diverse grasp/slide trajectories at scale"), not A/B's.
   - The perturbation/grasp-stability capability (§7.4) is preserved,
     rehomed onto Stage B episodes (`--perturb` flag) since it needs a
     dynamic object.
   - `data/shard_writer.py`'s `process_episode` had a related bug: it
     sliced `dump["qpos"][:, :16]` assuming the OLD fixed-base dof layout
     (dofs 0–15 = fingers); under the floating-base layout that silently
     grabbed `[wrist_pos3, wrist_rot3, finger10]` instead. Fixed to use the
     episode generator's own `qpos22`/`action22` fields directly (already
     correctly reordered via `genesis_to_prd_order`).

**Deferred items that existed but were never explicitly logged as open —
now implemented rather than silently left absent:**

3. **Horizon curriculum** (§5.7/§6.2: k=1→8 "as training stabilizes") was
   entirely unimplemented — training ran at one fixed horizon throughout.
   **Fix:** `training/curriculum.py` splits total steps evenly across
   stages (1, 2, 4, 8, capped at `predictor.max_horizon`); `training/train.py`
   builds one DataLoader per stage horizon and switches between them by
   step. Verified: an 16-step run showed horizon 1→2→4→8 exactly on the
   4/8/12/16-step schedule boundaries.

4. **bf16 mixed precision** (§6.2/§8: "Nebius H100-class GPUs support this
   natively") was entirely unimplemented. **Fix:** `training/train.py` adds
   an `amp_context()` helper (`torch.autocast`, no GradScaler needed — bf16
   doesn't require fp16-style loss scaling) gated by a new `train.precision`
   config key (`fp32` default; set `bf16` at Nebius launch time). **Verified
   forward pass only** — this machine's CPU (oneDNN/avx2_vnni_2 backend)
   does not support bf16 **backward** at all ("DNNL does not support
   bf16/f16 backward on the platform with avx2_vnni_2" — a genuine hardware
   limitation of this laptop, not a code bug); full forward+backward
   validation is deferred to actual Nebius GPU hardware where bf16 backward
   is natively supported.

5. **`base.yaml` batch_size** was 8 vs. PRD §6.2's stated "32–64 sequences."
   **Fix:** bumped to 32.

6. **Dockerfiles** (§8: CPU image for sim/data-gen, separate GPU image for
   training) didn't exist and the absence was never noted. **Fix:**
   `docker/sim.Dockerfile` (CPU, genesis-world + sim deps) and
   `docker/train.Dockerfile` (GPU, CUDA base + torch + PyG + wandb) added.
   **Not build-tested** — Docker isn't installed on this machine; the GPU
   image's CUDA tag/torch index must be reconciled against the actual
   Nebius VM at provisioning time regardless (per §11's "don't hardcode"
   guidance), so this is deferred rather than a gap to silently carry.

**Considered and kept as-is (a defensible design call, not a bug):** the
`image_native` ablation rasterizes force/shear/occupancy directly as image
channels rather than a depth/geometric rendering. PRD's own phrasing allows
a "pseudo-tactile image" as an alternative to strict depth, and a true
optical renderer would contradict the project's explicit no-optical-
rendering design (§1.3) for a comparison whose core structural point (fixed
2D per-link pixel grid vs. exact-FK 3D graph, with resolution loss) survives
either way.

**Also fixed in the same pass:** the "timestep-relative positional encoding"
listed in §5.5 as a per-taxel node feature is implemented instead as a
per-graph positional encoding at the predictor/sequence level — this matches
§5.1's own architecture diagram (encoder runs once per timestep; predictor
owns the temporal/sequence dimension) more coherently than duplicating a
time signal into every taxel's per-step features would, and is kept as-is.

All fixes re-verified: 50/50 tests pass (up from 46 — added FK-vs-Genesis
under a genuinely moving wrist, dof-order-conversion tests, and a rotvec↔quat
convention test), all 5 §7.2 variants retrained successfully on the
corrected Stage B data, CI green.

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

- **2026-07-04 (Phase 3, SUPERSEDED — see "Audit and correction" above):** ~~Stage A =
  210 randomized press episodes... static presses slip only during landing
  transients...~~ This entry described data that was actually Stage-B-like dynamics
  (active finger closing during contact) mislabeled as Stage A, on a fixed hand base.
  Both issues are now fixed; see the correction section. What remains true: object
  variants (3 spheres, 3 boxes), object-disjoint val split (variants 2 & 4), slip
  ground truth from contact tangential speeds, and slip pos_weight on the probe.
  Radius graphs still built at load time (cheap for 9-step windows); precompute into
  shards if the loader bottlenecks at cluster scale.
- **2026-07-04 (Phase 3):** webdataset on Windows: bare `C:\` paths and `file:///C:/`
  URLs both fail (different code paths); scheme-less relative forward-slash paths
  work everywhere (`data.shard_writer.local_wds_url`).

- **2026-07-04 (Phase 4, first pass — SUPERSEDED, rerun on corrected pipeline below):**
  CPU validation used a width-reduced overlay (`training/configs/phase4_cpu.yaml`:
  hidden 96, 3 layers — architecture otherwise identical) after measuring the
  full-size model at 60–100 s/step on laptop CPU; full-size training is a Phase 6 GPU
  matter. This measurement and the overlay itself remain valid and are reused for the
  rerun. The specific loss/canary numbers from this pass were on the mislabeled
  fixed-wrist "Stage A" data (see correction above) and are superseded by the rerun.
  §7.3/§7.4 eval harnesses ran end-to-end in this pass too; the finding that Stage A
  had almost no slip positives (3/432 val windows) directly motivated building a
  proper Stage B, which now shows 197/210 episodes with real slip events.

- **2026-07-04 (Phase 4, rerun on corrected pipeline — results):** All five §7.2
  variants retrained on Stage B shards (real floating wrist, horizon curriculum
  k=1→2→4→8, bf16-capable loop with precision=fp32 for this CPU pass) using the
  same `phase4_cpu` width overlay. All completed without crashes; prediction loss
  decreased in every run (baseline 0.48→0.06 over 300 steps; no_fk/image_native/
  no_vicreg/reconstruction each ↓ over 120 steps). Horizon curriculum verified
  switching 1→2→4→8 exactly on schedule. Collapse picture: baseline's canary
  oscillates (0.65–1.0) as VICReg's hinge activates/relaxes, consistent with the
  first pass; the 120-step ablation runs are too short to show this oscillation
  (it only becomes visible past ~step 130 in the baseline curve) — an
  apples-to-oranges step budget, not a discrepancy between variants. Curves:
  docs/figures/phase4_training.png (regenerated). §7.3 probe eval and §7.4
  downstream transfer re-run against the corrected Stage B checkpoint: slip-onset
  positives rose from 3/432 (old, mislabeled Stage A) to 13/432 val windows — still
  too few for a meaningful metric at this toy scale, but confirms Stage B is
  genuinely the richer slip signal Stage A was missing. Toy-scale runs remain loop
  validation ONLY; no H1/H2/H3 conclusions may be drawn from them (PRD §7.6) —
  full-size, full-data numbers are a Phase 6 matter.

## Phase 8+ flagged items (per PRD)

- Soft-body/MPM taxel substrate coupling to replace the rigid-contact + Gaussian-kernel force distribution approximation (PRD §5.4).
- ART-Glove / OSMO real-data zero-shot validation, only if such a dataset is publicly released (PRD §7.5).
- Stage D longer-horizon manipulation sequences (PRD §6.1).
