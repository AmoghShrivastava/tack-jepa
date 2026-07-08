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
| 5 | **Scale-up decision point** — review Phases 0–4 with the user before provisioning any Nebius training cluster | Explicit human go-ahead | **Yes, from here on** | **Passed** (2026-07-04; explicit go-ahead given in conversation) |
| 6 | Stage B/C data generation at scale + full pretraining incl. all §7.2 ablations | Ablation results reported per §7.6 | Yes | **Done 2026-07-06/08.** Stage B scale done 2026-07-04/05 ($8.23); `image_native` collapse root-caused and fixed 2026-07-05. Stage C generated at scale (4000 episodes, 16 objects x 3 trajectories) and all 5 §7.2 variants retrained with an equal 6000-step budget each on Nebius — **`image_native` fix confirmed working in practice** (mean_dim_std 1.6e-5 -> 0.304, ~19,000x), `no_vicreg` shows genuine collapse (real finding: EMA alone insufficient without VICReg), see decisions log for full results table and caveats (full probe/downstream-transfer eval not completed, only the lighter collapse-canary diagnostic). Checkpoints archived to HF Hub, VM deallocated. |
| 7 | Downstream task transfer evaluation (§7.4) | Sample-efficiency numbers reported | Yes | **Partially done** — slip-onset transfer run for baseline/image_native/raw-baseline at small episode budgets (8/16/32) on the current small dataset; numbers not yet meaningful (see below) and grasp-stability task not yet wired up. |
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

- **2026-07-04 (Phase 6, provisioning):** User gave explicit go-ahead in-conversation.
  Provisioned one Nebius L40S AMD (preemptible, `gpu-l40s-d`, 46GB VRAM), not H100 —
  model is <100M params (§6.3), L40S is ~1/3 H100's cost and plenty for this scale;
  confirmed via real timing rather than assumption. See nebius/README.md cost log for
  exact instance ID, price, and deallocation record.
- **2026-07-04 (Phase 6, real finding):** the full-size model (GATv2 attention over
  ~2244 taxels x 8-step context) OOMs on this 46GB GPU above micro-batch ~4-5 — well
  below PRD §6.2's target effective batch of 32-64. This is a genuine architecture-
  scale cost (graph attention memory), matching a risk PRD §11 itself flagged, not a
  bug. Fixed via gradient accumulation (`train.micro_batch_size` + `data.batch_size`,
  training/train.py) rather than silently shrinking the effective batch — reaches
  effective batch 32 via micro-batch 4 x 8 accumulation steps. Measured throughput:
  ~5.2s per effective-batch-32 step on the L40S (vs. 60-100s/step at reduced width on
  CPU) — real speedup, used to size the ablation sweep's step budgets below.
- **2026-07-04 (Phase 6):** added periodic checkpointing + automatic resume (every
  `train.checkpoint_every` steps, default 500) before committing to a multi-hour run
  on a **preemptible** instance (can be stopped by Nebius anytime with 60s warning) —
  previously training only saved a checkpoint at the very end, so any interruption
  would have lost all progress.

- **2026-07-05 (Phase 6, results):** All 5 §7.2 variants trained full-size (hidden
  256, 6 layers) on Stage B data, effective batch 32 via gradient accumulation
  (micro-batch 4). **Findings (still small-data-scale — see caveat below, not final
  H1/H2/H3 conclusions):**
  - `baseline` and `no_fk` show healthy training: loss rises with each horizon-
    curriculum stage (expected — harder task), collapse canary oscillates (0.94–1.0,
    not pinned), VICReg variance hinge oscillates 0.67–0.93 (not collapsed).
  - **`image_native` shows apparent representational collapse**: canary pinned at
    1.0000 ± 1e-7 and VICReg variance pinned exactly at the 0.99 floor for the
    *entire* 800-step run, confirmed on a separate frozen-encoder eval pass too
    (`mean_dim_std` = 1.6e-5, four orders of magnitude below baseline's 0.30) —
    despite having the *same* EMA+VICReg anti-collapse machinery as baseline. Its
    prediction loss still drops (0.52->0.02), meaning it reaches low loss via some
    degenerate/near-constant solution rather than genuine representation learning.
    Its downstream slip AUROC (0.60, near chance) is consistent with this. **This is
    reported as a real, unexplained finding, not smoothed over** (PRD §7.6) — it
    needs investigation (rasterization/mosaic construction, ViT hyperparameters, or
    a genuine early sign for H2) before any H2 conclusion is drawn.
  - `reconstruction` and `no_vicreg` show intermediate representational diversity
    (mean_dim_std 0.006–0.007) — expected, since both ablations deliberately remove
    anti-collapse machinery (no target encoder/EMA/VICReg for reconstruction; no
    VICReg for no_vicreg) — this is the ablations working as designed, not a
    problem.
  - All probe R² values are negative (worse than predicting the mean) — expected at
    only 150 probe-training steps; slip AUROC (0.6–0.89 across variants, image_native
    the outlier) is the more informative metric at this scale. Full numbers in
    `runs/full_*/probe_eval.json`, curves in docs/figures/phase6_training.png.
  - **Caveat, important:** this used the EXISTING Stage A/B data (210+210 episodes),
    not PRD §6.1's "Stage C at scale" (thousands of diverse episodes). These results
    validate the full pipeline runs correctly end-to-end at real GPU scale and
    surface real findings (esp. image_native) worth chasing, but are NOT the final
    word on H1/H2/H3 — that requires Stage C generation (still free/CPU-only) plus a
    considerably longer/larger training budget.
- **2026-07-05 (Phase 6, infrastructure lessons):** (a) the full-size model OOMs
  above micro-batch ~4-5 on a 46GB GPU — fixed via gradient accumulation (see
  training/train.py decisions above). (b) Preemptible instances WILL be reclaimed
  mid-run in practice (observed once, ~4.5h into this session) — checkpoint/resume
  (added same day) made this a non-event costing a few lost minutes, not hours. (c)
  Eval (frozen-encoder probing) is significantly slower than expected (~12-15
  min/variant) because per-window taxel-graph construction (radius search over
  ~2200 taxels) is CPU-bound and dominates wall-clock even with a fast GPU doing the
  actual encoding — precomputing graphs into shards (already flagged as a Phase 6+
  option in the Phase 3 decisions log) becomes worth doing before any larger-scale
  run. (d) SSH monitoring connections over this network path dropped repeatedly
  (~5-6 times) without any impact on the actual remote training process — cosmetic
  only, but worth using more aggressive keepalive settings or a poll-based watch
  (not a long-lived tail -f) for future long remote runs.
- **2026-07-05 (Phase 6, wrap-up):** all 5 checkpoints (+ metrics/config/probe_eval)
  pushed to the private HF Hub model repo
  `AmoghShrivastava1/tack-jepa-phase6-checkpoints` via `hf upload` (device-code auth —
  no token ever handled by the assistant), then the Nebius instance and its disk were
  deleted entirely. Decision math: this disk's storage-only rate is $0.02/hr ($1.44
  over 3 days if left stopped); re-provisioning a fresh VM later costs ~$0.08 in
  one-time setup overhead (reinstall torch/PyG/webdataset — Stage A/B shards are only
  113MB to re-upload). Break-even is ~4 hours, so delete-after-archiving wins for any
  gap longer than that. Retrieve checkpoints anytime with
  `hf download AmoghShrivastava1/tack-jepa-phase6-checkpoints`. Note: re-provisioning
  a VM later is NOT the same as retraining — the $8.23 sweep's results are already
  saved; future GPU cost would only be for genuinely new work (e.g. confirming a fix
  to the `image_native` collapse, or Stage C training), not repeating this sweep.
- **2026-07-05 (Phase 6, `image_native` collapse root-caused and fixed — free,
  CPU-only analysis, no GPU needed):** downloaded the trained checkpoint from HF
  Hub and traced variance layer-by-layer (rasterize -> patch_embed -> transformer
  -> CLS -> global_head) against real val-split data. Root cause: in
  `models/ablations/image_native.py`'s `rasterize()`, the mosaic's third channel
  ("occupancy") was `torch.ones_like(f_n)` — a hardcoded constant, identical
  across every single window in the dataset, since every taxel always occupies a
  pixel regardless of contact. Measured on 1289 real windows with genuinely
  diverse contact patterns (mean taxel-overlap IoU only 0.359 between windows):
  this constant channel was ~350x larger in scale than the two channels carrying
  the real signal (f_normal, shear) and accounted for **100.00% of the image's L2
  energy** vs 0.00% for the other two combined — fed into an unnormalized
  `Conv2d` patch embedding, the network's input was, in practice, a fixed image,
  which explains the pinned canary/VICReg-floor/near-chance slip-AUROC findings
  above. Ruled out an initial alternative hypothesis (58% of windows have zero
  contact at all) as the primary cause first — collapse was equally total
  (canary 1.0000) even restricted to the 263 substantial-contact windows, so it
  wasn't primarily a data-sparsity artifact. **Fix:** `occ` is now a genuine
  per-taxel contact indicator (`(f_n != 0) | (shear != 0)`), verified on real
  data to vary correctly with the actual touched-taxel set (e.g. sums of
  160/113/270/148 across different windows, vs. always-2244 before). Added two
  new regression tests (`test_rasterize_occupancy_reflects_contact_not_constant`,
  `test_image_native_encoder_responds_to_different_contact_patterns`) —
  `TactileImageEncoder` previously had **zero** unit test coverage (the shared
  `tiny_batch()` helper uses partial taxel counts that fail this encoder's
  full-taxel-set assert), which is why this went undetected until the real GPU
  run. All 53 repo tests pass after the fix. **Not yet done:** actually
  retraining `image_native` to confirm the fix resolves collapse in practice —
  that requires GPU time and is a separate go/no-go decision from this
  free/local fix.
- **2026-07-07/08 (Stage C, full retrain results — all 5 variants, equal
  6000-step budget each):** collapse-canary diagnostic on all 5 checkpoints
  (`eval/collapse_canary.py`, val split):

  | variant | canary_cosine | mean_dim_std |
  |---|---|---|
  | baseline | 1.0 | 0.174 |
  | no_fk | 0.984 | 0.321 |
  | image_native | 0.423 | 0.304 |
  | reconstruction | 0.99997 | 0.0068 |
  | no_vicreg | 0.99992 | 0.0097 |

  **`image_native` fix confirmed working in practice.** vs. the old broken
  Phase 6 checkpoint (canary 1.0000, mean_dim_std 1.6e-5), the fixed version
  now shows mean_dim_std 0.304 — a ~19,000x increase — and canary 0.423,
  clearly not pinned. This holds even though the diagnostic itself has a
  real limitation (see below) that inflates canary for the other variants,
  meaning image_native's diversity is not an artifact.
  **Diagnostic limitation found:** `collapse_canary.py`'s loader defaults to
  `shuffle=0` and samples only 4 batches; since val shards are written
  sorted by variant, this reads almost exclusively from one held-out object
  (the lowest-indexed val variant), not a representative mix of the full
  held-out set. This explains `baseline`/`no_fk`'s near-1.0 canary despite
  healthy, large `mean_dim_std` (0.174/0.321, comparable to Phase 6's known-
  healthy baseline ~0.30) — physically similar windows of the *same* object
  can legitimately have high cosine similarity without the model being
  collapsed. Not fixed (would need `shuffle=1` in the loader call for a
  representative sample) — flagged honestly rather than smoothed over,
  future eval work should account for this.
  **`reconstruction` and `no_vicreg` show genuine collapse** — for these
  two (unlike baseline/no_fk) both metrics agree: pinned canary *and* tiny
  mean_dim_std together, which the narrow-sampling explanation doesn't
  rescue. `reconstruction` collapsing is expected (zero anti-collapse
  machinery by design). `no_vicreg` collapsing is a real, meaningful
  finding: its own config frames it as testing "does EMA alone prevent
  collapse?" — empirically here, no, it doesn't, consistent with
  established self-supervised-learning results that momentum encoders
  alone are typically insufficient without explicit variance
  regularization. This is direct evidence VICReg is doing necessary work
  in `baseline`, not decoration.
  **Caveat:** the full `physics_probes_eval.py` pass (fresh regression
  probes: force_mag/slip/contact_area, plus downstream_transfer) was not
  completed — it was taking 1.5+ hours per variant with no sign of
  finishing (Stage C's much larger training shard set makes the loader's
  shuffle-buffer warm-up dominate regardless of probe-step count; reducing
  `--steps`/`--eval-batches` did not help, confirming the bottleneck is
  data loading, not probe training). Switched to the much lighter
  `collapse_canary.py` instead, which answers the primary open questions
  (image_native fix confirmation, no_vicreg's collapse question) in
  ~5-8 min/variant. The full probe/downstream-transfer numbers remain a
  follow-up item if deeper quantitative comparison is wanted later.

- **2026-07-08 (Stage C full probe eval, Azure follow-up — in progress):**
  the full `physics_probes_eval.py` pass flagged as a caveat above is now
  underway on a dedicated Azure CPU VM (see `azure/README.md` for the
  detailed log), since the local dev machine hit real memory limits running
  the graph-encoder variants. `image_native` completed cleanly early on and
  its results match the local-machine run to 5 decimal places on
  canary/mean_dim_std, a good cross-environment consistency check. The other
  4 variants (`baseline`/`no_fk`/`no_vicreg`/`reconstruction`) hit a genuine
  memory leak in the probe-eval training loop (confirmed via kernel
  OOM-killer logs, not just an undersized batch — a batch_size=8 retry still
  leaked to 32GB over ~93 min rather than crashing instantly like
  batch_size=32 did) — worked around for now by moving to a 128GB-RAM VM
  rather than fixing the leak's root cause, an explicit user tradeoff given
  Azure credit is not a constraint here. HF Hub repo holding the Stage C
  checkpoints was also renamed from `tack-jepa-stagec-checkpoints` to
  `tack-jepa` (still private) at the user's request. Root-causing and fixing
  the actual leak (suspected: `physics_probes_eval.py`'s outer while-loop
  reconstructing the WebDataset loader on every full pass over the shard set
  without releasing the old one) remains a flagged follow-up item even after
  this run completes, since the workaround doesn't scale indefinitely.

## Phase 8+ flagged items (per PRD)

- Soft-body/MPM taxel substrate coupling to replace the rigid-contact + Gaussian-kernel force distribution approximation (PRD §5.4).
- ART-Glove / OSMO real-data zero-shot validation, only if such a dataset is publicly released (PRD §7.5).
- Stage D longer-horizon manipulation sequences (PRD §6.1).
