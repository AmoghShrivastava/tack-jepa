# PRD — TacK-JEPA
### Tactile Kinematic Joint-Embedding Predictive Architecture
**A force-native, kinematically-grounded, action-conditioned world model for articulated multi-taxel tactile sensing**

Version 0.1 — Draft PRD for greenfield repository
Status: Pre-training — architecture, data pipeline, and eval harness to be fully built and validated before any large-scale GPU training run is executed.

---

## 0. One-paragraph summary

Every existing tactile world model / foundation model (SPARSH, AnyTouch, T3, UniT, TacForeSight, Visuo-Tactile World Models, Tactile-WAM) represents touch as a 2D image and asks a vision-style encoder (ViT, ResNet) to implicitly infer 3D contact geometry, force, and dynamics from pixel appearance. This project takes a structurally different bet: build a **force-native** (not optical/image-native) tactile world model over an **articulated multi-taxel hand**, where the 3D position of every tactile element is **computed exactly via forward kinematics** from known joint encoder angles rather than learned from appearance. The model never has to guess where in space a signal came from — that's arithmetic, not inference — so 100% of its learned capacity goes toward modeling contact **dynamics** (how force evolves, propagates, predicts slip, predicts the next state given an action), trained with a JEPA-style latent-prediction objective. This is a research-novelty bet, not an immediate deployment product: no comparable architecture exists in the current literature (verified June/July 2026), and there is no large installed base of real dense-taxel articulated hands yet — this is a bet on where tactile hardware (ART-Glove, OSMO-class hands) is heading, and a defensible, falsifiable research contribution regardless of hardware timing.

---

## 1. Problem Statement

### 1.1 What's broken today

Tactile sensing research in 2026 has two separate, unconnected threads:

1. **Cross-sensor tactile foundation models** (SPARSH, AnyTouch/AnyTouch2, T3, UniT, UniTac-NV, FTP-1) — these unify *optical* tactile sensors (GelSight, DIGIT, DuraGel) into shared embedding spaces via masked modeling and cross-sensor contrastive matching. They are all **image-patch-based**: a tactile reading is patchified like a photograph, and the model has to implicitly re-derive 3D contact geometry, deformation, and force from pixel appearance — the same problem monocular depth estimation has, just applied to touch.

2. **Touch-conditioned world models** (TacForeSight, Visuo-Tactile World Models, Tactile-WAM) — these predict future tactile/visual state conditioned on action, improving policy learning on contact-rich tasks. They are also image/force-signal-based on **1-2 fingertip sensors** and require real, expensive, hardware-specific instrumentation (e.g., TacForeSight needs real wrist force/torque sensors on a dual-finger gripper) to get their conditioning signal.

Neither thread engages with where hardware is actually heading: **distributed, whole-hand, kinematically-tracked tactile sensing** (ART-Glove: 2048 taxels + 22 DoF joint tracking at 120Hz; OSMO: similar class of device). These devices give you something optical tactile sensors structurally cannot: **exact, known 3D geometry of every tactile element, for free, at every timestep**, because the sensor patches are rigid and their pose is fully determined by joint encoder readings. Nobody is building a world model that exploits this.

### 1.2 The core insight

If you know the joint angles of an articulated multi-taxel hand, you know the exact 3D position of every taxel via forward kinematics — no learning required. Every image-based tactile model spends model capacity learning geometry that, on this class of hardware, is already known exactly. Removing that burden should let the model spend all of its capacity on the actual hard part: modeling contact **dynamics** — how force distributes, evolves, predicts slip, and responds to action.

This project tests that hypothesis directly, with an architecture built to make the comparison possible (see §7, Evaluation Plan).

### 1.3 Why this is buildable with zero physical hardware

The builder has no robot arm, no tactile sensor, and no lab. This project is designed around that constraint as a feature, not a limitation:

- All data comes from the **Genesis** physics simulator (rigid-body dynamics + contact solver), which computes real contact forces natively — no optical rendering step is needed at all, unlike GelSight/DIGIT-style simulators (TACTO, DIFFTACTILE, FOTS), which must approximate an elastomer's optical deformation. Reading contact-solver output directly at known taxel locations is *more* physically exact than any optical simulation pipeline, not a compromise.
- Forward kinematics is exact arithmetic given a URDF and joint state — no real hardware needed to compute it.
- Evaluation is primarily **intrinsic** (physics-probing against simulation ground truth) and **ablation-based** (does removing the kinematic-grounding module hurt accuracy), which requires no real dataset at all. Real-data zero-shot validation (ART-Glove / OSMO datasets, if released) is a stretch goal, not a requirement — see §7.5.

---

## 2. Non-Goals (explicitly out of scope for v1)

- **Not** attempting cross-sensor generalization across optical tactile sensor types (GelSight/DIGIT/DuraGel). This project commits to one sensor modality (force-native taxel arrays) and one embodiment class (an articulated multi-taxel hand). Breadth is deliberately traded for depth and structural novelty — see prior architecture discussion for why.
- **Not** training on or requiring any real robot hardware, real tactile sensor, or human-collected data at any point. 100% simulation-sourced training data.
- **Not** attempting soft-body/FEM-level physical realism in v1 (see §5.4 — rigid-contact + force-distribution kernel is the v1 approach; soft-body/MPM coupling is an explicitly flagged v2 stretch item).
- **Not** deploying to a real robot. This is a research architecture + eval project. No ROS integration, no real-time control loop, no physical safety considerations.
- **Not** starting any paid GPU training run without explicit human go-ahead. Everything in this PRD through Phase 5 (see §9) must be built, unit-tested, and validated at tiny scale (CPU or single small GPU, minutes not hours) before any Nebius multi-GPU training job is launched.

---

## 3. Literature Grounding (as of July 2026)

| Work | What it does | Why this project is different |
|---|---|---|
| SPARSH (Meta, ~2024-25) | Frozen ViT-Base MAE, self-supervised on optical tactile images across sensor types | Image-patch based; no explicit geometry; no world-model/prediction objective |
| AnyTouch / AnyTouch 2 | Unified static+dynamic optical tactile representation, masked modeling + cross-sensor matching | Cross-sensor generalization goal (not ours); still image-patch based |
| T3 / UniT / UniTac-NV | Cross-sensor / cross-embodiment transferable tactile representations | Same image-patch limitation; morphology-aware tokens, not kinematic grounding |
| FTP-1 | Generalist tactile policy across sensors via Morphology-Aware Tactile Token Space | Closest prior art to "handle heterogeneous sensor geometry" but still operates on sensor *images*, and targets policy-generalist breadth, not force-native single-embodiment depth |
| TacForeSight | Force-guided tactile world model, JEPA-adjacent latent dynamics, conditioned on real wrist force/torque | Requires real instrumented dual-finger hardware; not kinematically grounded; only 1-2 sensing points, not a distributed multi-taxel field |
| Visuo-Tactile World Models (VT-WM) | Multi-task world model combining vision+touch; improves object permanence / physics compliance | Vision+image-tactile fusion, not touch-only, not kinematically grounded |
| Tactile-WAM | Touch-aware world-action model; identifies and fixes "tactile pollution" (naive tactile fusion degrades video/action prediction) | Important cautionary finding we must design around (see §6.5); still image-tactile, fused with video |
| ART-Glove (CMU) | Hardware: 2048-taxel + 22-DoF articulated glove for human demonstration capture | Data-capture device, not a model. This project is the "what would a world model built for this class of hardware look like" answer nobody has published |
| OSMO | Open-source tactile glove for human-to-robot skill transfer | Same category as ART-Glove; potential future real-data source (§7.5) |
| VibeAct, Sound of Touch | Vibration/acoustic tactile sensing | Different modality entirely; noted as a possible future extension, not part of v1 |
| DIFFTACTILE, TacEx, FOTS | Optical/differentiable tactile *simulators* | Not used in v1 — this project avoids optical simulation entirely by being force-native |

**The specific gap this project fills:** kinematically-grounded (exact FK geometry) + force-native (no optical rendering, physics-solver-exact) + JEPA-style (latent, action-conditioned, predictive) world model, for a distributed multi-taxel articulated hand. This combination does not appear in the literature as of this PRD's writing.

---

## 4. Hypotheses (falsifiable, this is a research project — say so explicitly)

- **H1 (Kinematic grounding):** Providing exact forward-kinematics-computed 3D taxel positions as input improves downstream physics-probe accuracy (force, slip, contact area) versus an ablated variant that only sees raw joint angles as an opaque feature vector (must learn geometry implicitly).
- **H2 (Force-native vs. image-native):** For the same underlying simulated contact events, a force-native taxel-graph encoder achieves higher physics-probe accuracy than an image-patch encoder fed an equivalent optically-rendered view of the same event.
- **H3 (Latent prediction vs. reconstruction):** A JEPA-style latent-space prediction objective produces representations that transfer better to downstream sim tasks (grasp stability, slip classification) than an equivalent model trained to reconstruct raw future taxel values.

All three are designed to be testable with ablations built into this same codebase (§7.2) — no external baseline or dataset is required to test any of them.

---

## 5. System Architecture

### 5.1 High-level pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│  GENESIS SIMULATION                                                   │
│  Articulated hand (Allegro-class URDF) + object (YCB/procedural)      │
│  Rigid-body contact solver → per-contact-point normal + friction force│
└───────────────────────────────┬────────────────────────────────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  TAXEL FORCE SYNTHESIS     │   deterministic, not learned
                    │  Distribute contact-solver │
                    │  forces onto fixed taxel   │
                    │  layout via kernel weighting│
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  FORWARD KINEMATICS        │   deterministic, not learned
                    │  joint state → exact 3D    │
                    │  world-frame taxel positions│
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  GRAPH CONSTRUCTION        │
                    │  nodes = taxels (pos, normal,│
                    │  force vector, link id)    │
                    │  edges = radius graph (world│
                    │  frame) ∪ static intra-link │
                    │  backbone                  │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  CONTEXT ENCODER (online)  │  ← TRAINED
                    │  Graph Attention Transformer│
                    └────────────┬─────────────┘
                                 │           ┌─────────────────────┐
                    ┌────────────▼─────────┐ │  TARGET ENCODER (EMA)│ ← EMA of online,
                    │  PREDICTOR             │ │  same architecture   │   stop-gradient
                    │  latent(t) + action(t) │◄┤  produces latent(t+k)│
                    │  → predicted latent(t+k)│ │  target             │
                    └────────────┬─────────┘ └─────────────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  LOSSES                    │
                    │  • JEPA latent loss         │
                    │  • VICReg anti-collapse reg │
                    │  • Physics probe losses      │
                    │    (force / slip / contact  │
                    │     area — auxiliary heads)  │
                    └────────────────────────────┘
```

### 5.2 Embodiment: simulated hand

- **Base kinematic structure:** Allegro Hand URDF (open-source, 16 DoF, 4 fingers, widely used in dexterous manipulation research — real hardware exists, so this is not a purely synthetic embodiment). Source: `wonikrobotics/allegro-hand` description packages (verify current canonical URDF source at build time — multiple open mirrors exist; confirm license permits redistribution in the new repo).
- **Wrist:** modeled as a free 6-DoF floating base in early phases (simplifies action space: object stays fixed, hand moves) — matches the "floating object" insight validated in prior sim work: fully reachable contact from any approach direction.
- **Taxel layout (synthetic, since Allegro's real hardware has no taxels):** for each rigid link's collision mesh, use farthest-point sampling (FPS) over the mesh surface to place a fixed number of taxel nodes in the link's local frame (recommend 96–160 taxels per link depending on link surface area; target ~2,000–2,500 taxels hand-wide to match ART-Glove's published scale of 2048). Taxel positions in local frame are fixed at repo-build time and version-controlled (a JSON/NPZ artifact, not regenerated per run) so the sensor layout is a fixed, reproducible spec, exactly like real hardware would be.
- **Total action DoF:** 16 (finger joints) + 6 (floating wrist pose delta) = 22, matching ART-Glove's 22-DoF spec (this is intentional — the FK/action interface is designed to be a plausible target for eventual ART-Glove real-data transfer, see §7.5).

### 5.3 Taxel force synthesis (the critical, most novel-adjacent implementation detail)

Genesis's rigid-body solver produces discrete contact points with normal and friction force vectors — not a continuous pressure field. To synthesize taxel readings:

1. At each simulation timestep, query all active contact points between the hand and the object.
2. For each contact point, transform its position into the local frame of the link it belongs to.
3. Distribute that contact point's force onto nearby taxels using a fixed Gaussian kernel over local-frame distance (kernel bandwidth ≈ 1.5× average taxel spacing on that link), normalized so total force is conserved.
4. Sum contributions across all active contact points per taxel per timestep.
5. Record both the normal component and the tangential (shear) component per taxel — shear is what lets the model learn slip, not just static contact.

This kernel-distribution step is **fixed and deterministic** (not learned) — it is the simulated equivalent of "how does a real piezoresistive taxel array's mechanical coupling spread a point force across nearby elements," a simplification that should be explicitly documented as a modeling assumption and a candidate for v2 refinement (see §5.4).

### 5.4 Explicitly flagged v2 stretch item: soft-body coupling

Genesis supports MPM/soft-body materials. A more physically faithful v2 could model the taxel substrate as a thin compliant layer via Genesis's soft-body coupling instead of the rigid-contact + kernel-distribution approximation in §5.3. This is **out of scope for v1** — do not attempt until the rigid-contact pipeline and full architecture are validated end-to-end. Flag this in the repo's `ROADMAP.md` as a explicit "Phase 8+" item.

### 5.5 Graph construction

- **Nodes:** one per taxel. Feature vector = [3D world-frame position (from FK), surface normal (from FK + link geometry), force vector (normal + 2D shear, from §5.3), link-id one-hot or learned embedding, timestep-relative positional encoding].
- **Edges:** union of (a) a radius graph recomputed every timestep over FK-updated world-frame taxel positions (radius ≈ 1 cm, capped at k=16 nearest neighbors to bound compute), and (b) a static intra-link backbone graph (fixed at build time, connects taxels within the same rigid link regardless of current pose) to guarantee message passing within a link even when radius-graph edges are sparse.
- **Why recompute in world frame every timestep, not per-link-local:** this is what makes cross-finger contact (e.g., two fingertips pinching together) visible to the graph — in world frame, taxels on different fingers become graph-adjacent exactly when they're physically close, which is the actual signal a real distributed tactile sensor would give you.

### 5.6 Encoder

- Architecture: Graph Attention Transformer (GATv2-style attention layers, e.g. `torch_geometric.nn.GATv2Conv`), 6 layers, hidden dim 256, 8 attention heads.
- Per-node outputs projected to a per-node latent (dim 256); a learned attention-pooling layer (or a `[CLS]`-style virtual global node connected to all taxel nodes) produces one global context vector per timestep (dim 512).
- Two encoder instances: **online encoder** (trained via gradient descent) and **target encoder** (EMA of online encoder's weights, no gradient, momentum schedule e.g. 0.996 → 0.9999 over training, standard I-JEPA/V-JEPA/DINO-style teacher-student setup) — this is what prevents representational collapse in latent-prediction objectives and is required, not optional.

### 5.7 Predictor

- Input: sequence of past online-encoder context vectors (causal, last N timesteps, N configurable e.g. 8) + action embeddings for timesteps t..t+k (small MLP encoding the 22-dim action vector into the same latent dim, injected via FiLM conditioning or concatenation + cross-attention — recommend cross-attention for flexibility).
- Architecture: small causal Transformer (4 layers, dim 512) predicting the target encoder's context vector k steps ahead (k configurable, start with k=1, curriculum to longer horizons — see §6).
- Loss: smooth-L1 (Huber) between predicted latent and **stop-gradient** target-encoder latent — classic JEPA loss, this is what makes it JEPA and not a generic seq2seq predictor.

### 5.8 Anti-collapse regularization

- Add a VICReg-style variance + covariance regularization term on the online encoder's outputs (standard practice in every JEPA implementation — I-JEPA, V-JEPA, and their derivatives all need this or an equivalent EMA-teacher setup to avoid representation collapse to a constant vector). Both the EMA teacher AND the VICReg term should be implemented — belt and suspenders, ablate later if one turns out to be redundant.

### 5.9 Physics probe heads (auxiliary, for evaluation, not part of core JEPA loss)

- Small MLP heads (2 layers) taking the online encoder's per-node or global latent as input, trained (optionally with encoder frozen, optionally jointly with a small loss weight) to predict:
  - Per-taxel force magnitude (regression, MSE) — ground truth from §5.3.
  - Binary slip indicator per taxel (classification, BCE) — ground truth: relative tangential velocity between contact point and object surface exceeding a threshold, computed directly from simulation state.
  - Global contact area (regression, MSE) — ground truth: count of taxels with force above a threshold.
  - (Stretch) object surface friction coefficient and stiffness class, if variable material properties are added to the object set in Phase 2+.

### 5.10 Addressing "tactile pollution" (Tactile-WAM's finding) up front

Tactile-WAM found naive fusion of tactile signal into a shared model degrades unrelated predictions. This project sidesteps that specific failure mode by **not fusing with a video/vision world model at all** — this is a touch-only world model. But the analogous risk here is the physics-probe auxiliary losses (§5.9) degrading the core JEPA representation if weighted too heavily during joint training. Mitigation: probes are trained primarily as **frozen-encoder linear/small-MLP probes** for evaluation (§7), and only optionally fine-tuned jointly at a low loss weight (≤0.1× the JEPA loss) as a separate ablated variant — never as the primary training signal.

---

## 6. Training Plan

### 6.1 Curriculum (build and validate stage-by-stage, do not skip ahead)

| Stage | Description | Purpose |
|---|---|---|
| A | Static single-object press-only episodes (no wrist/finger motion during contact, just approach-contact-hold) | Validate basic force-geometry association; smallest/simplest data to debug the full pipeline end-to-end |
| B | Single-object dynamic grasp sequences (multi-timestep, closing fingers, varying grasp poses) | Introduce action-conditioning and temporal dynamics |
| C | Multi-object (YCB + procedural superquadrics), diverse grasp/slide/press trajectories at scale | Main pretraining stage |
| D (stretch) | Longer-horizon, multi-step manipulation sequences (e.g. grasp + lift + hold) | Longer-horizon world-model prediction; only after C is validated |

### 6.2 Hyperparameters (starting point, expect to tune)

- Optimizer: AdamW, lr 3e-4 (encoder/predictor), cosine schedule with warmup.
- Batch size: start small (32–64 sequences) on a single GPU for debugging; scale with available Nebius GPU count.
- EMA momentum: 0.996 → 0.9999 linear schedule over training.
- Prediction horizon k: curriculum from k=1 up to k=8 timesteps as training stabilizes.
- Precision: bf16 mixed precision (Nebius H100-class GPUs support this natively).

### 6.3 Distributed training

- PyTorch DDP for multi-GPU single-node; consider FSDP only if model size grows beyond what fits comfortably in DDP (unlikely at the scale described here — this is a small model by foundation-model standards, likely <100M params total).
- Use `torchrun` for launch; keep a single-GPU code path fully working first (debugging on Nebius multi-GPU clusters is expensive and slow to iterate on).

### 6.4 Nebius compute plan

- **Do not provision GPU compute until Phase 5 (§9) is reached and explicitly approved.**
- Recommended instance class for initial validation: a single Nebius GPU VM (1× H100 or equivalent) — confirm current SKU names and pricing via Nebius console/API at the time of provisioning, as these change.
- Recommended scale-up path: once single-GPU training loop is confirmed correct (loss decreasing, no collapse, probes improving on a small held-out set), move to a multi-GPU single-node instance (e.g. 8×H100) for the full Stage C pretraining run.
- Data generation (Genesis simulation) is CPU-bound and should run on a separate, cheaper CPU-only instance (or the CPU cores of the same node before the GPU stage starts) — do not run simulation on GPU-billed time.
- Always deallocate/stop GPU instances immediately after each run; log the exact stop command in `nebius/README.md` in the new repo alongside the provisioning command, mirroring the discipline used in this builder's prior Azure project (a persistent "always add the deallocate command right next to the launch command" habit).

### 6.5 Experiment tracking

- Weights & Biases for all training runs (loss curves, probe accuracy over training, EMA momentum schedule, collapse-detection metrics like output variance).
- Log a "representation collapse" canary metric explicitly (e.g., average pairwise cosine similarity of a fixed validation batch's latents) — this is the single most common JEPA failure mode and should be visible on every run's dashboard from day one, not added after something goes wrong.

---

## 7. Evaluation Plan

### 7.1 Why this needs a custom eval plan

No existing benchmark or prior published model targets this exact combination (kinematic-taxel + JEPA + articulated hand), so there is no external SOTA number to chase. Evaluation must be primarily **intrinsic** and **ablation-driven**.

### 7.2 Required ablations (this is the actual research contribution — do not skip any of these)

| Ablation | What changes | Tests |
|---|---|---|
| **Baseline (full model)** | As specified in §5 | — |
| **No-FK** | Replace exact FK-computed taxel positions with only the raw 22-dim joint angle vector as a single opaque global feature (no per-taxel position); model must learn geometry implicitly if it can | H1 |
| **Image-native** | Same underlying simulated contact events, rendered as an optical depth/pseudo-tactile image (simple synthetic renderer, not full TACTO — a depth-image projection of the contact patch is sufficient for this comparison) fed to a ViT-style patch encoder instead of the graph encoder | H2 |
| **Reconstruction objective** | Replace JEPA latent-prediction loss with direct regression to raw future per-taxel force values (no target encoder, no EMA, no latent space) | H3 |
| **No anti-collapse regularization** | Remove VICReg term, keep EMA teacher only | Sanity-checks whether EMA alone suffices |

### 7.3 Intrinsic physics-probing metrics

For the main model and every ablation above, report (on a held-out, object-disjoint validation split):
- Per-taxel force magnitude regression: MAE, R².
- Slip binary classification: accuracy, F1, AUROC.
- Contact area regression: MAE, R².
- Representation collapse canary (§6.5) — report even for "successful" runs, as a sanity check.

### 7.4 Downstream task transfer (the standard foundation-model validation)

Freeze the trained encoder; train small task heads on top for simulated downstream tasks, and compare sample efficiency against (a) training the same task head from raw taxel input with no pretraining, and (b) training on top of the image-native ablation's frozen encoder:

- **Grasp stability classification:** will this grasp hold under simulated perturbation (small object nudge)? Binary classification.
- **Slip-onset early prediction:** predict slip N timesteps before it's visible in raw force signal (this directly exercises the predictive/world-model property, not just static representation quality).
- **In-hand reorientation success prediction** (stretch, Stage D data required).

Report sample-efficiency curves (task accuracy vs. number of labeled training episodes) — the headline claim is a specific, falsifiable number, e.g. "≥20% fewer labeled episodes needed to reach 90% grasp-stability accuracy vs. no pretraining," decided and written down before running the experiment, not after.

### 7.5 External real-data validation (stretch goal, not a requirement)

- If ART-Glove's or OSMO's dataset is publicly released with raw per-taxel force + joint-angle logs, this becomes a genuine zero-shot transfer test: run the frozen encoder (no fine-tuning) on real data and report probe accuracy degradation relative to sim-validation numbers. Check dataset availability at the time this project starts — do not assume it is available; this PRD does not depend on it.
- Explicitly disclose in any writeup that, absent real data, all validation is simulation-internal — this is a real, stated limitation, not something to obscure.

### 7.6 Definition of done / success criteria

This project has succeeded at the research-contribution level if:
1. H1, H2, and H3 are each either confirmed or cleanly falsified with the ablation suite in §7.2 (a clean negative result is still a valid, reportable outcome — do not p-hack or cherry-pick).
2. The downstream sample-efficiency claim in §7.4 has a specific reported number, positive or negative.
3. All results are reproducible from a documented config + seed in the repo.

---

## 8. Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11 | Match Genesis's supported version at build time |
| Simulator | Genesis (`genesis-world`) | Rigid-body + contact solver; no optical rendering module needed for v1 |
| Robot description | URDF (Allegro Hand, open-source) | Confirm license for redistribution; store a pinned copy in-repo under `assets/urdf/` |
| ML framework | PyTorch 2.x | bf16 mixed precision on Nebius H100-class GPUs |
| Graph neural nets | PyTorch Geometric (`torch_geometric`) | GATv2Conv or equivalent attention-based graph layer |
| Config management | Hydra (or plain YAML + argparse if Hydra adds too much overhead for a solo project — decide in Phase 0) | |
| Data format | WebDataset (tar shards) | Consistent with prior sim-data-pipeline experience |
| Experiment tracking | Weights & Biases | `WANDB_API_KEY` required |
| Distributed training | `torchrun` + DDP | FSDP only if needed |
| Compute | Nebius AI Cloud (GPU VMs) | Confirm current instance SKUs/pricing at provisioning time via Nebius console or CLI |
| Artifact/dataset hosting | Hugging Face Hub | For checkpoints and any released dataset shards; use the `hf` CLI |
| Version control | Git + GitHub | New, separate repository per user instruction |
| CI | GitHub Actions | Lint + unit tests only (no GPU jobs in CI) |
| Testing | pytest | Unit tests for: FK correctness, taxel force synthesis conservation-of-force check, graph construction edge counts, encoder/predictor shape correctness, loss collapse canary |
| Environment reproducibility | Docker (CPU image for sim/data-gen; separate GPU image for training) | Mirrors the Dockerfile-per-purpose pattern from prior project experience |
| Package management | `uv` (fast) or `pip` | Pin exact versions in `pyproject.toml` |

### 8.1 External APIs / accounts required before starting

- **Nebius AI Cloud** account + API key/CLI auth — for GPU instance provisioning and object storage.
- **Weights & Biases** account + API key.
- **Hugging Face Hub** account + token — for checkpoint/dataset hosting.
- **GitHub** — new repository, `gh` CLI auth for PR/issue workflows if desired.
- No other third-party APIs are required — this project has no external data-collection dependency and no real-hardware API surface.

---

## 9. Phased Roadmap (build order — do not start Nebius GPU billing before Phase 5)

| Phase | Deliverable | Exit criteria | Requires GPU billing? |
|---|---|---|---|
| 0 | Repo scaffold, `pyproject.toml`, CI skeleton, this PRD committed as `PRD.md`, `ROADMAP.md` capturing this table | `pytest` runs (even with zero tests), CI green | No |
| 1 | Genesis environment: Allegro-class hand URDF loaded, single object, basic press episode runs headless | A single episode's raw contact-solver output can be dumped to disk and inspected | No (CPU only) |
| 2 | Taxel layout generation (FPS per link) + taxel force synthesis (§5.3) + forward-kinematics module, with unit tests (force conservation, FK correctness against known joint configs) | Unit tests pass; a visualized (matplotlib/plotly) taxel force heatmap for one episode looks physically sensible | No |
| 3 | Graph construction module + dataset sharding (WebDataset) + Stage A (static press) data generation at small scale (hundreds of episodes) | A PyTorch `Dataset`/`DataLoader` yields correctly-shaped graph batches | No |
| 4 | Full model implementation: online + EMA target encoder, predictor, JEPA loss, VICReg term, physics probe heads — trained on tiny Stage A data on CPU or a single cheap GPU for a few hundred steps, purely to confirm the loop runs and loss moves | Loss decreases, collapse canary does not indicate immediate collapse, no crashes across all ablation code paths (§7.2) implemented and runnable | Optional, minimal (a few minutes on a small single GPU is acceptable to validate the loop; not a training run) |
| 5 | **Scale-up decision point** — review Phase 0-4 results with the user before provisioning any real Nebius training cluster | Explicit human go-ahead | **Yes, from here on** |
| 6 | Stage B/C data generation at full scale on Nebius (CPU-side) + full pretraining run(s) including all ablations from §7.2 | Ablation results reported per §7.6 | Yes |
| 7 | Downstream task transfer evaluation (§7.4) | Sample-efficiency numbers reported | Yes (small, task-head training is cheap relative to pretraining) |
| 8+ | Stretch: soft-body coupling (§5.4), real-data zero-shot validation if datasets available (§7.5), longer-horizon Stage D | — | Yes |

---

## 10. Repository Structure (proposed)

```
tack-jepa/
├── PRD.md                          this document
├── ROADMAP.md                      phase table from §9, kept up to date
├── pyproject.toml
├── README.md                       short pointer to PRD + quickstart
│
├── assets/
│   └── urdf/                       pinned Allegro Hand URDF + meshes
│
├── sim/
│   ├── hand_env.py                 Genesis scene: hand + object + episode rollout
│   ├── taxel_layout.py             FPS taxel placement per link (build-time artifact)
│   ├── taxel_force_synthesis.py    §5.3 kernel-distribution module
│   ├── forward_kinematics.py       §5.2/5.5 FK module
│   └── episode_generator.py        Stage A/B/C/D trajectory generation
│
├── data/
│   ├── graph_construction.py       §5.5
│   ├── dataset.py                  WebDataset shard reader, PyG batching
│   └── shard_writer.py
│
├── models/
│   ├── encoder.py                  GATv2-based context/target encoder
│   ├── predictor.py                action-conditioned causal transformer
│   ├── jepa_loss.py                latent loss + VICReg
│   ├── probes.py                   §5.9 physics probe heads
│   └── ablations/                  §7.2 variant implementations (no_fk, image_native, reconstruction)
│
├── training/
│   ├── train.py
│   ├── configs/                    Hydra/YAML configs per stage + ablation
│   └── ema.py
│
├── eval/
│   ├── physics_probes_eval.py      §7.3
│   ├── downstream_transfer.py      §7.4
│   └── collapse_canary.py          §6.5
│
├── nebius/
│   ├── README.md                   provisioning + deallocation commands, cost log
│   └── launch_training.sh
│
├── tests/
│   └── ...                         unit tests per §8 (Testing row)
│
└── docs/
    └── literature.md                §3 table, kept current
```

---

## 11. Risks and Open Questions

- **Taxel force synthesis fidelity (§5.3) is a modeling assumption, not ground truth** — the kernel-distribution approach is a simplification of real transducer physics. This is explicitly flagged as a limitation and a v2 target (§5.4), not hidden.
- **Graph scale/compute cost:** ~2,000+ taxel nodes per timestep, with sequences of many timesteps, could be expensive for a full graph-attention transformer. Mitigation: cap radius-graph neighbors (k=16), consider hierarchical pooling (per-link pooling before cross-link attention) if compute becomes a bottleneck — flag this as a Phase 4 engineering decision, not pre-solved here.
- **H1 might be falsified** — it's possible a large enough model learns implicit geometry from raw joint angles well enough that explicit FK grounding doesn't measurably help. This is a valid, useful, reportable outcome, not a project failure.
- **URDF/license check** — confirm the specific Allegro Hand URDF source used is appropriately licensed for redistribution in a new public or private repo before committing it.
- **Nebius pricing/SKUs will have changed by the time this is built** — do not hardcode instance prices or names into training scripts; confirm live via Nebius console/CLI at provisioning time.
- **Real dataset for external validation may never materialize** — do not architect anything that assumes ART-Glove/OSMO data will be available; §7.5 is explicitly a stretch goal.

---

## 12. Glossary

- **JEPA (Joint-Embedding Predictive Architecture):** self-supervised training paradigm (LeCun et al.) where a model predicts the latent representation of a target (rather than reconstructing raw values), using an online encoder, an EMA "target" encoder, and a predictor network.
- **Taxel:** a single discrete tactile sensing element (by analogy to "pixel"), here representing one node of a distributed force-sensing array on a rigid link.
- **Forward kinematics (FK):** the deterministic computation of a robot's link/joint 3D poses in world frame given known joint angles and the kinematic chain (URDF).
- **EMA (exponential moving average) teacher:** a copy of the online encoder's weights updated as a slow-moving average, used as a stable prediction target to prevent representational collapse.
- **VICReg:** Variance-Invariance-Covariance Regularization, a standard anti-collapse regularization term used alongside or instead of EMA teachers in self-supervised latent-prediction methods.
- **Tactile pollution:** term coined by the Tactile-WAM paper for the failure mode where naive fusion of tactile signal into a multimodal model degrades unrelated (e.g. visual/action) predictions.

---

*TacK-JEPA PRD | v0.1 | Pre-training phase — architecture and eval harness under construction*
