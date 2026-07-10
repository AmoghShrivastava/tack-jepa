# TacK-JEPA

**Tactile Kinematic Joint-Embedding Predictive Architecture** — a force-native,
kinematically-grounded, action-conditioned world model for articulated
multi-taxel tactile sensing. Built entirely in simulation (Genesis physics
engine); no physical robot, tactile sensor, or lab required.

**Live demo:** [tack.tenacelabs.com](https://tack.tenacelabs.com) · **Weights:**
[huggingface.co/AmoghShrivastava1/tack-jepa](https://huggingface.co/AmoghShrivastava1/tack-jepa)
· **License:** MIT (see [License](#license))

---

## What this is

Existing tactile foundation models (SPARSH, AnyTouch, T3, UniT, TacForeSight,
and others) all represent touch as a 2D image and ask a vision-style encoder
to *infer* where in 3D space a contact happened from pixel appearance — the
same problem monocular depth estimation solves, just applied to touch. That
approach makes sense for the sensors that exist today (GelSight, DIGIT):
one or two optical patches per hand, each imaged like a tiny camera.

But a newer class of tactile hardware doesn't have that constraint. Devices
like CMU's ART-Glove (2048 taxels, 22-DoF joint tracking) and OSMO's
open-source tactile glove are **distributed and kinematically tracked**: every
tactile element sits on a rigid link whose pose is fully determined by joint
encoder readings. On this class of hardware, the 3D position of every taxel
is knowable **exactly**, via forward kinematics — arithmetic, not inference.

**The bet this project tests:** if a model never has to spend capacity
inferring geometry it could just be given, all of that capacity should go
toward the actually-hard part — modeling contact *dynamics* (how force
evolves, propagates, predicts slip, and responds to action). TacK-JEPA is a
force-native (not image-native), kinematically-grounded (exact FK, not
learned), JEPA-style (latent prediction, not reconstruction) world model
built to test that hypothesis directly, with the ablations needed to falsify
it built into the same codebase. No comparable architecture existed in the
literature as of this project's writing (see [`docs/literature.md`](docs/literature.md)
for the full survey) — this is a research-novelty bet on where tactile
hardware is heading, not an incremental improvement on an existing baseline.

## The core idea, in one picture

```
Genesis physics sim (Allegro-class hand + object)
        │  rigid-body contact solver → per-contact normal + friction force
        ▼
Taxel force synthesis (deterministic, not learned)
        │  distributes contact forces onto a fixed ~2,200-taxel layout
        ▼
Forward kinematics (deterministic, not learned)
        │  joint angles → exact 3D world-frame taxel positions
        ▼
Graph construction
        │  nodes = taxels (position, normal, force, link id)
        │  edges = radius graph (world frame) ∪ static intra-link backbone
        ▼
Graph Attention encoder  ──(EMA)──▶  Target encoder (frozen, stop-gradient)
        │                                        │
        ▼                                        ▼
Predictor (latent(t) + action(t) → predicted latent(t+k))  vs.  target latent(t+k)
        │
        ▼
JEPA latent loss + VICReg anti-collapse regularization
```

Everything above the encoder is **exact, deterministic physics/geometry** —
no learning happens until the graph encoder. The only thing the model has to
learn is what a JEPA model always learns: how the world's latent state
evolves under action.

## Hypotheses, and what the ablation suite actually found

This is a research project, not a product, so it's built around three
falsifiable hypotheses — each testable with an ablation already implemented
in this same codebase (`models/ablations/`), no external dataset needed:

| | Hypothesis | Ablation that tests it |
|---|---|---|
| **H1** | Exact FK-computed taxel positions improve physics-probe accuracy over an ablated variant that only sees raw joint angles as an opaque vector | `no_fk` |
| **H2** | A force-native taxel-graph encoder beats an image-patch encoder fed an equivalent rendered view of the same contact event | `image_native` |
| **H3** | JEPA-style latent prediction produces representations that transfer better than a model trained to reconstruct raw future taxel values | `reconstruction` |

A fifth ablation, `no_vicreg`, isolates whether the EMA teacher alone is
enough to prevent representational collapse, or whether VICReg regularization
is doing real work.

All five variants were trained on the full-scale Stage C dataset (16 object
variants, press/grasp/slide trajectories, 4000 episodes, object-disjoint
train/val split) for an equal 6000-step budget each on a single GPU. Honest
results, including the ones that complicated the simple story:

- **`no_vicreg` collapses — a real, meaningful negative result.** Its
  representation ends up both cosine-pinned *and* has near-zero per-dimension
  variance (`mean_dim_std` 0.0097 vs. baseline's 0.174), matching
  `reconstruction`'s expected collapse (it has zero anti-collapse machinery
  by design). This is direct evidence that VICReg is doing necessary work in
  the full model, not decoration — an EMA teacher alone was not sufficient
  here, consistent with prior self-supervised-learning results elsewhere.
- **`image_native` genuinely collapsed at first, was root-caused, and the fix
  was confirmed on a full retrain.** A first training pass showed total
  collapse (canary pinned at 1.0000, representation variance four orders of
  magnitude below baseline). Root cause: a hardcoded constant "occupancy"
  channel in the image rasterizer accounted for 100% of the input image's
  signal energy, so the encoder was effectively looking at a fixed image
  regardless of contact. After fixing the rasterizer to encode genuine
  per-taxel contact and retraining, representation variance rose roughly
  19,000× and the collapse signature disappeared — confirmed in practice, not
  just in a unit test.
- **A real, still-open finding: `image_native` beats the graph encoder on
  slip detection, and the gap doesn't shrink with more training.** At a
  matched probe-training budget, `image_native`'s slip-AUROC (0.94) is well
  ahead of `baseline`'s (0.60) — the gap actually *widened* with more probe
  training, not narrowed, which argues against "undertrained probe" as the
  explanation. The current best hypothesis is that slip is an inherently
  spatially-coarse signal that the image encoder's patch-shared
  representation captures more directly, while the graph encoder's
  fine-grained per-taxel output may be encoding at a resolution not well
  suited to this particular downstream task. Reported as a genuine, unresolved
  finding rather than smoothed over — see `ROADMAP.md`'s decisions log for
  the full investigation.
- **A diagnostic bug was caught and fixed along the way:** the collapse
  "canary" metric (raw pairwise cosine similarity) reads as pinned near 1.0
  for representations with a large shared mean vector, even when their
  per-dimension variance is genuinely healthy — this affected `baseline` and
  `no_fk`'s readings and was confirmed by mean-centering before computing
  cosine similarity (canary dropped from 1.0000 → 0.0023 once the shared mean
  was removed). `mean_dim_std` was the reliable signal the whole time.

The full probe-regression numbers (force magnitude, contact area) and the
downstream sample-efficiency comparison (§7.4 of the PRD) are not yet
complete — see `ROADMAP.md` for exactly what's done, what's in progress, and
what's an open follow-up. Everything here is reported as-is, including the
parts that didn't confirm the naive expectation.

## Repository layout

```
tack-jepa/
├── PRD.md              full design doc — architecture, literature review, eval plan (read this for depth)
├── ROADMAP.md           phase-by-phase build log + decisions log (the most detailed, most current record)
├── docs/literature.md   survey of prior tactile-model work and how this project differs
│
├── sim/                 Genesis scene, taxel layout (farthest-point sampling), taxel force synthesis,
│                         forward kinematics, Stage A/B/C episode generators
├── data/                graph construction, WebDataset sharding, PyG batching
├── models/               GATv2-based encoder, action-conditioned predictor, JEPA + VICReg losses,
│                         physics probe heads, and the four §7.2 ablation variants
├── training/             training loop, horizon curriculum, EMA, YAML configs
├── eval/                 physics probes, downstream transfer, collapse canary
├── docker/               CPU (sim/data-gen) and GPU (training) images
├── nebius/, azure/       compute provisioning logs (cost, decisions, deallocation discipline)
├── demo/                 the interactive grasp-episode demo behind tack.tenacelabs.com
├── assets/urdf/          vendored Allegro Hand URDF + meshes (see their own licenses)
└── tests/                53 unit tests: FK correctness, force conservation, graph construction,
                          model shapes, collapse-canary regression tests
```

## Quickstart

```bash
python -m venv .venv            # Python 3.10–3.13
.venv/Scripts/activate          # Windows; use .venv/bin/activate on Linux/macOS
pip install -e .[dev,sim,ml]
pip install torch --index-url https://download.pytorch.org/whl/cpu  # or a CUDA build
pytest
ruff check .
```

Or via Docker: `docker/sim.Dockerfile` (CPU, simulation/data-gen) and
`docker/train.Dockerfile` (GPU, training — reconcile the CUDA tag against your
actual GPU's driver before building; see the file's header comment).

### End-to-end pipeline

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

**Stage A** = static press-only (fixed object, floating wrist approaches and
holds, constant finger pose — no motion during contact). **Stage B** =
dynamic grasp (free object, wrist settles into engagement pose while fingers
actively close — real action-conditioned dynamics and slip). **Stage C**
(the main pretraining stage used for the results above) scales this to 16
object variants and diverse grasp/slide/press trajectories. See PRD §6.1 and
`ROADMAP.md`'s "Audit and correction" section for the full stage definitions
and a real discrepancy this project caught and fixed against its own spec.

## Try it

The live demo at [tack.tenacelabs.com](https://tack.tenacelabs.com) steps
through real grasp episodes rendered directly from the trained model's
contact-sensing pipeline — not a mockup. Source in `demo/`.

## Status

All five §7.2 ablation variants have been trained to completion on the
full-scale Stage C dataset and evaluated with the collapse-canary diagnostic
(see [Hypotheses](#hypotheses-and-what-the-ablation-suite-actually-found)
above). The full physics-probe regression pass and downstream
sample-efficiency comparison are in progress. See `ROADMAP.md` for the
complete, currently-accurate phase table and decisions log — it is kept up
to date and is the authoritative source on exactly what's done.

## License

MIT — see [LICENSE](LICENSE). Third-party Allegro Hand URDF/mesh assets
under `assets/urdf/` are vendored from DexSuite/SimLab and remain under
their original licenses (see the `LICENSE`/`DEX_URDF_LICENSE` files in that
directory). Model weights on Hugging Face are also released under MIT; the
training data itself is not released.
