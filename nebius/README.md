# Nebius compute

**Phase 5 go-ahead received 2026-07-04** — user explicitly authorized GPU provisioning
and training in that conversation. The hard gate is now open; the discipline below
still applies to every instance from here on.

## Discipline for when provisioning does start (Phase 5+)

- Every launch command written in this directory MUST be accompanied, immediately
  adjacent, by its stop/deallocate command.
- Confirm current instance SKUs and pricing live via the Nebius console/CLI at
  provisioning time — never trust names/prices written earlier (PRD §11).
- Data generation (Genesis sim) is CPU-bound: run it on a CPU-only instance, never
  on GPU-billed time (PRD §6.4).
- Keep a cost log in this file: date, instance type, duration, purpose, cost.
- Use `docker/sim.Dockerfile` on the CPU instance and `docker/train.Dockerfile` on
  the GPU instance (PRD §8) — reconcile the GPU image's CUDA tag / torch index
  against the actual VM's driver via `nvidia-smi` before building; don't trust the
  tag committed in the file (SKUs and CUDA versions change, see PRD §11).

## Cost log

### Instance 1: `magenta-gorilla-instance-9` (validation run)

- **Launched:** 2026-07-04, via Nebius console (project `default-project-eu-north1`,
  region eu-north1) — see reasoning below.
- **Platform:** NVIDIA L40S AMD (`gpu-l40s-d`), 1 GPU, 16 vCPUs, 96 GiB RAM,
  **Preemptible** (can be stopped anytime by Nebius with 60s warning — acceptable for
  a short validation run, not used for the long ablation sweep).
- **Boot disk:** Ubuntu 24.04 LTS for NVIDIA GPUs (CUDA 13), 200 GiB SSD (reduced from
  the 1280 GiB default — not needed for this workload).
- **Live price at provisioning (2026-07-04, confirmed in console, NOT trusted from
  memory per §11):** compute $0.89/hr + storage $0.02/hr = **$0.91/hr total**.
- **Why L40S, not H100:** model is <100M params (PRD §6.3); H100's memory
  bandwidth/NVLink advantages target much larger models. L40S is roughly a third the
  cost of H100 ($2.15/hr) and should be more than sufficient — this run's actual
  purpose is to get real steps/sec numbers to confirm that.
- **Instance ID:** `computeinstance-e00by75pxht27swrrb`, public IP `89.169.99.205`.
- **Deallocate command (run the moment this instance is no longer needed):**
  Console: VM overview page → "..." menu → Stop, then Delete once confirmed done.
  (CLI equivalent, if `nebius` CLI gets set up later: `nebius compute instance stop
  --id computeinstance-e00by75pxht27swrrb`.)
- **Preemption event:** instance was reclaimed by Nebius mid-sweep (~4.5 hours in, during
  the `reconstruction` variant) — confirmed via console (Stop Instance operation) and
  GPU/CPU metrics dropping to idle, not a training bug. `baseline`/`no_fk`/`image_native`
  had already checkpointed and were unaffected; `reconstruction` (no checkpoint had been
  hit yet at `checkpoint_every=500` default) lost ~760 steps, restarted at ~$0 cost
  (a few minutes). Restarted the same instance (console "Start VM") — got the same
  public IP back, disk/checkpoints intact. Increased `checkpoint_every` to 150 for the
  remaining variants to reduce any future loss.
- **Final status:** all 5 §7.2 variants trained to completion (2000 steps baseline,
  800 steps each ablation) + full eval harness (probe eval, downstream transfer,
  collapse canary) run against all checkpoints.
- **Checkpoints archived to HF Hub, then instance deleted entirely:** pushed all 5
  `checkpoint.pt` + metrics/config/probe_eval to the private model repo
  `AmoghShrivastava1/tack-jepa-phase6-checkpoints` (2026-07-05) via `hf upload`
  (device-code login, no token ever handled by the assistant). Decision to delete
  rather than leave stopped: this disk's storage-only rate is $0.02/hr ($1.44 over
  3 days); re-provisioning fresh later costs ~$0.08 in one-time setup overhead
  (reinstall torch/PyG/webdataset, ~5 min) since the Stage A/B shards (113MB) are
  trivial to re-upload — break-even is ~4 hours, so deleting after archiving wins
  for any gap longer than that. Checkpoints retrievable anytime via
  `hf download AmoghShrivastava1/tack-jepa-phase6-checkpoints`.
- **Actual total cost: $8.23** (account balance $135.40 -> $127.17 across the whole
  session — provisioning, the memory/OOM investigation, the full training sweep
  including one preemption recovery, and the complete eval pass). Well within the
  $9-15 estimate given after the OOM/accumulation finding.

### Instance 2: `tackjepa-stagec` (Stage C full retrain, all 5 variants)

- **Launched:** 2026-07-06, via Nebius console (project `default-project-eu-north1`,
  region eu-north1), after PRD-scale Stage C data generation (16 object variants:
  6 primitives + 4 Genesis-bundled meshes + 6 procedural superquadrics; press/grasp/
  slide trajectories; 4000 episodes -> 3250 train / 750 val object-disjoint shards,
  see ROADMAP.md Stage C decisions log) and the `image_native` collapse fix
  (see prior ROADMAP.md entry) were both complete.
- **Platform:** NVIDIA L40S AMD (`gpu-l40s-d`), 1 GPU, 16 vCPUs, 96 GiB RAM,
  **Preemptible**.
- **Boot disk:** Ubuntu 24.04 LTS for NVIDIA GPUs (CUDA 13), 250 GiB SSD (bumped up
  from 200 GiB last time — Stage C's shards are larger, 1.4GB vs 113MB).
- **Live price at provisioning (2026-07-06, confirmed in console):** compute
  $0.89/hr + storage $0.03/hr (250GiB) = **$0.92/hr total**.
- **Instance ID:** `computeinstance-e00z5mfrpz1d5v7s6b`, public IP `89.169.103.86`.
- **Deallocate command (run the moment this instance is no longer needed):**
  Console: VM overview page → Settings tab → "Delete virtual machine" (deletes VM
  + disk together, confirmed by name). CLI equivalent if set up:
  `nebius compute instance delete --id computeinstance-e00z5mfrpz1d5v7s6b`.
- **Training budget:** all 5 §7.2 variants (baseline, no_fk, image_native [with the
  occupancy-channel fix], reconstruction, no_vicreg) at an **equal 6000-step budget
  each** (30,000 variant-steps total) — deliberately equal this time for a fair
  comparison, unlike Phase 6's asymmetric 2000/800 split. `data.shard_dir=
  datasets/shards_c`, bf16, effective batch 32 via gradient accumulation
  (micro-batch 4), `checkpoint_every=150`. Launched sequentially via
  `run_stagec_sweep.sh` under `nohup` so the sweep survives SSH disconnects.
- **Cost estimate (given to user before launch, explicitly approved):** based on
  Phase 6's actual rate (~575 variant-steps/hour including eval overhead), 30,000
  variant-steps projects to **~52 hours (~2.2 days), ~$47-50** — a real jump from
  Phase 6's $8.23/~5hrs, flagged and confirmed with the user beforehand given both
  the cost and the higher cumulative preemption risk of a multi-day continuous run.
- **Status:** training started 2026-07-06 (`baseline_stagec` step 0 confirmed
  running, 93% GPU utilization, ~17.7GB VRAM). In progress — will update with
  final cost/outcome once complete.
- **Preemption event:** instance was reclaimed by Nebius partway through `no_fk`
  (confirmed via console: "Stop Instance" performed by "None", i.e. not
  user-initiated). `baseline_stagec` had already finished all 6000 steps
  beforehand (unaffected); `no_fk_stagec` was interrupted around step ~650-700,
  with its last checkpoint saved at step 600 (`checkpoint_every=150` limited the
  loss to under 100 steps). Restarted the same instance via console — got a new
  public IP (89.169.103.86 -> 89.169.103.53), disk/checkpoints intact. Relaunched
  the remaining variants (`no_fk`, `image_native`, `reconstruction`, `no_vicreg`)
  via a new `run_stagec_sweep2.sh`; `no_fk` auto-resumed cleanly from its step-600
  checkpoint (confirmed in logs: "resumed from checkpoint at step 600"), GPU back
  to 100% utilization. No manual data loss beyond the ~50-100 uncheckpointed
  steps.
- **SIGTERM handler added mid-sweep** (training/train.py): catches Nebius's
  ~60s preemption warning and force-saves a checkpoint immediately instead of
  waiting for the next `checkpoint_every` interval — verified with an in-
  process `signal.raise_signal` test before deploying. Applied by copying the
  updated file to the VM and doing a controlled restart of the in-progress
  `no_fk` run (resumed cleanly from its existing checkpoint, ~90 steps lost)
  so the rest of the sweep is protected.
- **GPU utilization optimization**: `image_native` (a small ViT over a tiny
  mosaic image) only uses ~1.5GB/46GB VRAM and a few % utilization — since
  billing is per wall-clock hour regardless of utilization, ran `reconstruction`
  concurrently alongside it as a second process (independent run_name/
  checkpoint paths, no conflict) rather than waiting for the sequential sweep
  script to get to it. Killed only the sequential wrapper script (leaving
  `image_native`'s process untouched) and set up a small waiter script to
  auto-launch `no_vicreg` once `reconstruction` exits. Combined VRAM usage
  ~26.4GB/46GB. Estimated savings: ~3.5 hours of wall-clock (roughly
  `image_native`'s remaining runtime, since it now overlaps with
  `reconstruction` instead of running before it).
