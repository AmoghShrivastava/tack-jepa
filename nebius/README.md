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
