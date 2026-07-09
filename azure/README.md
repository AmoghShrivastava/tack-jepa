# Azure eval VM log

Local-machine-only follow-up to the Stage C GPU training effort (see
`nebius/README.md` for the training run itself). The full `physics_probes_eval.py`
pass (fresh regression probes: force_mag/slip/contact_area) was deferred during
the original Nebius run due to a loader-warmup slowness issue, and the local
Windows dev machine turned out to be memory-constrained (16GB total) for the
graph-encoder variants. Provisioned an Azure CPU VM (no GPU needed — this is
inference/probe-training, not JEPA pretraining) to finish it, explicitly
authorized by the user ("run the eval tests on an azure vm ... i have
practically unlimited credits").

- **Resource group:** `tackjepa-eval-rg` (eastus) — dedicated, separate from
  the pre-existing unrelated `tactility-rg` project in the same subscription.
- **VM:** `tackjepa-eval-vm`, Ubuntu 22.04.
- **2026-07-08, initial provisioning:** `Standard_E4as_v5` and
  `Standard_E4s_v6` both failed with `SkuNotAvailable` (capacity restrictions
  in eastus). Landed on `Standard_D8s_v7` (8 vCPU, 32GB RAM, no restriction) —
  chosen as memory-optimized-ish general purpose since the known bottleneck
  was RAM, not GPU or raw CPU count.
- **Setup:** codebase transferred via `tar` + `scp` (repo is private on
  GitHub, simplest to skip credential setup on a throwaway VM). Stage C shard
  data (`datasets/shards_c`, 1.3GB, 127 shards) transferred the same way.
  `pip install -e ".[ml]"` (torch/torch_geometric/webdataset) into a venv.
  Checkpoints pulled from the private HF Hub repo (`hf auth login` device-code
  flow, same pattern as the Nebius runs) rather than re-transferring 1GB from
  local disk.
- **OOM #1 (batch_size=32, default):** `baseline`/`no_fk`/`no_vicreg`/
  `reconstruction` all crashed near-instantly with a single ~8.6GB tensor
  allocation failure in `GATv2Conv`'s edge-update step — this reproduced a
  smaller-scale version of the same crash seen on the local 16GB Windows
  machine, just confirming the graph encoder's per-batch memory footprint at
  the training batch size is large. `image_native` (ViT-based, no graph attn)
  ran fine throughout — finished in ~17 min, results matched the local
  16GB-machine run to 5 decimal places on canary/dim_std (good consistency
  check).
- **Added `--batch-size` CLI override** to `eval/physics_probes_eval.py`
  (defaults to the run's training config value if unset) so eval memory
  footprint can be decoupled from the training-time batch size without
  touching the archived `resolved_config.yaml` files.
- **OOM #2 (batch_size=8):** `baseline` ran for ~93 minutes — much longer than
  the instant batch=32 crash — before still being OOM-killed at the full
  32GB, RSS having grown gradually rather than spiking. Confirmed via
  `dmesg -T | grep 'killed process'` (kernel OOM-killer, not a Python-level
  crash — no traceback in the eval's own log). This ruled out "batch size is
  just too big" as the sole explanation: something in the probe-eval loop
  (most likely candidate: `physics_probes_eval.py`'s outer `while step <
  train_steps` loop calls `loader_for("train")` again on every full pass over
  the shard set, opening a fresh `WebDataset`/tar pipeline each time without
  the old one being released) leaks memory roughly per training step, not
  just per batch.
- **Decision point:** rather than spend further time debugging the leak's
  root cause, gave the user the choice (bigger VM to brute-force through it /
  actually fix the leak / shrink the eval workload). User chose to brute-force
  with a much bigger VM, citing effectively unlimited Azure credit.
- **Resize:** `az vm deallocate` → `az vm resize --size Standard_E16as_v7`
  (16 vCPU, 128GB RAM, no capacity restriction in eastus) → `az vm start`.
  Same public IP (`20.228.219.201`) preserved across the resize.
- **Confirmed the leak is real, not a batch-size artifact:** re-ran at full
  batch_size=32 on 128GB — `baseline`'s RSS climbed past the 32GB mark that
  killed it twice before (peaked ~38GB at one point, later observed
  oscillating down to ~5GB then back up to ~30GB across the run, consistent
  with the per-epoch loader-recreation theory: memory drops when a fresh
  pass starts, climbs again as it re-accumulates) without crashing, thanks to
  the much larger ceiling.
- **Process-monitoring lesson (self-correction):** misread `ps aux`'s
  cumulative-CPU-time column (`TIME`, format `MM:SS` once >1hr of CPU-seconds
  accumulate) as wall-clock elapsed time in two consecutive status updates to
  the user, reporting "2h32min" and "3h30min" elapsed when the real
  wall-clock elapsed (`ps -o etime`) was ~20-30 min — the discrepancy is
  explained by ~700% CPU utilization (7 of 16 cores) inflating cumulative CPU
  time roughly 7x over wall time. Caught via a direct `ps -o etime` check
  against `date`, corrected immediately and explicitly to the user rather
  than left standing.

**Status: all 5 variants complete** (2026-07-09, full results + closing
analysis in ROADMAP.md's decisions log). Not yet archived or torn down —
resource group `tackjepa-eval-rg` remains live and billing until the user
confirms deallocation/deletion (same shutdown discipline as the Nebius runs:
never leave a provisioned VM running without an explicit stop decision
logged here).
