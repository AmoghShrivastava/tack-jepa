# GPU image for training (PRD §8). NOT for simulation/data-gen (§6.4: sim is
# CPU-bound and must never run on GPU-billed time — use docker/sim.Dockerfile
# and sync the resulting datasets/shards/ in).
#
# CUDA version must match the actual Nebius GPU VM's driver at provisioning
# time (§11: "Nebius pricing/SKUs will have changed... do not hardcode")  —
# confirm via `nvidia-smi` on the instance and adjust the base image tag and
# the torch --index-url below before building.
#
#   docker build -f docker/train.Dockerfile -t tack-jepa-train .
#   docker run --rm --gpus all -v "$(pwd):/workspace" tack-jepa-train \
#       python -m training.train --variant baseline train.device=cuda train.precision=bf16

FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY models ./models
COPY training ./training
COPY data ./data
COPY sim ./sim
COPY eval ./eval

# Verify the CUDA tag above against the Nebius VM before building; the
# --index-url below must match (cu121/cu126/... per current PyTorch release).
RUN pip install --no-cache-dir -e .[dev,ml] \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126 \
    && pip install --no-cache-dir wandb

COPY . .

CMD ["python", "-m", "training.train", "--variant", "baseline"]
