# CPU image for simulation / data generation (PRD §8).
# Genesis's rigid-body solver is CPU-bound (§6.4) — never run sim on GPU-billed
# time. Build/run from the repo root:
#
#   docker build -f docker/sim.Dockerfile -t tack-jepa-sim .
#   docker run --rm -v "$(pwd):/workspace" tack-jepa-sim \
#       python -m sim.episode_generator --stage b --out datasets/stage_b --per-variant 35

FROM python:3.10-slim

# Genesis / trimesh / pyglet pull in a handful of native libs (OpenGL, X11
# headers for headless rendering contexts, git for pip VCS installs).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    libgl1 libglu1-mesa libx11-6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml README.md ./
# Editable install of just the metadata first (no source yet) keeps the pip
# layer cacheable across source-only changes; copy the rest after.
COPY sim ./sim
COPY data ./data
COPY assets ./assets

RUN pip install --no-cache-dir -e .[dev,sim] \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .

CMD ["python", "-m", "pytest", "-q"]
