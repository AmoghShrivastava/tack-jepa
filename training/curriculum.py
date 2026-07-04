"""Prediction-horizon curriculum: k=1 up to k=8 as training stabilizes (PRD §5.7, §6.2).

Splits total training steps evenly across an increasing sequence of horizons
(1, 2, 4, 8 by default, capped at the predictor's max_horizon) so early
training predicts one step ahead (easiest) and later training predicts
further out. Each stage needs its own DataLoader (windows are built with a
fixed horizon at construction time — see data/dataset.py), so callers build
one loader per stage and use `horizon_at(step)` to pick which iterator to
pull from.
"""

from __future__ import annotations

import numpy as np

DEFAULT_STAGES = (1, 2, 4, 8)


def curriculum_stages(max_horizon: int, stages: tuple = DEFAULT_STAGES) -> list[int]:
    out = [k for k in stages if k <= max_horizon]
    return out or [1]


def make_horizon_schedule(total_steps: int, max_horizon: int, stages: tuple = DEFAULT_STAGES):
    """Returns horizon_at(step) -> int, stepping through curriculum_stages()
    at even intervals across [0, total_steps)."""
    ks = curriculum_stages(max_horizon, stages)
    bounds = np.linspace(0, total_steps, len(ks) + 1)[1:]

    def horizon_at(step: int) -> int:
        for k, b in zip(ks, bounds, strict=True):
            if step < b:
                return k
        return ks[-1]

    return horizon_at, ks
