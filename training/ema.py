"""EMA target-encoder update (PRD §5.6, §6.2).

Standard I-JEPA/V-JEPA/DINO-style teacher: target weights are an exponential
moving average of the online encoder's, never receiving gradients. Momentum
follows a linear schedule 0.996 -> 0.9999 over training.
"""

from __future__ import annotations

import torch


def ema_momentum(step: int, total_steps: int, start: float = 0.996, end: float = 0.9999) -> float:
    if total_steps <= 1:
        return end
    frac = min(max(step / (total_steps - 1), 0.0), 1.0)
    return start + frac * (end - start)


@torch.no_grad()
def ema_update(target: torch.nn.Module, online: torch.nn.Module, momentum: float):
    """target <- momentum * target + (1 - momentum) * online, in place."""
    for pt, po in zip(target.parameters(), online.parameters(), strict=True):
        pt.mul_(momentum).add_(po.detach(), alpha=1.0 - momentum)
    for bt, bo in zip(target.buffers(), online.buffers(), strict=True):
        bt.copy_(bo)


def make_target(online: torch.nn.Module) -> torch.nn.Module:
    """Deep-copied, gradient-free clone of the online encoder."""
    import copy

    target = copy.deepcopy(online)
    for p in target.parameters():
        p.requires_grad_(False)
    return target
