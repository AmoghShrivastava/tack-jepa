"""JEPA latent loss + VICReg anti-collapse regularization (PRD §5.7, §5.8).

The core objective: smooth-L1 between the predictor's output and the
STOP-GRADIENT target-encoder latent — predicting representations, not raw
values, is what makes this JEPA. VICReg's variance + covariance terms guard
against collapse alongside the EMA teacher (belt and suspenders per the PRD;
ablate `no_vicreg` later to see if one suffices).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def jepa_latent_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Huber loss to a detached target. (B, D) x (B, D) -> scalar."""
    return F.smooth_l1_loss(predicted, target.detach())


def vicreg_regularizer(
    z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Variance + covariance terms of VICReg on a batch of latents (B, D).

    variance: hinge pushing every dimension's batch std above gamma
    covariance: squared off-diagonal covariance, decorrelating dimensions
    (the invariance term is played by the JEPA prediction loss itself)
    """
    if z.shape[0] < 2:
        zero = z.sum() * 0.0
        return zero, zero
    z = z - z.mean(dim=0)
    std = torch.sqrt(z.var(dim=0) + eps)
    var_loss = F.relu(gamma - std).mean()
    n, d = z.shape
    cov = (z.T @ z) / (n - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag**2).sum() / d
    return var_loss, cov_loss


@torch.no_grad()
def collapse_canary(z: torch.Tensor) -> float:
    """Mean pairwise cosine similarity of a batch of latents (B, D).

    ~1.0 -> representations collapsed to a constant vector; healthy training
    sits well below. Logged on every run from day one (PRD §6.5).
    """
    if z.shape[0] < 2:
        return float("nan")
    zn = F.normalize(z.float(), dim=1)
    sim = zn @ zn.T
    b = sim.shape[0]
    return ((sim.sum() - b) / (b * (b - 1))).item()
