"""Reconstruction ablation head (PRD §7.2, tests H3).

Replaces the JEPA latent objective: the predictor's output vector is decoded
straight into raw future per-taxel force readings (T x 3) and regressed
against them. No target encoder, no EMA, no latent-space loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RawForceDecoder(nn.Module):
    def __init__(self, dim: int = 512, n_taxels: int = 2244, hidden: int = 1024):
        super().__init__()
        self.n_taxels = n_taxels
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, n_taxels * 3)
        )

    def forward(self, pred_latent: torch.Tensor) -> torch.Tensor:
        return self.net(pred_latent).view(-1, self.n_taxels, 3)

    @staticmethod
    def loss(decoded: torch.Tensor, target_force: torch.Tensor) -> torch.Tensor:
        """MSE against the raw future taxel force field (B, T, 3)."""
        return nn.functional.mse_loss(decoded, target_force)
