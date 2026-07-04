"""Physics probe heads (PRD §5.9, §5.10).

Small MLPs on top of encoder latents predicting sim ground truth. Primary use
is FROZEN-encoder evaluation (§7.3); optional joint training must stay at
<=0.1x the JEPA loss weight to avoid the tactile-pollution analogue (§5.10).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


class PhysicsProbes(nn.Module):
    """force magnitude (per-taxel reg) | slip (per-taxel logit) | contact area (global reg)."""

    def __init__(self, node_dim: int = 256, global_dim: int = 512, hidden: int = 128):
        super().__init__()
        self.force_head = _mlp(node_dim, hidden, 1)
        self.slip_head = _mlp(node_dim, hidden, 1)
        self.area_head = _mlp(global_dim, hidden, 1)

    def forward(
        self, node_latent: torch.Tensor, global_latent: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {
            "force_mag": self.force_head(node_latent).squeeze(-1),   # (N,)
            "slip_logit": self.slip_head(node_latent).squeeze(-1),   # (N,)
            "contact_area": self.area_head(global_latent).squeeze(-1),  # (B,)
        }

    @staticmethod
    def losses(
        out: dict[str, torch.Tensor],
        force_mag: torch.Tensor,
        slip: torch.Tensor,
        contact_area: torch.Tensor,
        slip_pos_weight: float | None = 50.0,
    ) -> dict[str, torch.Tensor]:
        # slip labels are heavily imbalanced in Stage A (static presses slip
        # only in landing transients, ~1e-4 positive rate) — weight positives
        pw = (
            torch.tensor(slip_pos_weight, device=slip.device)
            if slip_pos_weight
            else None
        )
        return {
            "force_mag": nn.functional.mse_loss(out["force_mag"], force_mag),
            "slip": nn.functional.binary_cross_entropy_with_logits(
                out["slip_logit"], slip, pos_weight=pw
            ),
            "contact_area": nn.functional.mse_loss(out["contact_area"], contact_area),
        }
