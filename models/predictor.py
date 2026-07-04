"""Action-conditioned latent predictor (PRD §5.7).

A small causal transformer over the last N online-encoder context vectors,
cross-attending to embedded actions for timesteps t-N+1 .. t+k, predicting the
TARGET encoder's context vector k steps ahead. A learned [PRED] query token
(with a horizon embedding for curriculum over k) reads out the prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ActionConditionedPredictor(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        action_dim: int = 22,
        n_layers: int = 4,
        heads: int = 8,
        context_len: int = 8,
        max_horizon: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.context_len = context_len
        self.max_horizon = max_horizon
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.ctx_pe = nn.Parameter(torch.zeros(context_len, dim))
        self.act_pe = nn.Parameter(torch.zeros(context_len + max_horizon, dim))
        self.pred_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.horizon_emb = nn.Embedding(max_horizon + 1, dim)
        nn.init.normal_(self.ctx_pe, std=0.02)
        nn.init.normal_(self.act_pe, std=0.02)
        nn.init.normal_(self.pred_token, std=0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 2,
            batch_first=True,
            norm_first=True,
            dropout=0.0,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(dim)

    def forward(
        self,
        context: torch.Tensor,  # (B, N, dim) past online context vectors, oldest first
        actions: torch.Tensor,  # (B, N + k, action_dim) actions t-N+1 .. t+k
        horizon: int = 1,
    ) -> torch.Tensor:
        B, N, _ = context.shape
        if N > self.context_len:
            raise ValueError(f"context length {N} exceeds max {self.context_len}")
        if horizon > self.max_horizon:
            raise ValueError(f"horizon {horizon} exceeds max {self.max_horizon}")

        tgt = context + self.ctx_pe[:N]
        pred = self.pred_token.expand(B, 1, -1) + self.horizon_emb(
            torch.full((B, 1), horizon, device=context.device, dtype=torch.long)
        )
        tgt = torch.cat([tgt, pred], dim=1)  # (B, N+1, dim)

        # causal mask over [context .. PRED]; PRED sits last so it sees all context
        L = N + 1
        causal = torch.triu(
            torch.ones(L, L, device=context.device, dtype=torch.bool), diagonal=1
        )

        memory = self.action_mlp(actions) + self.act_pe[: actions.shape[1]]
        h = self.decoder(tgt=tgt, memory=memory, tgt_mask=causal)
        return self.out_norm(h[:, -1])  # (B, dim) predicted latent at t+k
