"""Taxel-graph context encoder: GATv2 attention over the taxel graph (PRD §5.6).

Two instances exist at training time: the online encoder (gradient-trained)
and the EMA target encoder (training/ema.py). `use_geometry=False` implements
the No-FK ablation (§7.2/H1): per-taxel world positions and normals are
withheld and the raw joint vector is injected as one opaque global feature.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import softmax as scatter_softmax


class GATBlock(nn.Module):
    """Pre-norm transformer-style block: GATv2 attention + FFN, both residual."""

    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.att = GATv2Conv(hidden, hidden // heads, heads=heads)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden)
        )

    def forward(self, x, edge_index):
        x = x + self.att(self.norm1(x), edge_index)
        return x + self.ffn(self.norm2(x))


class AttentionPool(nn.Module):
    """Learned attention pooling of node latents into one vector per graph."""

    def __init__(self, hidden: int, out_dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        w = scatter_softmax(self.score(x).squeeze(-1), batch)
        n_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        pooled = torch.zeros(n_graphs, x.shape[1], device=x.device, dtype=x.dtype)
        pooled.index_add_(0, batch, x * w.unsqueeze(-1))
        return self.proj(pooled)


class TaxelGraphEncoder(nn.Module):
    def __init__(
        self,
        n_links: int = 21,
        link_emb_dim: int = 16,
        hidden: int = 256,
        n_layers: int = 6,
        heads: int = 8,
        node_out: int = 256,
        global_dim: int = 512,
        qpos_dim: int = 22,
        use_geometry: bool = True,
    ):
        super().__init__()
        self.use_geometry = use_geometry
        in_dim = 3 + link_emb_dim + (6 if use_geometry else 0)  # force + link [+ pos/normal]
        self.link_emb = nn.Embedding(n_links, link_emb_dim)
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.blocks = nn.ModuleList(GATBlock(hidden, heads) for _ in range(n_layers))
        self.node_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, node_out))
        self.pool = AttentionPool(hidden, global_dim)
        # No-FK ablation: geometry is withheld, so the model gets the raw joint
        # vector as a single opaque global feature (PRD §7.2) fused post-pool
        if use_geometry:
            self.qpos_mlp = None
        else:
            self.qpos_mlp = nn.Sequential(
                nn.Linear(qpos_dim, global_dim), nn.GELU(), nn.Linear(global_dim, global_dim)
            )

    def forward(
        self,
        force: torch.Tensor,        # (N, 3) [f_normal, shear1, shear2]
        link_index: torch.Tensor,   # (N,) long
        edge_index: torch.Tensor,   # (2, E) long
        batch: torch.Tensor,        # (N,) long graph id per node
        pos: torch.Tensor,          # (N, 3) world taxel positions (FK)
        normal: torch.Tensor,       # (N, 3) world taxel normals (FK)
        qpos: torch.Tensor,         # (B, 22) raw joint state
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats = [force, self.link_emb(link_index)]
        if self.use_geometry:
            feats = [pos, normal, *feats]
        x = self.input_proj(torch.cat(feats, dim=-1))
        for blk in self.blocks:
            x = blk(x, edge_index)
        node_latent = self.node_head(x)
        global_latent = self.pool(x, batch)
        if self.qpos_mlp is not None:
            global_latent = global_latent + self.qpos_mlp(qpos)
        return node_latent, global_latent
