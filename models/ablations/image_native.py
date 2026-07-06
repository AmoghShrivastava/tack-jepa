"""Image-native ablation encoder (PRD §7.2, tests H2).

The SAME simulated contact events are rendered as 2D pseudo-tactile images —
one 16x16 tile per link, taxel readings rasterized onto a fixed 2D chart of
each link's surface (per-link PCA plane of the taxel cloud, computed once from
the layout) — arranged into a mosaic and fed to a ViT-style patch encoder.
Exactly what optical-tactile-image models consume: per-element force as
pixels, geometry only implicit in the chart. No 3D positions, no FK, no graph.

Per-taxel probe outputs are read back through each taxel's pixel -> patch
token, mirroring how image-based models are probed. Channels per pixel:
[f_normal, |shear|, occupancy].
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from sim.taxel_layout import TaxelLayout

TILE = 16          # pixels per link tile side
MOSAIC_COLS = 7    # 21 links -> 3 x 7 mosaic
MOSAIC_ROWS = 3


def build_taxel_pixel_map(layout: TaxelLayout) -> np.ndarray:
    """(T,) flat pixel index in the mosaic for every taxel. Deterministic:
    per-link PCA plane of the local taxel cloud -> unit square -> TILE grid."""
    W = MOSAIC_COLS * TILE
    out = np.zeros(layout.n_taxels, dtype=np.int64)
    for li in range(len(layout.link_names)):
        idx = np.flatnonzero(layout.link_index == li)
        pts = layout.positions[idx]
        centered = pts - pts.mean(0)
        # two principal axes of the taxel cloud = the link's 2D chart
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        uv = centered @ vt[:2].T
        uv = (uv - uv.min(0)) / np.maximum(uv.max(0) - uv.min(0), 1e-9)
        ij = np.clip((uv * (TILE - 1)).round().astype(np.int64), 0, TILE - 1)
        tile_r, tile_c = divmod(li, MOSAIC_COLS)
        rows = tile_r * TILE + ij[:, 1]
        cols = tile_c * TILE + ij[:, 0]
        out[idx] = rows * W + cols
    return out


class TactileImageEncoder(nn.Module):
    """ViT-style patch encoder over the pseudo-tactile mosaic."""

    def __init__(
        self,
        layout: TaxelLayout | None = None,
        dim: int = 256,
        n_layers: int = 6,
        heads: int = 8,
        patch: int = 4,
        node_out: int = 256,
        global_dim: int = 512,
    ):
        super().__init__()
        layout = layout or TaxelLayout.load()
        self.n_taxels = layout.n_taxels
        pixel_map = torch.as_tensor(build_taxel_pixel_map(layout))
        self.register_buffer("pixel_map", pixel_map, persistent=False)
        self.H, self.W = MOSAIC_ROWS * TILE, MOSAIC_COLS * TILE
        self.patch = patch
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        n_tokens = (self.H // patch) * (self.W // patch)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_tokens + 1, dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.cls, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 2,
            batch_first=True, norm_first=True, dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.node_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, node_out))
        self.global_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, global_dim))

        # taxel -> token index (which patch its pixel falls in)
        py, px = pixel_map // self.W, pixel_map % self.W
        token_map = (py // patch) * (self.W // patch) + (px // patch)
        self.register_buffer("token_map", token_map, persistent=False)

    def rasterize(self, force: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """(N,3) taxel readings + graph ids -> (B, 3, H, W) mosaics.

        Channels: f_normal, |shear|, occupancy. Multiple taxels landing on one
        pixel sum — the resolution loss inherent to image representations that
        H2 is about.

        occupancy is a per-taxel CONTACT indicator (1 if this taxel currently
        registers any force, else 0) — NOT "this pixel maps to a valid taxel
        slot" (that would be constant across every window, since the taxel
        layout is fixed geometry, and would carry zero information while
        dominating the image's magnitude next to the much smaller f_normal/
        shear values — this was a real bug found via Phase 6 collapse
        analysis: occupancy was `ones_like(f_n)`, a hardcoded constant image
        fed into every training step, ~350x larger in scale than the actual
        signal and completely swamping it in the unnormalized patch_embed).
        """
        B = int(batch.max().item()) + 1
        n_pix = self.H * self.W
        f_n = force[:, 0]
        shear = force[:, 1:].norm(dim=1)
        occ = ((f_n.abs() > 0) | (shear > 0)).to(force.dtype)
        pix = self.pixel_map.repeat(B)[: len(batch)] + batch * n_pix
        img_flat = force.new_zeros(B * n_pix, 3)
        img_flat.index_add_(0, pix, torch.stack([f_n, shear, occ], dim=1))
        return img_flat.view(B, n_pix, 3).permute(0, 2, 1).reshape(B, 3, self.H, self.W)

    def forward(self, force, link_index, edge_index, batch, pos, normal, qpos):
        """Same signature as TaxelGraphEncoder; ignores pos/normal/edges/qpos
        by design — that is the ablation."""
        # the fixed pixel/token maps assume every graph carries the full taxel
        # set in layout order (true for all shards produced by this repo)
        assert len(batch) % self.n_taxels == 0, "graphs must have the full taxel set"
        img = self.rasterize(force, batch)
        tokens = self.patch_embed(img).flatten(2).transpose(1, 2)  # (B, n_tok, dim)
        B = tokens.shape[0]
        x = torch.cat([self.cls.expand(B, -1, -1), tokens], dim=1) + self.pos_emb
        x = self.encoder(x)
        cls, patch_tokens = x[:, 0], x[:, 1:]
        # per-taxel latent = its patch token
        node_latent = self.node_head(patch_tokens.reshape(-1, patch_tokens.shape[-1]))
        n_tok = patch_tokens.shape[1]
        tok_idx = self.token_map.repeat(B) + torch.repeat_interleave(
            torch.arange(B, device=force.device) * n_tok, self.n_taxels
        )
        node_latent = node_latent[tok_idx]
        return node_latent, self.global_head(cls)
