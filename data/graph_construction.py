"""Taxel graph construction (PRD §5.5).

Nodes: one per taxel. Edges: union of
  (a) a radius graph recomputed per timestep over FK-updated WORLD-frame taxel
      positions (radius ~1 cm, capped at k nearest) — this is what makes
      cross-finger contact visible: taxels on different fingers become
      adjacent exactly when physically close; and
  (b) a static intra-link kNN backbone (fixed at build time, local frame) so
      message passing within a link survives sparse radius edges.

Node features stored per graph: world position (3), world normal (3), signed
normal force (1), 2D shear (2), link index (embedded model-side). The raw
22-dim action/joint state rides along globally — ablations choose what to use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from sim.taxel_layout import TaxelLayout

RADIUS = 0.01  # 1 cm, PRD §5.5
K_MAX = 16


def build_static_backbone(layout: TaxelLayout, k: int = 6) -> np.ndarray:
    """(2, E) undirected intra-link kNN edges over LOCAL-frame taxel positions."""
    edges = []
    for li in range(len(layout.link_names)):
        idx = np.flatnonzero(layout.link_index == li)
        pts = layout.positions[idx]
        kk = min(k + 1, len(idx))
        _, nbr = cKDTree(pts).query(pts, k=kk)
        src = np.repeat(idx, kk - 1)
        dst = idx[nbr[:, 1:]].reshape(-1)  # drop self (column 0)
        edges.append(np.stack([src, dst]))
    e = np.concatenate(edges, axis=1)
    return _undirected_unique(e)


def build_radius_graph(
    world_pos: np.ndarray, radius: float = RADIUS, k_max: int = K_MAX
) -> np.ndarray:
    """(2, E) undirected edges between taxels within `radius`, capped at k_max
    nearest neighbors per node."""
    tree = cKDTree(world_pos)
    dist, nbr = tree.query(world_pos, k=k_max + 1, distance_upper_bound=radius)
    n = world_pos.shape[0]
    src = np.repeat(np.arange(n), k_max)
    dst = nbr[:, 1:].reshape(-1)
    ok = dst < n  # cKDTree pads missing neighbors with n
    e = np.stack([src[ok], dst[ok]])
    return _undirected_unique(e)


def _undirected_unique(e: np.ndarray) -> np.ndarray:
    """Symmetrize and dedupe an edge list; no self-loops."""
    e = e[:, e[0] != e[1]]
    both = np.concatenate([e, e[::-1]], axis=1)
    key = both[0].astype(np.int64) * (both.max() + 1) + both[1]
    _, keep = np.unique(key, return_index=True)
    return both[:, keep]


@dataclass
class TaxelGraph:
    """One timestep's graph sample (numpy; converted to PyG Data at load time)."""

    pos: np.ndarray          # (T, 3) world-frame taxel positions (FK output)
    normal: np.ndarray       # (T, 3) world-frame taxel normals
    force: np.ndarray        # (T, 3) [f_normal, f_shear_1, f_shear_2]
    link_index: np.ndarray   # (T,) int
    edge_index: np.ndarray   # (2, E) radius ∪ backbone
    qpos: np.ndarray         # (22,) joint state: 16 fingers + 6 wrist pose
    action: np.ndarray       # (22,) action for this step (target deltas)

    # per-taxel ground-truth labels for physics probes (§5.9)
    force_mag: np.ndarray    # (T,) ground-truth |force|
    slip: np.ndarray         # (T,) binary slip indicator
    contact_area: float      # scalar: count of taxels above force threshold


class GraphBuilder:
    def __init__(
        self,
        layout: TaxelLayout,
        radius: float = RADIUS,
        k_max: int = K_MAX,
        backbone_k: int = 6,
        contact_force_threshold: float = 0.05,
    ):
        self.layout = layout
        self.radius = radius
        self.k_max = k_max
        self.contact_force_threshold = contact_force_threshold
        self.backbone = build_static_backbone(layout, k=backbone_k)

    def build(
        self,
        link_pos: np.ndarray,     # (L, 3) world link positions (layout order)
        link_rot: np.ndarray,     # (L, 3, 3)
        f_normal: np.ndarray,     # (T,)
        f_shear: np.ndarray,      # (T, 2)
        qpos: np.ndarray,
        action: np.ndarray,
        slip: np.ndarray | None = None,
    ) -> TaxelGraph:
        li = self.layout.link_index
        world_pos = (
            np.einsum("tij,tj->ti", link_rot[li], self.layout.positions)
            + link_pos[li]
        )
        world_normal = np.einsum("tij,tj->ti", link_rot[li], self.layout.normals)

        radius_edges = build_radius_graph(world_pos, self.radius, self.k_max)
        edge_index = _undirected_unique(
            np.concatenate([radius_edges, self.backbone], axis=1)
        )

        force = np.concatenate([f_normal[:, None], f_shear], axis=1)
        force_mag = np.sqrt(f_normal**2 + (f_shear**2).sum(1))
        return TaxelGraph(
            pos=world_pos.astype(np.float32),
            normal=world_normal.astype(np.float32),
            force=force.astype(np.float32),
            link_index=li.astype(np.int64),
            edge_index=edge_index.astype(np.int64),
            qpos=np.asarray(qpos, dtype=np.float32),
            action=np.asarray(action, dtype=np.float32),
            force_mag=force_mag.astype(np.float32),
            slip=(
                slip.astype(np.float32)
                if slip is not None
                else np.zeros(len(world_pos), dtype=np.float32)
            ),
            contact_area=float((force_mag > self.contact_force_threshold).sum()),
        )
