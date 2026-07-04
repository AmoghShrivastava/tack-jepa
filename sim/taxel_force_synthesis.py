"""Taxel force synthesis: distribute contact-solver forces onto taxels (PRD §5.3).

Deterministic, not learned. For each solver contact point on a hand link:
transform it into the link's local frame, spread its force vector over that
link's taxels with a Gaussian kernel over local distance (bandwidth =
`bandwidth_scale` x that link's mean taxel spacing), weights normalized so
total force is conserved. Per taxel we record the full local-frame force
vector plus its decomposition into a signed normal component and a 2D shear
vector in a fixed per-taxel tangent basis (shear is what carries slip).

Modeling assumption (PRD §5.3/§11): this kernel stands in for the mechanical
coupling of a real taxel array's substrate; a compliant soft-body version is
an explicitly-flagged v2 item (§5.4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.taxel_layout import TaxelLayout


def tangent_basis(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic orthonormal (t1, t2) per unit normal. (T,3) -> (T,3),(T,3)."""
    n = normals
    # pick the reference axis least aligned with n, per row
    ref = np.tile(np.array([1.0, 0.0, 0.0]), (len(n), 1))
    ref[np.abs(n[:, 0]) > 0.9] = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, ref)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(n, t1)  # already unit
    return t1, t2


@dataclass
class TaxelForces:
    """Per-taxel readings for one timestep, all in link-local frames."""

    force: np.ndarray     # (T, 3) full distributed force vector
    f_normal: np.ndarray  # (T,) signed component along the taxel outward normal
    f_shear: np.ndarray   # (T, 2) tangential components in the taxel's basis

    @property
    def magnitude(self) -> np.ndarray:
        return np.linalg.norm(self.force, axis=1)


class TaxelForceSynthesizer:
    def __init__(self, layout: TaxelLayout, bandwidth_scale: float = 1.5):
        self.layout = layout
        self.bandwidth_scale = bandwidth_scale
        # per-link views, precomputed once
        self._taxel_idx = [
            np.flatnonzero(layout.link_index == li) for li in range(len(layout.link_names))
        ]
        self._sigma = bandwidth_scale * layout.spacing  # (L,)
        self._t1, self._t2 = tangent_basis(layout.normals)

    def synthesize(
        self,
        contact_pos_world: np.ndarray,   # (C, 3)
        contact_force_world: np.ndarray,  # (C, 3) force applied to the hand
        contact_link: np.ndarray,         # (C,) index into layout.link_names
        link_pos: np.ndarray,             # (L, 3) world position per layout link
        link_rot: np.ndarray,             # (L, 3, 3) world rotation per layout link
    ) -> TaxelForces:
        T = self.layout.n_taxels
        force = np.zeros((T, 3))
        for li in np.unique(contact_link):
            li = int(li)
            tidx = self._taxel_idx[li]
            if tidx.size == 0:
                continue
            mask = contact_link == li
            R, p = link_rot[li], link_pos[li]
            # world -> link local
            c_pos = (contact_pos_world[mask] - p) @ R          # (Ci, 3)
            c_frc = contact_force_world[mask] @ R              # (Ci, 3)
            taxels = self.layout.positions[tidx]               # (Ti, 3)
            d2 = ((taxels[:, None, :] - c_pos[None, :, :]) ** 2).sum(-1)  # (Ti, Ci)
            w = np.exp(-d2 / (2.0 * self._sigma[li] ** 2))
            w_sum = w.sum(axis=0, keepdims=True)
            w = np.divide(w, w_sum, out=np.zeros_like(w), where=w_sum > 0)
            # degenerate case: contact so far from every taxel that all weights
            # underflow to 0 — dump the force on the nearest taxel instead of
            # silently losing it (conservation is a tested invariant)
            dead = np.flatnonzero(w_sum[0] == 0)
            for ci in dead:
                w[np.argmin(d2[:, ci]), ci] = 1.0
            force[tidx] += w @ c_frc
        f_normal = (force * self.layout.normals).sum(axis=1)
        f_shear = np.stack(
            [(force * self._t1).sum(axis=1), (force * self._t2).sum(axis=1)], axis=1
        )
        return TaxelForces(force=force, f_normal=f_normal, f_shear=f_shear)
