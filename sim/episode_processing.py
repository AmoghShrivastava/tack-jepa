"""Turn raw episode dumps into per-taxel force fields (bridges §5.3 and §5.5).

An episode dump (sim/hand_env.py) stores contacts with GLOBAL solver link
indices and both force sides; here we extract hand-side contacts, map them to
taxel-layout link order, and run the deterministic force synthesis per step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sim.forward_kinematics import quat_to_matrix
from sim.taxel_force_synthesis import TaxelForceSynthesizer
from sim.taxel_layout import TaxelLayout


def load_episode(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def _layout_order_maps(dump: dict, layout: TaxelLayout) -> tuple[np.ndarray, dict[int, int]]:
    """(entity->layout link order permutation, global solver idx -> layout idx)."""
    entity_names = [str(s) for s in dump["link_names"]]
    perm = np.array([entity_names.index(ln) for ln in layout.link_names])
    global_idx = dump["link_global_idx"]
    global_to_layout = {
        int(global_idx[entity_names.index(ln)]): li
        for li, ln in enumerate(layout.link_names)
    }
    return perm, global_to_layout


def link_poses_layout_order(
    dump: dict, layout: TaxelLayout
) -> tuple[np.ndarray, np.ndarray]:
    """Per-step link poses reordered to layout link order.

    Returns (S, L, 3) positions and (S, L, 3, 3) rotations.
    """
    perm, _ = _layout_order_maps(dump, layout)
    pos = dump["link_pos"][:, perm]
    quat = dump["link_quat"][:, perm]
    S, L = quat.shape[:2]
    rot = np.empty((S, L, 3, 3))
    for s in range(S):
        for li in range(L):
            rot[s, li] = quat_to_matrix(quat[s, li])
    return pos, rot


def hand_contacts_at_step(
    dump: dict, step: int, global_to_layout: dict[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contacts touching the hand at `step`: (pos (C,3), force-on-hand (C,3),
    layout link idx (C,)). Contacts with both sides on the hand (self-contact)
    contribute one record per side."""
    sel = np.flatnonzero(dump["contact_step"] == step)
    pos_out, frc_out, link_out = [], [], []
    for i in sel:
        p = dump["contact_position"][i]
        for side in ("a", "b"):
            gl = int(dump[f"contact_link_{side}"][i])
            if gl in global_to_layout:
                pos_out.append(p)
                frc_out.append(dump[f"contact_force_{side}"][i])
                link_out.append(global_to_layout[gl])
    if not pos_out:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    return np.stack(pos_out), np.stack(frc_out), np.array(link_out, dtype=np.int64)


@dataclass
class EpisodeTaxelData:
    """Synthesized taxel readings for a whole episode."""

    force: np.ndarray      # (S, T, 3) local-frame force vectors
    f_normal: np.ndarray   # (S, T)
    f_shear: np.ndarray    # (S, T, 2)
    link_pos: np.ndarray   # (S, L, 3) layout order
    link_rot: np.ndarray   # (S, L, 3, 3)

    @property
    def magnitude(self) -> np.ndarray:  # (S, T)
        return np.linalg.norm(self.force, axis=-1)


def synthesize_episode(
    dump: dict, layout: TaxelLayout, synthesizer: TaxelForceSynthesizer | None = None
) -> EpisodeTaxelData:
    synthesizer = synthesizer or TaxelForceSynthesizer(layout)
    _, global_to_layout = _layout_order_maps(dump, layout)
    link_pos, link_rot = link_poses_layout_order(dump, layout)
    S = dump["qpos"].shape[0]
    T = layout.n_taxels
    force = np.zeros((S, T, 3))
    f_normal = np.zeros((S, T))
    f_shear = np.zeros((S, T, 2))
    for s in range(S):
        c_pos, c_frc, c_link = hand_contacts_at_step(dump, s, global_to_layout)
        if len(c_pos) == 0:
            continue
        out = synthesizer.synthesize(c_pos, c_frc, c_link, link_pos[s], link_rot[s])
        force[s] = out.force
        f_normal[s] = out.f_normal
        f_shear[s] = out.f_shear
    return EpisodeTaxelData(
        force=force, f_normal=f_normal, f_shear=f_shear,
        link_pos=link_pos, link_rot=link_rot,
    )


def taxel_world_positions_at_step(
    layout: TaxelLayout, data: EpisodeTaxelData, step: int
) -> np.ndarray:
    """(T, 3) world positions of every taxel at `step` (FK-exact)."""
    li = layout.link_index
    return (
        np.einsum("tij,tj->ti", data.link_rot[step][li], layout.positions)
        + data.link_pos[step][li]
    )
