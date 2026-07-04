"""Fixed taxel layout: farthest-point sampling over each link's surface (PRD §5.2).

The layout is a build-time artifact (assets/taxel_layout.npz), generated once,
version-controlled, and never regenerated per run — the sensor spec is fixed,
exactly like real hardware. Taxel positions/normals are stored in each link's
LOCAL frame; world-frame positions come from FK at runtime.

Sampling surface: the links' *visual* meshes (the actual skin surface), not the
primitive collision geometry. Per-link taxel counts are proportional to surface
area, clipped to [min_per_link, max_per_link] per the PRD's 96-160 guidance,
targeting ~2,000-2,500 taxels hand-wide (ART-Glove scale: 2048).

Run as a script to (re)generate the artifact:

    python -m sim.taxel_layout
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sim.forward_kinematics import rpy_to_matrix
from sim.hand_env import ALLEGRO_URDF, ASSETS_DIR

DEFAULT_LAYOUT_PATH = ASSETS_DIR / "taxel_layout.npz"


def parse_visual_meshes(urdf_path: Path) -> dict[str, list[tuple[Path, np.ndarray, np.ndarray]]]:
    """link name -> [(mesh path, origin_pos, origin_rot 3x3)], links with mesh visuals only."""
    root = ET.parse(urdf_path).getroot()
    out: dict[str, list] = {}
    for link in root.findall("link"):
        entries = []
        for vis in link.findall("visual"):
            mesh = vis.find("geometry/mesh")
            if mesh is None:
                continue
            pos = np.zeros(3)
            rot = np.eye(3)
            origin = vis.find("origin")
            if origin is not None:
                if "xyz" in origin.attrib:
                    pos = np.array([float(v) for v in origin.attrib["xyz"].split()])
                if "rpy" in origin.attrib:
                    rot = rpy_to_matrix([float(v) for v in origin.attrib["rpy"].split()])
            entries.append((urdf_path.parent / mesh.attrib["filename"], pos, rot))
        if entries:
            out[link.attrib["name"]] = entries
    return out


def farthest_point_sampling(points: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Greedy FPS: indices of k points maximizing mutual minimum distance."""
    n = points.shape[0]
    if k >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    chosen = np.empty(k, dtype=np.int64)
    chosen[0] = rng.integers(n)
    dist = np.linalg.norm(points - points[chosen[0]], axis=1)
    for i in range(1, k):
        chosen[i] = int(np.argmax(dist))
        dist = np.minimum(dist, np.linalg.norm(points - points[chosen[i]], axis=1))
    return chosen


@dataclass
class TaxelLayout:
    """Fixed sensor spec: per-taxel local-frame position/normal + owning link."""

    link_names: list[str]          # links that carry taxels, in order
    link_index: np.ndarray         # (T,) int — index into link_names per taxel
    positions: np.ndarray          # (T, 3) local frame
    normals: np.ndarray            # (T, 3) local frame, unit
    spacing: np.ndarray            # (L,) mean nearest-neighbor spacing per link

    @property
    def n_taxels(self) -> int:
        return self.positions.shape[0]

    def per_link(self, link: str) -> np.ndarray:
        return self.positions[self.link_index == self.link_names.index(link)]

    def local_points_by_link(self) -> dict[str, np.ndarray]:
        """For KinematicChain.taxel_world_positions."""
        return {ln: self.per_link(ln) for ln in self.link_names}

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            link_names=np.array(self.link_names),
            link_index=self.link_index,
            positions=self.positions,
            normals=self.normals,
            spacing=self.spacing,
        )

    @classmethod
    def load(cls, path: Path = DEFAULT_LAYOUT_PATH) -> TaxelLayout:
        d = np.load(path)
        return cls(
            link_names=[str(s) for s in d["link_names"]],
            link_index=d["link_index"],
            positions=d["positions"],
            normals=d["normals"],
            spacing=d["spacing"],
        )


def generate_layout(
    urdf_path: Path = ALLEGRO_URDF,
    budget: int = 2400,
    min_per_link: int = 96,
    max_per_link: int = 160,
    n_candidates: int = 4096,
    seed: int = 0,
) -> TaxelLayout:
    import trimesh

    visuals = parse_visual_meshes(urdf_path)
    # Load + transform each link's mesh into the link local frame
    meshes: dict[str, trimesh.Trimesh] = {}
    for link, entries in visuals.items():
        parts = []
        for mesh_path, pos, rot in entries:
            m = trimesh.load(mesh_path, force="mesh")
            tf = np.eye(4)
            tf[:3, :3] = rot
            tf[:3, 3] = pos
            m.apply_transform(tf)
            parts.append(m)
        meshes[link] = parts[0] if len(parts) == 1 else trimesh.util.concatenate(parts)

    areas = {link: m.area for link, m in meshes.items()}
    total_area = sum(areas.values())
    # area-proportional allocation, clipped to the PRD's per-link band
    counts = {
        link: int(np.clip(round(budget * areas[link] / total_area), min_per_link, max_per_link))
        for link in meshes
    }

    link_names = sorted(meshes)  # deterministic order
    all_pos, all_nrm, all_idx, spacing = [], [], [], []
    for li, link in enumerate(link_names):
        m = meshes[link]
        pts, face_idx = trimesh.sample.sample_surface(
            m, n_candidates, seed=seed + li
        )
        keep = farthest_point_sampling(np.asarray(pts), counts[link], seed=seed + li)
        pos = np.asarray(pts)[keep]
        nrm = np.asarray(m.face_normals)[np.asarray(face_idx)[keep]]
        nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
        all_pos.append(pos)
        all_nrm.append(nrm)
        all_idx.append(np.full(len(keep), li, dtype=np.int64))
        # mean nearest-neighbor distance -> kernel bandwidth basis (PRD §5.3)
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        spacing.append(d.min(axis=1).mean())

    return TaxelLayout(
        link_names=link_names,
        link_index=np.concatenate(all_idx),
        positions=np.concatenate(all_pos).astype(np.float64),
        normals=np.concatenate(all_nrm).astype(np.float64),
        spacing=np.array(spacing),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_LAYOUT_PATH)
    parser.add_argument("--budget", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    layout = generate_layout(budget=args.budget, seed=args.seed)
    layout.save(args.out)
    print(f"links with taxels: {len(layout.link_names)}")
    for li, link in enumerate(layout.link_names):
        n = int((layout.link_index == li).sum())
        print(f"  {link:20s} {n:4d} taxels, spacing {layout.spacing[li] * 1000:.2f} mm")
    print(f"total taxels: {layout.n_taxels}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
