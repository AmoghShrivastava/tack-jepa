"""Procedural superquadric mesh generation (PRD §6.1 Stage C: "procedural
superquadrics" alongside multi-object mesh diversity).

Superquadrics (superellipsoids) are a parametric family that smoothly spans
spheres, boxes, cylinders, pillows, and star-like shapes via two shape
exponents (e1, e2) plus three axis scales — giving a much larger and smoother
object-shape distribution than a handful of hand-picked primitives, without
needing any external assets.

    python -m sim.superquadric --out assets/superquadrics --n 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
DEFAULT_SUPERQUADRIC_DIR = ASSETS_DIR / "superquadrics"


def _fexp(x: np.ndarray, e: float) -> np.ndarray:
    """signed power: sign(x) * |x|^e, the standard superquadric building block."""
    return np.sign(x) * np.abs(x) ** e


def superquadric_mesh(
    e1: float, e2: float, scale: tuple[float, float, float], n_u: int = 24, n_v: int = 48
) -> trimesh.Trimesh:
    """Parametric superellipsoid surface, triangulated on a (u,v) grid.

    e1 controls the north-south (pole-to-pole) profile, e2 the equatorial
    cross-section; e=1 gives an ellipsoid, e<1 pinches toward a box/star, e>1
    rounds toward a pillow. scale = (a1, a2, a3) semi-axis lengths (meters).

    Explicitly closes both poles to single vertices and welds the v-seam
    (rather than sampling u in [-pi/2, pi/2] and v in [-pi, pi] as a plain
    grid, which leaves the poles as degenerate near-zero-area rings and the
    seam unstitched) — a real Phase 6/Stage C bug: that degenerate collision
    geometry caused a NaN acceleration crash in Genesis's rigid solver on one
    of these objects mid-generation.
    """
    a1, a2, a3 = scale
    u = np.linspace(-np.pi / 2, np.pi / 2, n_u)[1:-1]  # interior rings only
    v = np.linspace(-np.pi, np.pi, n_v, endpoint=False)  # distinct, wraps via modulo
    uu, vv = np.meshgrid(u, v, indexing="ij")
    cu, su = np.cos(uu), np.sin(uu)
    cv, sv = np.cos(vv), np.sin(vv)
    x = a1 * _fexp(cu, e1) * _fexp(cv, e2)
    y = a2 * _fexp(cu, e1) * _fexp(sv, e2)
    z = a3 * _fexp(su, e1)
    ring_verts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    n_rings = len(u)

    verts = np.concatenate(
        [np.array([[0.0, 0.0, -a3]]), ring_verts, np.array([[0.0, 0.0, a3]])], axis=0
    )
    bot_idx, top_idx = 0, len(verts) - 1

    def ring_idx(row: int, col: int) -> int:
        return 1 + row * n_v + (col % n_v)

    faces = []
    for j in range(n_v):
        faces.append([bot_idx, ring_idx(0, j + 1), ring_idx(0, j)])
    for i in range(n_rings - 1):
        for j in range(n_v):
            a, b = ring_idx(i, j), ring_idx(i, j + 1)
            c, d = ring_idx(i + 1, j), ring_idx(i + 1, j + 1)
            faces.append([a, b, d])
            faces.append([a, d, c])
    for j in range(n_v):
        faces.append([top_idx, ring_idx(n_rings - 1, j), ring_idx(n_rings - 1, j + 1)])

    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
    mesh.merge_vertices()
    mesh.fix_normals()
    return mesh


# (e1, e2, (a1,a2,a3)) — hand-graspable scale (~0.02-0.06m semi-axes), spanning
# box-like (low e), ellipsoid (e=1), and pillow/star-like (high e) shapes,
# with varying elongation per axis for genuine multi-object diversity.
VARIANTS: list[tuple[float, float, tuple[float, float, float]]] = [
    (0.2, 0.2, (0.03, 0.03, 0.03)),   # rounded cube
    (0.3, 1.0, (0.035, 0.025, 0.03)),  # rounded box, one axis squashed
    (1.0, 1.0, (0.03, 0.022, 0.026)),  # ellipsoid
    (1.6, 1.0, (0.028, 0.028, 0.038)),  # pillow, elongated on z
    (0.5, 1.8, (0.032, 0.032, 0.024)),  # star-cross-section disc
    (0.8, 0.4, (0.024, 0.024, 0.045)),  # tapered capsule-ish
]


def build_superquadric_variants(out_dir: Path = DEFAULT_SUPERQUADRIC_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (e1, e2, scale) in enumerate(VARIANTS):
        mesh = superquadric_mesh(e1, e2, scale)
        p = out_dir / f"superquadric_{i:02d}.obj"
        mesh.export(p)
        paths.append(p)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_SUPERQUADRIC_DIR)
    args = parser.parse_args()
    paths = build_superquadric_variants(args.out)
    for p in paths:
        m = trimesh.load(p, force="mesh")
        print(f"{p.name}: verts={len(m.vertices)} faces={len(m.faces)} "
              f"watertight={m.is_watertight} extents={m.bounding_box.extents}")


if __name__ == "__main__":
    main()
