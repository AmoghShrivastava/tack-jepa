"""The committed taxel layout artifact is a valid, in-spec sensor spec (PRD §5.2)."""

import numpy as np
import pytest

from sim.taxel_layout import DEFAULT_LAYOUT_PATH, TaxelLayout, farthest_point_sampling


@pytest.fixture(scope="module")
def layout():
    assert DEFAULT_LAYOUT_PATH.is_file(), "run `python -m sim.taxel_layout` to generate"
    return TaxelLayout.load()


def test_total_in_prd_band(layout):
    assert 2000 <= layout.n_taxels <= 2500


def test_per_link_counts_in_band(layout):
    counts = np.bincount(layout.link_index, minlength=len(layout.link_names))
    assert (counts >= 96).all() and (counts <= 160).all()
    assert len(layout.link_names) == 21  # every geometric link carries taxels


def test_geometry_sane(layout):
    assert np.isfinite(layout.positions).all()
    # local-frame link coordinates are all within 20 cm of the link origin
    assert np.linalg.norm(layout.positions, axis=1).max() < 0.2
    assert np.allclose(np.linalg.norm(layout.normals, axis=1), 1.0, atol=1e-8)
    assert (layout.spacing > 0).all() and (layout.spacing < 0.02).all()


def test_fps_spreads_points():
    rng = np.random.default_rng(0)
    pts = rng.uniform(size=(500, 3))
    idx = farthest_point_sampling(pts, 32, seed=0)
    assert len(np.unique(idx)) == 32
    sel = pts[idx]
    d = np.linalg.norm(sel[:, None] - sel[None, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    # FPS min pairwise distance beats random selection's, decisively
    rand = pts[rng.choice(500, 32, replace=False)]
    dr = np.linalg.norm(rand[:, None] - rand[None, :], axis=-1)
    np.fill_diagonal(dr, np.inf)
    assert d.min() > dr.min()
