"""Graph construction: edge counts, symmetry, cross-finger adjacency (PRD §5.5, §8)."""

import numpy as np
import pytest

from data.graph_construction import (
    GraphBuilder,
    build_radius_graph,
    build_static_backbone,
)
from sim.taxel_layout import DEFAULT_LAYOUT_PATH, TaxelLayout


@pytest.fixture(scope="module")
def layout():
    return TaxelLayout.load(DEFAULT_LAYOUT_PATH)


def _is_symmetric(e: np.ndarray) -> bool:
    fwd = set(map(tuple, e.T))
    return all((b, a) in fwd for a, b in fwd)


def test_backbone_intra_link_only(layout):
    e = build_static_backbone(layout, k=6)
    assert (e[0] != e[1]).all()
    assert _is_symmetric(e)
    # every edge stays within one link
    assert (layout.link_index[e[0]] == layout.link_index[e[1]]).all()
    # every taxel participates
    assert np.unique(e).size == layout.n_taxels


def test_radius_graph_basic():
    # two clusters 1 m apart: no cross-cluster edges at r=1 cm
    rng = np.random.default_rng(0)
    a = rng.normal(scale=0.003, size=(20, 3))
    b = a + np.array([1.0, 0, 0])
    e = build_radius_graph(np.concatenate([a, b]), radius=0.01, k_max=16)
    assert e.size > 0
    assert _is_symmetric(e)
    same_side = (e[0] < 20) == (e[1] < 20)
    assert same_side.all()


def test_radius_graph_k_cap():
    # 50 coincident-ish points: degree must be capped at k_max
    rng = np.random.default_rng(1)
    pts = rng.normal(scale=1e-4, size=(50, 3))
    e = build_radius_graph(pts, radius=0.01, k_max=8)
    # the compute bound that matters: total edges <= 2 * n * k_max
    # (each node contributes <= k_max outgoing edges, then symmetrized)
    assert e.shape[1] <= 2 * 50 * 8


def test_full_graph_and_cross_finger_adjacency(layout):
    builder = GraphBuilder(layout)
    L = len(layout.link_names)
    T = layout.n_taxels

    # spread all links far apart -> only backbone + intra-link radius edges
    link_pos = np.arange(L)[:, None] * np.array([1.0, 0, 0])
    link_rot = np.tile(np.eye(3), (L, 1, 1))
    qpos = np.zeros(22, dtype=np.float32)
    g = builder.build(
        link_pos, link_rot, np.zeros(T), np.zeros((T, 2)), qpos, qpos
    )
    assert g.pos.shape == (T, 3)
    assert g.normal.shape == (T, 3)
    assert g.force.shape == (T, 3)
    cross = layout.link_index[g.edge_index[0]] != layout.link_index[g.edge_index[1]]
    assert not cross.any()

    # overlap two fingertip links -> cross-link radius edges appear
    la = layout.link_names.index("link_3.0_tip")
    lb = layout.link_names.index("link_7.0_tip")
    link_pos[lb] = link_pos[la]
    g2 = builder.build(
        link_pos, link_rot, np.zeros(T), np.zeros((T, 2)), qpos, qpos
    )
    ia, ib = layout.link_index[g2.edge_index[0]], layout.link_index[g2.edge_index[1]]
    cross_ab = ((ia == la) & (ib == lb)) | ((ia == lb) & (ib == la))
    assert cross_ab.any(), "overlapping fingertips must become graph-adjacent"


def test_labels(layout):
    builder = GraphBuilder(layout, contact_force_threshold=0.5)
    L, T = len(layout.link_names), layout.n_taxels
    link_pos = np.arange(L)[:, None] * np.array([1.0, 0, 0])
    link_rot = np.tile(np.eye(3), (L, 1, 1))
    f_normal = np.zeros(T)
    f_normal[:10] = 2.0  # 10 taxels in firm contact
    qpos = np.zeros(22, dtype=np.float32)
    slip = np.zeros(T)
    slip[3] = 1.0
    g = builder.build(link_pos, link_rot, f_normal, np.zeros((T, 2)), qpos, qpos, slip)
    assert g.contact_area == 10.0
    assert np.allclose(g.force_mag[:10], 2.0)
    assert g.slip[3] == 1.0 and g.slip.sum() == 1.0
