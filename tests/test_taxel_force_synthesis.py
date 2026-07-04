"""Force-conservation and locality tests for taxel force synthesis (PRD §8)."""

import numpy as np
import pytest

from sim.forward_kinematics import axis_angle_to_matrix
from sim.taxel_force_synthesis import TaxelForceSynthesizer, tangent_basis
from sim.taxel_layout import TaxelLayout


@pytest.fixture
def layout():
    """Two links: a 4x4 grid patch and a small 3-taxel strip."""
    rng = np.random.default_rng(0)
    g = np.stack(np.meshgrid(np.linspace(0, 0.03, 4), np.linspace(0, 0.03, 4)), -1)
    grid = np.concatenate([g.reshape(-1, 2), np.zeros((16, 1))], axis=1)
    strip = np.array([[0.0, 0.0, 0.01], [0.01, 0.0, 0.01], [0.02, 0.0, 0.01]])
    positions = np.concatenate([grid, strip])
    normals = np.concatenate(
        [np.tile([0.0, 0.0, 1.0], (16, 1)), np.tile([0.0, 1.0, 0.0], (3, 1))]
    )
    # small random tilt so normals aren't axis-aligned edge cases everywhere
    tilt = axis_angle_to_matrix(rng.normal(size=3), 0.1)
    normals[:16] = normals[:16] @ tilt.T
    return TaxelLayout(
        link_names=["patch", "strip"],
        link_index=np.array([0] * 16 + [1] * 3),
        positions=positions,
        normals=normals,
        spacing=np.array([0.01, 0.01]),
    )


@pytest.fixture
def world():
    """Nontrivial world poses for the two links."""
    rot0 = axis_angle_to_matrix([0, 0, 1], 0.7)
    rot1 = axis_angle_to_matrix([1, 1, 0], -0.4)
    link_pos = np.array([[0.1, 0.0, 0.2], [-0.05, 0.3, 0.1]])
    link_rot = np.stack([rot0, rot1])
    return link_pos, link_rot


def test_tangent_basis_orthonormal():
    rng = np.random.default_rng(2)
    n = rng.normal(size=(50, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    t1, t2 = tangent_basis(n)
    for a, b in [(t1, t2), (t1, n), (t2, n)]:
        assert np.allclose((a * b).sum(1), 0, atol=1e-12)
    assert np.allclose(np.linalg.norm(t1, axis=1), 1, atol=1e-12)
    assert np.allclose(np.linalg.norm(t2, axis=1), 1, atol=1e-12)


def test_force_conservation(layout, world):
    link_pos, link_rot = world
    synth = TaxelForceSynthesizer(layout)
    rng = np.random.default_rng(3)
    # contacts near each link's surface, in world frame
    c_local = rng.uniform(0, 0.03, size=(8, 3)) * np.array([1, 1, 0.1])
    c_link = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    c_pos = np.stack(
        [link_rot[li] @ c_local[i] + link_pos[li] for i, li in enumerate(c_link)]
    )
    c_frc = rng.normal(scale=2.0, size=(8, 3))

    out = synth.synthesize(c_pos, c_frc, c_link, link_pos, link_rot)

    # total force conserved per link (compare in world frame)
    for li in (0, 1):
        taxels = layout.link_index == li
        got_world = link_rot[li] @ out.force[taxels].sum(axis=0)
        want_world = c_frc[c_link == li].sum(axis=0)
        assert np.allclose(got_world, want_world, atol=1e-10)

    # normal + shear decomposition reconstructs the full vector magnitude
    recon = out.f_normal**2 + (out.f_shear**2).sum(1)
    assert np.allclose(recon, (out.force**2).sum(1), atol=1e-10)


def test_locality_nearest_taxel_dominates(layout, world):
    link_pos, link_rot = world
    synth = TaxelForceSynthesizer(layout)
    # contact exactly at taxel 5 of the patch
    target = 5
    c_pos = (link_rot[0] @ layout.positions[target] + link_pos[0])[None]
    c_frc = np.array([[0.0, 0.0, -1.0]])
    out = synth.synthesize(c_pos, c_frc, np.array([0]), link_pos, link_rot)
    mags = out.magnitude
    assert mags.argmax() == target
    # strip link untouched
    assert np.allclose(mags[16:], 0)


def test_far_contact_degenerate_case(layout, world):
    """A contact absurdly far from all taxels must still conserve force."""
    link_pos, link_rot = world
    synth = TaxelForceSynthesizer(layout)
    c_pos = (link_rot[0] @ np.array([5.0, 5.0, 5.0]) + link_pos[0])[None]
    c_frc = np.array([[1.0, 2.0, 3.0]])
    out = synth.synthesize(c_pos, c_frc, np.array([0]), link_pos, link_rot)
    got_world = link_rot[0] @ out.force[layout.link_index == 0].sum(axis=0)
    assert np.allclose(got_world, c_frc[0], atol=1e-10)


def test_no_contacts(layout, world):
    link_pos, link_rot = world
    synth = TaxelForceSynthesizer(layout)
    out = synth.synthesize(
        np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, dtype=int), link_pos, link_rot
    )
    assert np.allclose(out.force, 0)
