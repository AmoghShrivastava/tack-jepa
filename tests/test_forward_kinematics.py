"""FK correctness against hand-derived poses (PRD §8 testing row, Phase 2).

Pure numpy — runs everywhere including CI. Cross-validation against Genesis's
link poses lives in test_fk_vs_genesis.py (skipped where genesis is absent).
"""

import numpy as np
import pytest

from sim.forward_kinematics import (
    KinematicChain,
    axis_angle_to_matrix,
    matrix_to_quat,
    quat_to_matrix,
    rpy_to_matrix,
)
from sim.hand_env import ALLEGRO_URDF

TWO_LINK_URDF = """<?xml version="1.0"?>
<robot name="two_link">
  <link name="base"/>
  <link name="arm"/>
  <link name="tip"/>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="arm"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit effort="1" lower="-3.14" upper="3.14" velocity="1"/>
  </joint>
  <joint name="wrist" type="fixed">
    <parent link="arm"/><child link="tip"/>
    <origin xyz="0.2 0 0"/>
  </joint>
</robot>
"""


@pytest.fixture
def two_link(tmp_path):
    p = tmp_path / "two_link.urdf"
    p.write_text(TWO_LINK_URDF)
    return KinematicChain.from_urdf(p)


def test_rotation_utils_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        axis = rng.normal(size=3)
        angle = rng.uniform(-np.pi, np.pi)
        R = axis_angle_to_matrix(axis, angle)
        # proper rotation
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)
        # quat roundtrip
        assert np.allclose(quat_to_matrix(matrix_to_quat(R)), R, atol=1e-12)


def test_rpy_convention():
    # yaw of pi/2 sends +x to +y
    R = rpy_to_matrix([0, 0, np.pi / 2])
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    # roll of pi/2 sends +y to +z
    R = rpy_to_matrix([np.pi / 2, 0, 0])
    assert np.allclose(R @ [0, 1, 0], [0, 0, 1], atol=1e-12)


def test_two_link_hand_computed(two_link):
    # q=0: tip at (0.2, 0, 0.1)
    poses = two_link.fk({"shoulder": 0.0})
    assert np.allclose(poses["tip"].pos, [0.2, 0.0, 0.1], atol=1e-12)
    # q=pi/2 about z: tip swings to (0, 0.2, 0.1)
    poses = two_link.fk({"shoulder": np.pi / 2})
    assert np.allclose(poses["tip"].pos, [0.0, 0.2, 0.1], atol=1e-12)
    # base offset + base yaw of pi: everything flips in x/y
    yaw_pi = matrix_to_quat(rpy_to_matrix([0, 0, np.pi]))
    poses = two_link.fk({"shoulder": 0.0}, base_pos=(1, 0, 0), base_quat=yaw_pi)
    assert np.allclose(poses["tip"].pos, [0.8, 0.0, 0.1], atol=1e-12)


def test_allegro_chain_structure():
    chain = KinematicChain.from_urdf(ALLEGRO_URDF)
    assert chain.root_link == "base_link"
    assert len(chain.actuated_joint_names) == 16
    assert chain.actuated_joint_names[0] == "joint_0.0"
    # 23 links: base + palm + wrist + 16 finger links + 4 tips
    assert len(chain.link_names) == 23
    poses = chain.fk(np.zeros(16))
    assert set(poses) == set(chain.link_names)
    for lp in poses.values():
        assert np.isfinite(lp.pos).all()
        assert np.allclose(lp.rot @ lp.rot.T, np.eye(3), atol=1e-10)
    # at zero pose, the three fingers extend +z from the base...
    for tip in ("link_3.0_tip", "link_7.0_tip", "link_11.0_tip"):
        assert poses[tip].pos[2] > 0.05
    # ...while the thumb mounts sideways/below the palm plane (rpy 0,-1.66,-1.57)
    thumb_tip = poses["link_15.0_tip"].pos
    assert thumb_tip[2] < 0.0
    assert np.linalg.norm(thumb_tip) > 0.08


def test_taxel_world_positions_rigid(two_link):
    pts = {"tip": np.array([[0.0, 0.0, 0.0], [0.01, 0.02, 0.03]])}
    rng = np.random.default_rng(1)
    d0 = None
    for _ in range(5):
        q = rng.uniform(-np.pi, np.pi)
        world = two_link.taxel_world_positions(pts, {"shoulder": q})["tip"]
        d = np.linalg.norm(world[0] - world[1])
        d0 = d if d0 is None else d0
        # rigid transform preserves distances
        assert np.isclose(d, d0, atol=1e-12)
    # identity config: local frame of tip is offset by (0.2, 0, 0.1)
    world = two_link.taxel_world_positions(pts, {"shoulder": 0.0})["tip"]
    assert np.allclose(world[0], [0.2, 0.0, 0.1], atol=1e-12)
