"""genesis<->PRD dof-order conversion (fixes the fixed-wrist discrepancy).

Pure numpy — no Genesis needed. See ROADMAP.md decisions log: Genesis's
native floating-base layout is [wrist_pos3, wrist_rot3, finger16]; PRD §5.2
specifies action/state order [finger16, wrist_pos3, wrist_rot3].
"""

import numpy as np

from sim.hand_env import genesis_to_prd_order, prd_to_genesis_order


def test_roundtrip():
    v = np.arange(22).astype(np.float32)
    assert np.allclose(prd_to_genesis_order(genesis_to_prd_order(v)), v)
    assert np.allclose(genesis_to_prd_order(prd_to_genesis_order(v)), v)


def test_field_mapping():
    genesis = np.zeros(22)
    genesis[0:3] = [1, 2, 3]      # wrist pos
    genesis[3:6] = [4, 5, 6]      # wrist rotvec
    genesis[6:22] = np.arange(16)  # fingers
    prd = genesis_to_prd_order(genesis)
    assert np.allclose(prd[0:16], np.arange(16))
    assert np.allclose(prd[16:19], [1, 2, 3])
    assert np.allclose(prd[19:22], [4, 5, 6])


def test_batched():
    v = np.random.default_rng(0).normal(size=(5, 7, 22)).astype(np.float32)
    out = genesis_to_prd_order(v)
    assert out.shape == v.shape
    assert np.allclose(out[..., 0:16], v[..., 6:22])
    assert np.allclose(out[..., 16:22], v[..., 0:6])
