"""Phase 1: Genesis loads the Allegro URDF and a press episode produces contacts.

Requires the `sim` extra (genesis-world); skipped where it isn't installed (CI).
Marked slow: Genesis JIT-compiles kernels on first scene build.
"""

import numpy as np
import pytest

genesis = pytest.importorskip("genesis")

from sim.hand_env import ALLEGRO_URDF, run_press_episode  # noqa: E402


def test_urdf_exists():
    assert ALLEGRO_URDF.is_file()


def test_press_episode_produces_contacts(tmp_path):
    dump = run_press_episode(tmp_path / "ep.npz", n_settle=40, n_close=80, n_hold=40)

    n_steps = dump["qpos"].shape[0]
    assert n_steps == 160
    # 16 finger dofs when base is fixed
    assert dump["qpos"].shape[1] == 16
    # link poses recorded for every link at every step
    assert dump["link_pos"].shape[0] == n_steps
    assert dump["link_pos"].shape[2] == 3
    assert dump["link_quat"].shape[2] == 4

    # the ball must actually touch the hand at some point
    assert dump["contact_step"].size > 0
    # contact positions are 3D world-frame points near the hand (z above floor)
    pos = dump["contact_position"]
    assert pos.ndim == 2 and pos.shape[1] == 3

    # forces are finite, nonzero somewhere
    force = dump["contact_force_a"]
    mags = np.linalg.norm(force, axis=-1)
    assert np.isfinite(mags).all()
    assert mags.max() > 0
