"""Cross-validate our standalone FK against Genesis's link poses (PRD Phase 2).

If these agree at random joint configurations, taxel world positions computed
offline from logged joint states are exactly what the simulator saw — the
'geometry is arithmetic, not inference' claim made checkable. Now exercised
under a genuinely moving floating wrist (not just a fixed constant base),
since HandEnv defaults to a free 6-DoF wrist (PRD §5.2).

Requires genesis (skipped in CI); reuses one scene for all configs.
"""

import numpy as np
import pytest

pytest.importorskip("genesis")

from sim.forward_kinematics import KinematicChain, quat_to_matrix  # noqa: E402
from sim.hand_env import ALLEGRO_URDF, WRIST_POS, WRIST_ROT, HandEnv  # noqa: E402


@pytest.fixture(scope="module")
def env():
    return HandEnv()


@pytest.fixture(scope="module")
def chain():
    return KinematicChain.from_urdf(ALLEGRO_URDF)


def test_link_poses_match_genesis(env, chain):
    rng = np.random.default_rng(7)
    # joint limits from the URDF via genesis
    joints = [env.hand.get_joint(n) for n in env.joint_names]
    lo = np.array([j.dofs_limit[0][0] for j in joints])
    hi = np.array([j.dofs_limit[0][1] for j in joints])
    dof_idx = [j.dofs_idx_local[0] for j in joints]

    for trial in range(5):
        q = rng.uniform(lo, hi)
        qpos = np.zeros(env.n_dofs)
        for d, v in zip(dof_idx, q, strict=True):
            qpos[d] = v
        # also randomize the floating wrist itself (position + rotvec) — this
        # is real, controllable state now, not a fixed constant
        qpos[WRIST_POS] = rng.uniform(-0.05, 0.05, size=3) + [0.0, 0.0, 0.25]
        qpos[WRIST_ROT] = rng.uniform(-0.4, 0.4, size=3) + [0.0, -np.pi / 2, 0.0]
        env.hand.set_dofs_position(qpos, zero_velocity=True)

        gs_pos = np.asarray(env.hand.get_links_pos())
        gs_quat = np.asarray(env.hand.get_links_quat())
        base_pos = np.asarray(env.hand.get_pos())
        base_quat = np.asarray(env.hand.get_quat())

        ours = chain.fk(
            dict(zip(env.joint_names, q, strict=True)),
            base_pos=base_pos,
            base_quat=base_quat,
        )
        for li, name in enumerate(env.link_names):
            if name not in ours:
                continue
            got_pos = ours[name].pos
            assert np.allclose(got_pos, gs_pos[li], atol=1e-5), (
                f"trial {trial} link {name}: ours {got_pos} vs genesis {gs_pos[li]}"
            )
            # compare rotations (quat sign is arbitrary): R_ours ~ R_genesis
            R_gs = quat_to_matrix(gs_quat[li])
            assert np.allclose(ours[name].rot, R_gs, atol=1e-5), (
                f"trial {trial} link {name} rotation mismatch"
            )
