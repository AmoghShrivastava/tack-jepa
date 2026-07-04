"""Stage A data generation: randomized static press episodes (PRD §6.1).

Each episode: object drops into the palm-up hand (settle), fingers close to a
randomized light-squeeze pose (close), then hold. Recorded per step: joint
state, control action, link poses AND velocities, object state, and raw
contacts extended with the relative tangential speed at each contact point —
the exact slip ground truth of §5.9 (tangential velocity between the contact
point on the hand and on the object, computed from simulator state).

Episodes vary: object shape/size (one Genesis scene per variant, reset between
episodes), drop position, box yaw, close magnitude and timing.

    python -m sim.episode_generator --out datasets/stage_a --per-variant 35
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sim.hand_env import PALM_UP_QUAT, HandEnv, _to_numpy

# (kind, dims) — 6 variants; sizes chosen to rest in the Allegro palm
VARIANTS: list[tuple] = [
    ("sphere", 0.022),
    ("sphere", 0.028),
    ("sphere", 0.034),
    ("box", (0.04, 0.04, 0.04)),
    ("box", (0.055, 0.045, 0.035)),
    ("box", (0.05, 0.05, 0.06)),
]

N_SETTLE, N_CLOSE, N_HOLD = 40, 120, 60
N_PERTURB = 60  # post-nudge steps for grasp-stability labeling (§7.4)
BASE_POS = (0.0, 0.0, 0.25)


def finger_poses(env: HandEnv, close_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Open 'basket' pose and a light-squeeze close pose, scaled per episode."""
    open_pose = np.zeros(env.n_dofs, dtype=np.float32)
    close_pose = np.zeros(env.n_dofs, dtype=np.float32)
    targets = {
        (0, 4, 8): (0.0, 0.0),      # abduction
        (12,): (0.9, 1.3),          # thumb opposition
        (13,): (0.2, 0.4),          # thumb abduction
        (1, 5, 9, 14): (0.2, 0.6),  # proximal flexion
        (2, 6, 10, 15): (0.2, 0.5),  # middle flexion
        (3, 7, 11): (0.1, 0.35),    # distal flexion
    }
    for name in env.joint_names:
        jid = int(name.split("_")[1].split(".")[0])
        dof = env.hand.get_joint(name).dofs_idx_local[0]
        for jids, (o, c) in targets.items():
            if jid in jids:
                open_pose[dof] = o
                close_pose[dof] = o + (c - o) * close_scale
    return open_pose, close_pose


def qpos22(env: HandEnv, base_pos=BASE_POS, base_quat=PALM_UP_QUAT) -> np.ndarray:
    """22-dim state: 16 finger joints + 6 wrist pose (fixed base in Stage A).

    Wrist encoded as position + rotation-vector placeholder of the constant
    base pose so the interface already matches the PRD's 22-DoF action space.
    """
    q = np.zeros(22, dtype=np.float32)
    q[:16] = _to_numpy(env.hand.get_dofs_position())[:16]
    q[16:19] = base_pos
    # palm-up base: rotation about y by -pi/2 -> rotation vector (0, -pi/2, 0)
    q[19:22] = (0.0, -np.pi / 2, 0.0)
    return q


def _point_velocities(p, link_pos, link_vel, link_ang):
    """Velocity of a world point rigidly attached to a link."""
    return link_vel + np.cross(link_ang, p - link_pos)


def contact_tangential_speeds(
    contacts: dict,
    env: HandEnv,
    link_vel: np.ndarray,
    link_ang: np.ndarray,
    link_pos: np.ndarray,
    obj_state: dict,
) -> np.ndarray:
    """|tangential relative velocity| at each hand contact (slip ground truth)."""
    n_c = len(contacts.get("position", []))
    if n_c == 0:
        return np.zeros(0, dtype=np.float32)
    global_to_entity = {int(g): i for i, g in enumerate(env.link_global_idx)}
    obj_global = set(int(g) for g in env.obj_link_global_idx)
    out = np.zeros(n_c, dtype=np.float32)
    for i in range(n_c):
        p = contacts["position"][i]
        n = contacts["normal"][i]
        v_sides = []
        for side in ("a", "b"):
            gl = int(contacts[f"link_{side}"][i])
            if gl in global_to_entity:
                e = global_to_entity[gl]
                v_sides.append(_point_velocities(p, link_pos[e], link_vel[e], link_ang[e]))
            elif gl in obj_global:
                v_sides.append(
                    _point_velocities(p, obj_state["pos"], obj_state["vel"], obj_state["ang"])
                )
            else:  # ground plane
                v_sides.append(np.zeros(3))
        v_rel = v_sides[0] - v_sides[1]
        v_tan = v_rel - np.dot(v_rel, n) * n
        out[i] = np.linalg.norm(v_tan)
    return out


def rollout_press_episode(
    env: HandEnv,
    rng: np.random.Generator,
    drop_pos,
    close_scale,
    perturb: bool = False,
):
    env.scene.reset()
    env.obj.set_pos(drop_pos)
    if env.object_spec[0] == "box":
        yaw = rng.uniform(0, np.pi / 2)
        env.obj.set_quat(np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]))
    open_pose, close_pose = finger_poses(env, close_scale)
    env.hand.set_dofs_position(open_pose)

    steps: list[dict] = []
    contact_rows: dict[str, list] = {}
    contact_steps: list[int] = []

    def record(t: int, action: np.ndarray):
        state = env.get_state()
        link_vel = _to_numpy(env.hand.get_links_vel())
        link_ang = _to_numpy(env.hand.get_links_ang())
        contacts = env.get_contacts()
        obj_state = {
            "pos": state["obj_pos"],
            "vel": state["obj_vel"],
            "ang": _to_numpy(env.obj.get_ang()),
        }
        tan_speed = contact_tangential_speeds(
            contacts, env, link_vel, link_ang, state["link_pos"], obj_state
        )
        n = len(tan_speed)
        if n:
            for k, v in contacts.items():
                contact_rows.setdefault(k, []).append(np.atleast_1d(v))
            contact_rows.setdefault("tangential_speed", []).append(tan_speed)
            contact_steps.extend([t] * n)
        state["action"] = action.copy()
        state["obj_ang"] = obj_state["ang"]
        state["link_vel"] = link_vel
        state["link_ang"] = link_ang
        steps.append(state)

    t = 0
    for _ in range(N_SETTLE):
        env.step(open_pose)
        record(t, open_pose)
        t += 1
    for i in range(N_CLOSE):
        alpha = (i + 1) / N_CLOSE
        target = open_pose + alpha * (close_pose - open_pose)
        env.step(target)
        record(t, target)
        t += 1
    for _ in range(N_HOLD):
        env.step(close_pose)
        record(t, close_pose)
        t += 1

    stable = None
    perturb_step = None
    if perturb:
        # nudge: horizontal velocity kick to the held object (PRD §7.4 grasp
        # stability ground truth), then observe whether the grasp survives
        perturb_step = t
        kick_dir = rng.uniform(0, 2 * np.pi)
        speed = rng.uniform(0.15, 0.5)
        vel = np.zeros(env.obj.n_dofs, dtype=np.float32)
        vel[0], vel[1] = speed * np.cos(kick_dir), speed * np.sin(kick_dir)
        vel[2] = rng.uniform(0.0, 0.15)
        env.obj.set_dofs_velocity(vel)
        for _ in range(N_PERTURB):
            env.step(close_pose)
            record(t, close_pose)
            t += 1
        end = steps[-1]["obj_pos"]
        horiz = float(np.linalg.norm(end[:2] - np.asarray(BASE_POS[:2])))
        stable = bool(end[2] > 0.15 and horiz < 0.15)

    dump = {
        "joint_names": np.array(env.joint_names),
        "link_names": np.array(env.link_names),
        "link_global_idx": env.link_global_idx,
        "obj_link_global_idx": env.obj_link_global_idx,
        "dt": np.float64(env.dt),
        "contact_step": np.array(contact_steps, dtype=np.int64),
    }
    for key in (
        "qpos", "qvel", "link_pos", "link_quat", "link_vel", "link_ang",
        "obj_pos", "obj_quat", "obj_vel", "obj_ang", "action",
    ):
        dump[key] = np.stack([s[key] for s in steps])
    for k, v in contact_rows.items():
        dump[f"contact_{k}"] = np.concatenate(v, axis=0)
    if perturb:
        dump["stable"] = np.uint8(stable)
        dump["perturb_step"] = np.int64(perturb_step)
    return dump


def generate_stage_a(
    out_dir: Path,
    per_variant: int,
    seed: int = 0,
    perturb: bool = False,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest = []
    for vi, spec in enumerate(VARIANTS):
        env = HandEnv(object_spec=spec)
        for ei in range(per_variant):
            drop = np.array(
                [rng.uniform(-0.02, 0.03), rng.uniform(-0.02, 0.02), rng.uniform(0.32, 0.35)]
            )
            close_scale = rng.uniform(0.75, 1.25)
            dump = rollout_press_episode(env, rng, drop, close_scale, perturb=perturb)
            # store the 22-dim action/state convention alongside raw arrays
            dump["action22"] = np.concatenate(
                [
                    dump["action"][:, :16],
                    np.zeros((dump["action"].shape[0], 6), dtype=np.float32),
                ],
                axis=1,
            )
            name = f"ep_{vi}_{ei:04d}"
            np.savez_compressed(out_dir / f"{name}.npz", **dump)
            n_contact_steps = np.unique(dump["contact_step"]).size
            manifest.append((name, spec, n_contact_steps))
            extra = f" stable={bool(dump['stable'])}" if perturb else ""
            print(
                f"{name}: {spec} steps={dump['qpos'].shape[0]} "
                f"contact_steps={n_contact_steps} "
                f"max_tan_speed={dump.get('contact_tangential_speed', np.zeros(1)).max():.4f}"
                f"{extra}"
            )
    with open(out_dir / "manifest.txt", "w") as f:
        for row in manifest:
            f.write(f"{row}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("datasets/stage_a"))
    parser.add_argument("--per-variant", type=int, default=35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode", choices=["press", "perturb"], default="press",
        help="perturb adds an object nudge + grasp-stability label (§7.4)",
    )
    args = parser.parse_args()
    generate_stage_a(args.out, args.per_variant, args.seed, perturb=args.mode == "perturb")


if __name__ == "__main__":
    main()
