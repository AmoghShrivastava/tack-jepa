"""Stage A/B data generation (PRD §6.1).

Stage A — static single-object press-only episodes: the object is a FIXED
rigid body at a stationary pose; the floating 6-DoF wrist approaches it
(randomized start/final pose) and holds; FINGER POSE IS CONSTANT for the
entire episode (a per-episode randomized curl, but never interpolated
mid-episode) — no finger motion during contact at all, matching PRD §6.1's
"no wrist/finger motion during contact, just approach-contact-hold" once the
wrist reaches its held pose. Purpose: validate basic force-geometry
association with the simplest possible dynamics.

Stage B — dynamic grasp sequences: the object is a FREE rigid body (drops
under gravity into the hand, as validated in Phases 1-3); the wrist ALSO
moves (a randomized start offset settling into the nominal engagement pose)
WHILE fingers actively interpolate open -> close over multiple timesteps.
This is where action-conditioning and temporal dynamics are first exercised,
per PRD §6.1.

Both stages exercise genuine floating-wrist control (all 22 action dims carry
real information — see ROADMAP.md decisions log for why this replaced an
earlier fixed-base implementation). Approach directions are randomized
around the validated palm-up engagement geometry with modest position/
rotation jitter; full arbitrary-direction reach/grasp diversity is Stage C's
explicit scope per PRD §6.1 ("diverse grasp/slide trajectories at scale"),
not attempted here.

    python -m sim.episode_generator --stage a --out datasets/stage_a --per-variant 35
    python -m sim.episode_generator --stage b --out datasets/stage_b --per-variant 35
"""

from __future__ import annotations

import argparse
from pathlib import Path

import genesis
import numpy as np

from sim.hand_env import (
    PALM_UP_ROTVEC,
    HandEnv,
    _to_numpy,
    finger_dof_indices,
    genesis_to_prd_order,
)

# (kind, dims) — 6 variants; sizes chosen to rest in the Allegro palm
VARIANTS: list[tuple] = [
    ("sphere", 0.022),
    ("sphere", 0.028),
    ("sphere", 0.034),
    ("box", (0.04, 0.04, 0.04)),
    ("box", (0.055, 0.045, 0.035)),
    ("box", (0.05, 0.05, 0.06)),
]

# Stage C multi-object pool (PRD §6.1): the Stage A/B primitives, plus mesh
# objects (Genesis's own bundled assets rather than external YCB — see
# ROADMAP.md Phase 6/Stage C decision log for why: untested external mesh
# import carries real non-convex/collision-instability risk that would
# corrupt tactile data, while these ship pre-tested with this exact engine)
# and procedurally generated superquadrics for smooth shape-family diversity.
_GENESIS_MESH_DIR = Path(genesis.__file__).resolve().parent / "assets" / "meshes"
_SUPERQUADRIC_DIR = Path(__file__).resolve().parent.parent / "assets" / "superquadrics"

VARIANTS_C_MESH: list[tuple] = [
    ("mesh", (str(_GENESIS_MESH_DIR / "bolt_nut" / "bolt.stl"), 1.0)),
    ("mesh", (str(_GENESIS_MESH_DIR / "bolt_nut" / "nut.stl"), 1.0)),
    ("mesh", (str(_GENESIS_MESH_DIR / "tank.obj"), 0.5)),
    ("mesh", (str(_GENESIS_MESH_DIR / "bunny.obj"), 0.06)),
]


def _variants_c_superquadric() -> list[tuple]:
    from sim.superquadric import build_superquadric_variants

    paths = build_superquadric_variants(_SUPERQUADRIC_DIR)
    return [("mesh", (str(p), 1.0)) for p in paths]


def variants_c() -> list[tuple]:
    """Full Stage C object pool: primitives + bundled meshes + superquadrics."""
    return VARIANTS + VARIANTS_C_MESH + _variants_c_superquadric()

# Validated engagement geometry (Phases 1-3): palm-up hand, object resting in
# the half-open "basket" shape of the fingers just above the palm.
NOMINAL_WRIST_POS = np.array([0.0, 0.0, 0.25])
NOMINAL_WRIST_ROTVEC = np.array(PALM_UP_ROTVEC)
NOMINAL_OBJECT_POS = np.array([0.0, 0.0, 0.308])

N_A_APPROACH, N_A_HOLD = 80, 140
N_B_SETTLE, N_B_CLOSE, N_B_HOLD = 40, 120, 60
N_B_PERTURB = 60  # post-nudge steps for grasp-stability labeling (§7.4)
N_C_SETTLE, N_C_CLOSE, N_C_SLIDE, N_C_HOLD = 40, 100, 100, 40
C_SLIDE_DIST = (0.02, 0.05)  # meters, lateral drag distance range
C_SLIDE_GRIP = (0.35, 0.65)  # deliberately looser than Stage B's grasp so the
                              # drag actually produces slip, not a rigid carry

# jitter bands (position: meters, rotation: radians, applied per-axis).
# Start-pose jitter is generous (varies the approach path); end-pose jitter
# (where contact must actually happen) is tighter so Stage A reliably makes
# contact while still varying final geometry meaningfully — tuned empirically
# so most episodes register contact (PRD §6.1's "force-geometry association"
# needs episodes that actually have force).
A_START_POS_JITTER, A_START_ROT_JITTER = 0.025, 0.25
A_END_POS_JITTER, A_END_ROT_JITTER = 0.008, 0.08
B_POS_JITTER, B_ROT_JITTER = 0.018, 0.17


def finger_targets(env: HandEnv, curl: float | tuple[float, float]) -> np.ndarray:
    """Full n_dofs target vector with the wrist slice left at 0 (caller fills
    it in) and fingers set to a curl amount in [0, 1] (0=open flat, 1=full
    curl toward the PRD's original close pose). `curl` may be a (low, high)
    pair for open/close targets or a single scalar for a static pose."""
    lo, hi = curl if isinstance(curl, tuple) else (curl, curl)
    target = np.zeros((2, env.n_dofs), dtype=np.float32)
    dof = finger_dof_indices(env)
    band = {
        (0, 4, 8): (0.0, 0.0),        # abduction: neutral
        (12,): (0.9, 1.3),            # thumb opposition
        (13,): (0.2, 0.4),            # thumb abduction
        (1, 5, 9, 14): (0.2, 0.6),    # proximal flexion
        (2, 6, 10, 15): (0.2, 0.5),   # middle flexion
        (3, 7, 11): (0.1, 0.35),      # distal flexion
    }
    for name in env.joint_names:
        jid = int(name.split("_")[1].split(".")[0])
        d = dof[name]
        for jids, (o, c) in band.items():
            if jid in jids:
                target[0, d] = o + (c - o) * lo
                target[1, d] = o + (c - o) * hi
    return target  # (2, n_dofs): [low_curl_target, high_curl_target]


def _record_factory(env: HandEnv):
    contact_rows: dict[str, list] = {}
    contact_steps: list[int] = []
    steps: list[dict] = []

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

    return record, steps, contact_rows, contact_steps


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


def _finalize(
    env: HandEnv, steps: list[dict], contact_rows: dict, contact_steps: list[int]
) -> dict:
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
    # PRD §5.2 action/state convention: [finger16, wrist_pos3, wrist_rotvec3]
    dump["qpos22"] = genesis_to_prd_order(dump["qpos"])
    dump["action22"] = genesis_to_prd_order(dump["action"])
    return dump


def rollout_stage_a_episode(
    env: HandEnv, rng: np.random.Generator, object_pos: np.ndarray
) -> dict:
    """Static press: object fixed, wrist approaches + holds, fingers constant."""
    env.scene.reset()
    env.obj.set_pos(object_pos)
    if env.object_spec[0] == "box":
        yaw = rng.uniform(0, np.pi / 2)
        env.obj.set_quat(np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]))

    start_pos = NOMINAL_WRIST_POS + rng.uniform(-A_START_POS_JITTER, A_START_POS_JITTER, size=3)
    start_rot = NOMINAL_WRIST_ROTVEC + rng.uniform(-A_START_ROT_JITTER, A_START_ROT_JITTER, size=3)
    end_pos = NOMINAL_WRIST_POS + rng.uniform(-A_END_POS_JITTER, A_END_POS_JITTER, size=3)
    end_rot = NOMINAL_WRIST_ROTVEC + rng.uniform(-A_END_ROT_JITTER, A_END_ROT_JITTER, size=3)
    curl = rng.uniform(0.85, 1.2)  # constant for the whole episode — no finger motion
    finger_target = finger_targets(env, curl)[0]

    env.hand.set_dofs_position(
        np.concatenate([start_pos, start_rot, finger_target[6:]]).astype(np.float32),
        zero_velocity=True,
    )

    record, steps, contact_rows, contact_steps = _record_factory(env)
    t = 0
    for i in range(N_A_APPROACH):
        alpha = (i + 1) / N_A_APPROACH
        wrist = start_pos + alpha * (end_pos - start_pos)
        wrist_rot = start_rot + alpha * (end_rot - start_rot)
        target = np.concatenate([wrist, wrist_rot, finger_target[6:]]).astype(np.float32)
        env.step(target)
        record(t, target)
        t += 1
    hold_target = np.concatenate([end_pos, end_rot, finger_target[6:]]).astype(np.float32)
    for _ in range(N_A_HOLD):
        env.step(hold_target)
        record(t, hold_target)
        t += 1

    return _finalize(env, steps, contact_rows, contact_steps)


def rollout_stage_b_episode(
    env: HandEnv,
    rng: np.random.Generator,
    drop_pos: np.ndarray,
    close_scale: float,
    perturb: bool = False,
) -> dict:
    """Dynamic grasp: object free-falls, wrist settles into engagement pose
    while fingers actively close — multi-timestep, action-conditioned."""
    env.scene.reset()
    env.obj.set_pos(drop_pos)
    if env.object_spec[0] == "box":
        yaw = rng.uniform(0, np.pi / 2)
        env.obj.set_quat(np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]))

    start_pos = NOMINAL_WRIST_POS + rng.uniform(-B_POS_JITTER, B_POS_JITTER, size=3)
    start_rot = NOMINAL_WRIST_ROTVEC + rng.uniform(-B_ROT_JITTER, B_ROT_JITTER, size=3)
    end_pos, end_rot = NOMINAL_WRIST_POS, NOMINAL_WRIST_ROTVEC
    fingers = finger_targets(env, (0.0, 1.0))
    open_fingers, close_fingers = fingers[0], fingers[1]
    close_fingers = open_fingers + (close_fingers - open_fingers) * close_scale

    env.hand.set_dofs_position(
        np.concatenate([start_pos, start_rot, open_fingers[6:]]).astype(np.float32),
        zero_velocity=True,
    )

    record, steps, contact_rows, contact_steps = _record_factory(env)
    t = 0
    n_wrist_settle = N_B_SETTLE + N_B_CLOSE  # wrist arrives by end of close phase
    for i in range(N_B_SETTLE):
        alpha = (i + 1) / n_wrist_settle
        wrist = start_pos + alpha * (end_pos - start_pos)
        wrist_rot = start_rot + alpha * (end_rot - start_rot)
        target = np.concatenate([wrist, wrist_rot, open_fingers[6:]]).astype(np.float32)
        env.step(target)
        record(t, target)
        t += 1
    for i in range(N_B_CLOSE):
        alpha_w = (N_B_SETTLE + i + 1) / n_wrist_settle
        beta = (i + 1) / N_B_CLOSE
        wrist = start_pos + alpha_w * (end_pos - start_pos)
        wrist_rot = start_rot + alpha_w * (end_rot - start_rot)
        finger = open_fingers + beta * (close_fingers - open_fingers)
        target = np.concatenate([wrist, wrist_rot, finger[6:]]).astype(np.float32)
        env.step(target)
        record(t, target)
        t += 1
    hold_target = np.concatenate([end_pos, end_rot, close_fingers[6:]]).astype(np.float32)
    for _ in range(N_B_HOLD):
        env.step(hold_target)
        record(t, hold_target)
        t += 1

    stable = None
    if perturb:
        # nudge: horizontal velocity kick to the held object (PRD §7.4 grasp
        # stability ground truth), then observe whether the grasp survives
        kick_dir = rng.uniform(0, 2 * np.pi)
        speed = rng.uniform(0.15, 0.5)
        vel = np.zeros(env.obj.n_dofs, dtype=np.float32)
        vel[0], vel[1] = speed * np.cos(kick_dir), speed * np.sin(kick_dir)
        vel[2] = rng.uniform(0.0, 0.15)
        env.obj.set_dofs_velocity(vel)
        for _ in range(N_B_PERTURB):
            env.step(hold_target)
            record(t, hold_target)
            t += 1
        end = steps[-1]["obj_pos"]
        horiz = float(np.linalg.norm(end[:2] - NOMINAL_OBJECT_POS[:2]))
        stable = bool(end[2] > 0.15 and horiz < 0.15)

    dump = _finalize(env, steps, contact_rows, contact_steps)
    if perturb:
        dump["stable"] = np.uint8(stable)
    return dump


def rollout_stage_c_slide_episode(
    env: HandEnv, rng: np.random.Generator, drop_pos: np.ndarray, grip_scale: float
) -> dict:
    """Slide: object free-falls into a deliberately loose grip (PRD §6.1 Stage
    C "slide" trajectory), then the wrist drags it laterally across the
    palm/fingers — the loose grip lets it actually slip rather than being
    carried rigidly, producing the tangential-slip diversity the slip probe
    (§5.9) and downstream slip-onset transfer task (§7.4) need."""
    env.scene.reset()
    env.obj.set_pos(drop_pos)
    if env.object_spec[0] == "box":
        yaw = rng.uniform(0, np.pi / 2)
        env.obj.set_quat(np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]))

    start_pos = NOMINAL_WRIST_POS + rng.uniform(-B_POS_JITTER, B_POS_JITTER, size=3)
    start_rot = NOMINAL_WRIST_ROTVEC + rng.uniform(-B_ROT_JITTER, B_ROT_JITTER, size=3)
    settle_pos, settle_rot = NOMINAL_WRIST_POS, NOMINAL_WRIST_ROTVEC
    fingers = finger_targets(env, (0.0, 1.0))
    open_fingers, close_fingers = fingers[0], fingers[1]
    grip_fingers = open_fingers + (close_fingers - open_fingers) * grip_scale

    env.hand.set_dofs_position(
        np.concatenate([start_pos, start_rot, open_fingers[6:]]).astype(np.float32),
        zero_velocity=True,
    )

    record, steps, contact_rows, contact_steps = _record_factory(env)
    t = 0
    n_wrist_settle = N_C_SETTLE + N_C_CLOSE
    for i in range(N_C_SETTLE):
        alpha = (i + 1) / n_wrist_settle
        wrist = start_pos + alpha * (settle_pos - start_pos)
        wrist_rot = start_rot + alpha * (settle_rot - start_rot)
        target = np.concatenate([wrist, wrist_rot, open_fingers[6:]]).astype(np.float32)
        env.step(target)
        record(t, target)
        t += 1
    for i in range(N_C_CLOSE):
        alpha_w = (N_C_SETTLE + i + 1) / n_wrist_settle
        beta = (i + 1) / N_C_CLOSE
        wrist = start_pos + alpha_w * (settle_pos - start_pos)
        wrist_rot = start_rot + alpha_w * (settle_rot - start_rot)
        finger = open_fingers + beta * (grip_fingers - open_fingers)
        target = np.concatenate([wrist, wrist_rot, finger[6:]]).astype(np.float32)
        env.step(target)
        record(t, target)
        t += 1

    slide_dir = rng.uniform(0, 2 * np.pi)
    slide_dist = rng.uniform(*C_SLIDE_DIST)
    slide_delta = slide_dist * np.array([np.cos(slide_dir), np.sin(slide_dir), 0.0])
    for i in range(N_C_SLIDE):
        alpha = (i + 1) / N_C_SLIDE
        wrist = settle_pos + alpha * slide_delta
        target = np.concatenate([wrist, settle_rot, grip_fingers[6:]]).astype(np.float32)
        env.step(target)
        record(t, target)
        t += 1
    hold_target = np.concatenate(
        [settle_pos + slide_delta, settle_rot, grip_fingers[6:]]
    ).astype(np.float32)
    for _ in range(N_C_HOLD):
        env.step(hold_target)
        record(t, hold_target)
        t += 1

    return _finalize(env, steps, contact_rows, contact_steps)


def generate_stage(
    stage: str, out_dir: Path, per_variant: int, seed: int = 0, perturb: bool = False
):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    variants = variants_c() if stage == "c" else VARIANTS
    n_failed = 0
    # append-mode + flush per line: a crash (Genesis's rigid solver can raise
    # on a genuinely degenerate contact configuration — hit once in practice
    # during Stage C generation) must not lose the manifest for everything
    # already generated, only the one episode that failed.
    with open(out_dir / "manifest.txt", "a") as manifest_f:
        for vi, spec in enumerate(variants):
            # Stage C uses a genuinely free object even for "press"-style
            # episodes (PRD §6.1's diverse-trajectory main pretraining stage)
            # — only Stage A pins the object in place for the simplest
            # possible debug signal.
            env = HandEnv(object_spec=spec, object_fixed=(stage == "a"))
            for ei in range(per_variant):
                name = f"ep_{vi}_{ei:04d}"
                npz_path = out_dir / f"{name}.npz"
                if npz_path.exists():
                    continue  # resume: skip episodes already on disk

                try:
                    if stage == "a":
                        obj_pos = NOMINAL_OBJECT_POS + rng.uniform(-0.01, 0.01, size=3)
                        dump = rollout_stage_a_episode(env, rng, obj_pos)
                        traj = "press"
                    elif stage == "b":
                        drop = NOMINAL_OBJECT_POS + np.array(
                            [rng.uniform(-0.02, 0.03), rng.uniform(-0.02, 0.02), rng.uniform(-0.01, 0.02)]
                        )
                        close_scale = rng.uniform(0.75, 1.25)
                        dump = rollout_stage_b_episode(env, rng, drop, close_scale, perturb=perturb)
                        traj = "grasp"
                    else:  # stage c: mix press/grasp/slide trajectories (PRD §6.1)
                        traj = rng.choice(["press", "grasp", "slide"])
                        drop = NOMINAL_OBJECT_POS + np.array(
                            [rng.uniform(-0.02, 0.03), rng.uniform(-0.02, 0.02), rng.uniform(-0.01, 0.02)]
                        )
                        if traj == "press":
                            dump = rollout_stage_a_episode(env, rng, drop)
                        elif traj == "grasp":
                            close_scale = rng.uniform(0.75, 1.25)
                            dump = rollout_stage_b_episode(env, rng, drop, close_scale, perturb=perturb)
                        else:
                            grip_scale = rng.uniform(*C_SLIDE_GRIP)
                            dump = rollout_stage_c_slide_episode(env, rng, drop, grip_scale)
                except Exception as e:  # noqa: BLE001 — genuinely any solver failure
                    n_failed += 1
                    print(f"{name}: {spec} FAILED ({type(e).__name__}: {e}) — skipping, rebuilding env")
                    # a NaN/solver exception can leave the scene's internal
                    # state corrupted; rebuild fresh rather than risk silently
                    # bad data on every subsequent episode of this variant
                    env = HandEnv(object_spec=spec, object_fixed=(stage == "a"))
                    continue

                np.savez_compressed(npz_path, **dump)
                n_contact_steps = np.unique(dump["contact_step"]).size
                manifest_f.write(f"{(name, spec, traj, n_contact_steps)}\n")
                manifest_f.flush()
                max_tan = dump.get("contact_tangential_speed", np.zeros(1)).max()
                extra = f" stable={bool(dump['stable'])}" if perturb and traj == "grasp" else ""
                print(
                    f"{name}: {spec} traj={traj} steps={dump['qpos'].shape[0]} "
                    f"contact_steps={n_contact_steps} max_tan_speed={max_tan:.4f}{extra}"
                )
    if n_failed:
        print(f"generation done with {n_failed} episode(s) skipped due to solver failures")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["a", "b", "c"], required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--per-variant", type=int, default=35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--perturb", action="store_true",
        help="Stage B/C grasp episodes only: add an object nudge + grasp-stability label (§7.4)",
    )
    args = parser.parse_args()
    out = args.out or Path(f"datasets/stage_{args.stage}")
    generate_stage(args.stage, out, args.per_variant, args.seed, perturb=args.perturb)


if __name__ == "__main__":
    main()
