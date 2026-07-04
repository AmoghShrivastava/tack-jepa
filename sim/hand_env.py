"""Genesis scene: Allegro hand + object, headless episode rollout (PRD §5.1-5.2).

Phase 1 scope: load the hand URDF, drop a ball into the half-closed fingers,
close them (a press episode), and record raw contact-solver output per step.
Run as a script to dump one episode:

    python -m sim.hand_env --out episodes/phase1_smoke.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
ALLEGRO_URDF = ASSETS_DIR / "urdf" / "allegro_hand" / "allegro_hand_right.urdf"

# fingertip links are fixed-jointed; Genesis merges those into their parents by
# default, which would break per-link contact attribution against the taxel
# layout (21 links) — keep them as distinct solver links
TIP_LINKS = ["link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip"]

# base rotation Ry(-pi/2) as (w,x,y,z): palm normal (+x in base frame) -> +z,
# i.e. palm faces up, fingers extend horizontally toward -x
PALM_UP_QUAT = (0.7071068, 0.0, -0.7071068, 0.0)

_gs_initialized = False


def init_genesis(backend: str = "cpu", logging_level: str = "warning"):
    """Initialize Genesis exactly once per process."""
    global _gs_initialized
    import genesis as gs

    if not _gs_initialized:
        gs.init(backend=getattr(gs, backend), logging_level=logging_level)
        _gs_initialized = True
    return gs


def _to_numpy(x) -> np.ndarray:
    """Genesis returns backend tensors or numpy depending on version/backend."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):  # torch-like
        return x.detach().cpu().numpy()
    return np.asarray(x)


class HandEnv:
    """Headless Genesis scene with an Allegro hand and a single object."""

    def __init__(
        self,
        dt: float = 0.01,
        hand_pos: tuple = (0.0, 0.0, 0.25),
        hand_quat: tuple = PALM_UP_QUAT,
        hand_fixed: bool = True,
        object_radius: float = 0.028,
        object_pos: tuple = (0.0, 0.0, 0.33),
        object_spec: tuple | None = None,  # ("sphere", r) | ("box", (sx, sy, sz))
        show_viewer: bool = False,
    ):
        gs = init_genesis()
        self.gs = gs
        self.dt = dt
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=dt),
            show_viewer=show_viewer,
        )
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        # recompute_inertia: several upstream Allegro links carry inertia
        # tensors violating the A+B>=C triangle inequality; Genesis rebuilds
        # them from collision geometry + mass instead of rejecting the model
        self.hand = self.scene.add_entity(
            gs.morphs.URDF(
                file=str(ALLEGRO_URDF),
                pos=hand_pos,
                quat=hand_quat,
                fixed=hand_fixed,
                recompute_inertia=True,
                links_to_keep=TIP_LINKS,
            )
        )
        if object_spec is None:
            object_spec = ("sphere", object_radius)
        kind, dims = object_spec
        if kind == "sphere":
            obj_morph = gs.morphs.Sphere(radius=float(dims), pos=object_pos)
        elif kind == "box":
            obj_morph = gs.morphs.Box(size=tuple(dims), pos=object_pos)
        else:
            raise ValueError(f"unknown object kind {kind!r}")
        self.object_spec = (kind, dims)
        self.obj = self.scene.add_entity(obj_morph)
        self.scene.build()

        self.joint_names = [j.name for j in self.hand.joints]
        self.link_names = [ln.name for ln in self.hand.links]
        self.link_global_idx = np.array([ln.idx for ln in self.hand.links])
        self.obj_link_global_idx = np.array([ln.idx for ln in self.obj.links])
        self.n_dofs = self.hand.n_dofs

    # ------------------------------------------------------------------ state
    def get_state(self) -> dict:
        return {
            "qpos": _to_numpy(self.hand.get_dofs_position()),
            "qvel": _to_numpy(self.hand.get_dofs_velocity()),
            "link_pos": _to_numpy(self.hand.get_links_pos()),
            "link_quat": _to_numpy(self.hand.get_links_quat()),
            "obj_pos": _to_numpy(self.obj.get_pos()),
            "obj_quat": _to_numpy(self.obj.get_quat()),
            "obj_vel": _to_numpy(self.obj.get_vel()),
        }

    def get_contacts(self) -> dict:
        """Raw contact-solver output for the hand (world frame)."""
        raw = self.hand.get_contacts()
        return {k: _to_numpy(v) for k, v in raw.items()}

    # ------------------------------------------------------------------ sim
    def step(self, target_qpos: np.ndarray | None = None):
        if target_qpos is not None:
            self.hand.control_dofs_position(target_qpos)
        self.scene.step()


def run_press_episode(
    out_path: Path,
    n_settle: int = 60,
    n_close: int = 150,
    n_hold: int = 90,
) -> dict:
    """Drop a ball into half-open fingers, close them, hold. Record everything."""
    env = HandEnv()

    # Half-open "basket" start pose; close toward a grasp. Joint order comes
    # from the entity itself; finger flexion joints get the curl, abduction
    # joints (0, 4, 8) stay near zero, thumb opposition (12) stays engaged.
    open_pose = np.zeros(env.n_dofs, dtype=np.float32)
    close_pose = np.zeros(env.n_dofs, dtype=np.float32)
    for name in env.joint_names:
        if not name.startswith("joint_"):
            continue
        jid = int(name.split("_")[1].split(".")[0])
        dof = env.hand.get_joint(name).dofs_idx_local[0]
        if jid in (0, 4, 8):            # finger abduction: keep neutral
            open_pose[dof], close_pose[dof] = 0.0, 0.0
        elif jid == 12:                 # thumb opposition
            open_pose[dof], close_pose[dof] = 0.9, 1.3
        elif jid == 13:                 # thumb abduction
            open_pose[dof], close_pose[dof] = 0.2, 0.4
        elif jid in (1, 5, 9, 14):      # proximal flexion: light squeeze only
            open_pose[dof], close_pose[dof] = 0.2, 0.6
        elif jid in (2, 6, 10, 15):     # middle flexion
            open_pose[dof], close_pose[dof] = 0.2, 0.5
        else:                           # distal flexion (3, 7, 11)
            open_pose[dof], close_pose[dof] = 0.1, 0.35

    records: list[dict] = []
    contact_rows: dict[str, list] = {}
    contact_steps: list[int] = []

    def record(t: int):
        state = env.get_state()
        state["t"] = t
        records.append(state)
        contacts = env.get_contacts()
        n = len(contacts.get("position", []))
        if n:
            for k, v in contacts.items():
                contact_rows.setdefault(k, []).append(np.atleast_1d(v))
            contact_steps.extend([t] * n)

    t = 0
    env.hand.set_dofs_position(open_pose)
    for _ in range(n_settle):
        env.step(open_pose)
        record(t)
        t += 1
    for i in range(n_close):
        alpha = (i + 1) / n_close
        env.step(open_pose + alpha * (close_pose - open_pose))
        record(t)
        t += 1
    for _ in range(n_hold):
        env.step(close_pose)
        record(t)
        t += 1

    # Flatten into a single npz (contacts concatenated with a step column).
    dump = {
        "joint_names": np.array(env.joint_names),
        "link_names": np.array(env.link_names),
        # contacts use GLOBAL solver link indices; these arrays map them back
        "link_global_idx": env.link_global_idx,
        "obj_link_global_idx": env.obj_link_global_idx,
        "dt": np.float64(env.dt),
        "qpos": np.stack([r["qpos"] for r in records]),
        "qvel": np.stack([r["qvel"] for r in records]),
        "link_pos": np.stack([r["link_pos"] for r in records]),
        "link_quat": np.stack([r["link_quat"] for r in records]),
        "obj_pos": np.stack([r["obj_pos"] for r in records]),
        "obj_quat": np.stack([r["obj_quat"] for r in records]),
        "obj_vel": np.stack([r["obj_vel"] for r in records]),
        "contact_step": np.array(contact_steps, dtype=np.int64),
    }
    for k, v in contact_rows.items():
        dump[f"contact_{k}"] = np.concatenate(v, axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **dump)
    return dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("episodes/phase1_smoke.npz"))
    args = parser.parse_args()

    dump = run_press_episode(args.out)
    n_steps = dump["qpos"].shape[0]
    n_contacts = dump["contact_step"].shape[0]
    steps_with_contact = np.unique(dump["contact_step"]).shape[0]
    print(f"episode steps:            {n_steps}")
    print(f"total contact records:    {n_contacts}")
    print(f"steps with >=1 contact:   {steps_with_contact}")
    if n_contacts:
        forces = dump.get("contact_force_a")
        if forces is not None:
            mags = np.linalg.norm(forces.reshape(n_contacts, -1), axis=1)
            print(f"force magnitude (N):      min={mags.min():.4f} "
                  f"mean={mags.mean():.4f} max={mags.max():.4f}")
        links = dump.get("contact_link_a")
        if links is not None:
            print(f"distinct links in contact: {np.unique(links).size}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
