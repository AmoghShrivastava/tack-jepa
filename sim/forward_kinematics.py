"""Standalone forward kinematics from a URDF (PRD §5.2, §5.5).

Pure numpy — deliberately independent of Genesis so that (a) taxel world
positions can be computed from logged joint states without a simulator in the
loop, and (b) correctness is testable two ways: against hand-derived poses and
against Genesis's own link poses.

Conventions: quaternions are (w, x, y, z), matching Genesis. Joint angles are
passed as a dict or an array ordered by `chain.actuated_joint_names`.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def rpy_to_matrix(rpy) -> np.ndarray:
    """URDF rpy = fixed-axis XYZ rolls: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def rotvec_to_quat(rotvec) -> np.ndarray:
    """Rotation vector (axis-angle, magnitude = angle) -> quaternion (w,x,y,z).

    Genesis's floating-base free joint represents wrist orientation this way
    (verified empirically 2026-07-04) — this is the inverse of the mapping a
    caller needs when going from a commanded/logged rotvec to a quaternion
    for FK's base_quat argument.
    """
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return matrix_to_quat(axis_angle_to_matrix(rotvec / angle, angle))


def axis_angle_to_matrix(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1 - (xx + yy)],
        ]
    )


@dataclass
class Joint:
    name: str
    joint_type: str  # "revolute" | "fixed" (all the Allegro URDF contains)
    parent: str
    child: str
    origin_pos: np.ndarray
    origin_rot: np.ndarray  # 3x3
    axis: np.ndarray | None  # None for fixed joints


@dataclass
class LinkPose:
    pos: np.ndarray  # (3,)
    rot: np.ndarray  # (3, 3)

    @property
    def quat(self) -> np.ndarray:
        return matrix_to_quat(self.rot)


@dataclass
class KinematicChain:
    root_link: str
    link_names: list[str]
    joints: list[Joint]
    actuated_joint_names: list[str] = field(init=False)
    _children: dict[str, list[Joint]] = field(init=False)

    def __post_init__(self):
        self.actuated_joint_names = [
            j.name for j in self.joints if j.joint_type != "fixed"
        ]
        self._children = {}
        for j in self.joints:
            self._children.setdefault(j.parent, []).append(j)

    @classmethod
    def from_urdf(cls, path: str | Path) -> KinematicChain:
        root = ET.parse(path).getroot()
        link_names = [ln.attrib["name"] for ln in root.findall("link")]
        joints = []
        for je in root.findall("joint"):
            jtype = je.attrib["type"]
            if jtype not in ("revolute", "fixed", "continuous"):
                raise NotImplementedError(f"joint type {jtype!r} not supported")
            origin = je.find("origin")
            xyz = np.zeros(3)
            rot = np.eye(3)
            if origin is not None:
                if "xyz" in origin.attrib:
                    xyz = np.array([float(v) for v in origin.attrib["xyz"].split()])
                if "rpy" in origin.attrib:
                    rot = rpy_to_matrix([float(v) for v in origin.attrib["rpy"].split()])
            axis = None
            if jtype != "fixed":
                axis_el = je.find("axis")
                axis = (
                    np.array([float(v) for v in axis_el.attrib["xyz"].split()])
                    if axis_el is not None
                    else np.array([1.0, 0.0, 0.0])
                )
            joints.append(
                Joint(
                    name=je.attrib["name"],
                    joint_type="revolute" if jtype == "continuous" else jtype,
                    parent=je.find("parent").attrib["link"],
                    child=je.find("child").attrib["link"],
                    origin_pos=xyz,
                    origin_rot=rot,
                    axis=axis,
                )
            )
        children = {j.child for j in joints}
        roots = [ln for ln in link_names if ln not in children]
        if len(roots) != 1:
            raise ValueError(f"expected exactly one root link, got {roots}")
        return cls(root_link=roots[0], link_names=link_names, joints=joints)

    def fk(
        self,
        qpos: dict[str, float] | np.ndarray,
        base_pos=(0.0, 0.0, 0.0),
        base_quat=(1.0, 0.0, 0.0, 0.0),
    ) -> dict[str, LinkPose]:
        """World pose of every link given joint angles and a base pose."""
        if not isinstance(qpos, dict):
            qpos_arr = np.asarray(qpos, dtype=np.float64)
            if qpos_arr.shape != (len(self.actuated_joint_names),):
                raise ValueError(
                    f"qpos has shape {qpos_arr.shape}, expected "
                    f"({len(self.actuated_joint_names)},) "
                    f"ordered by actuated_joint_names"
                )
            qpos = dict(zip(self.actuated_joint_names, qpos_arr, strict=True))

        poses = {
            self.root_link: LinkPose(
                pos=np.asarray(base_pos, dtype=np.float64),
                rot=quat_to_matrix(base_quat),
            )
        }
        stack = [self.root_link]
        while stack:
            parent = stack.pop()
            pp = poses[parent]
            for j in self._children.get(parent, ()):
                rot = pp.rot @ j.origin_rot
                pos = pp.pos + pp.rot @ j.origin_pos
                if j.joint_type == "revolute":
                    rot = rot @ axis_angle_to_matrix(j.axis, qpos[j.name])
                poses[j.child] = LinkPose(pos=pos, rot=rot)
                stack.append(j.child)
        return poses

    def taxel_world_positions(
        self,
        link_local_points: dict[str, np.ndarray],
        qpos,
        base_pos=(0.0, 0.0, 0.0),
        base_quat=(1.0, 0.0, 0.0, 0.0),
    ) -> dict[str, np.ndarray]:
        """Map per-link local-frame points (N_i, 3) to world frame — the exact
        'taxel positions are arithmetic, not inference' step (PRD §1.2)."""
        poses = self.fk(qpos, base_pos=base_pos, base_quat=base_quat)
        return {
            link: (poses[link].rot @ pts.T).T + poses[link].pos
            for link, pts in link_local_points.items()
        }
