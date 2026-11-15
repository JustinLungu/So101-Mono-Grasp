"""SO-101 forward / inverse kinematics on top of placo and the SO-101 URDF.

Units match lerobot's SO101Follower: joint angles in DEGREES (use_degrees=True), positions in METERS.

All positions are in the ARM FRAME: origin on the mounting surface directly below the shoulder_pan
rotation axis, x forward, y left, z up. That's the URDF `base_link` shifted forward by ~3.9 cm (base_link's
origin sits behind the axis, where nobody can measure from with a tape).

The arm has 5 joints, so it cannot reach an arbitrary 6D pose. For picking we ask for:
  - the gripper tip position (3D),
  - the approach direction (straight down, or tilted toward the base when the point is too far),
  - the jaw yaw: which horizontal direction the jaws open along (the wrist roll handles this).
"""

from __future__ import annotations

import contextlib
import io
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import placo

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
TIP_FRAME = "gripper_frame_link"
HINGE_FRAME = "gripper"  # the gripper joint frame; its z axis is the jaw hinge axis


@contextlib.contextmanager
def _quiet_stderr():
    """placo prints a self-collision warning for the neutral pose on every load; hide it."""
    fd = os.dup(2)
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), 2)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                yield
        finally:
            os.dup2(fd, 2)
            os.close(fd)


@dataclass
class IKResult:
    joints_deg: np.ndarray  # 5 arm joints, degrees, ARM_JOINTS order
    pos_error_m: float
    approach_error_deg: float
    yaw_error_deg: float
    approach_pitch_deg: float  # 90 = straight down

    def ok(self, pos_tol_m: float = 0.005, approach_tol_deg: float = 8.0, yaw_tol_deg: float = 15.0) -> bool:
        return (
            self.pos_error_m <= pos_tol_m
            and self.approach_error_deg <= approach_tol_deg
            and self.yaw_error_deg <= yaw_tol_deg
        )


class SO101Kinematics:
    def __init__(self, urdf_path: str | Path):
        with _quiet_stderr():
            self.robot = placo.RobotWrapper(str(urdf_path))
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)
        self.solver.enable_joint_limits(True)
        self.solver.mask_dof("gripper")
        self.solver.add_regularization_task(1e-4)
        self.tip_task = self.solver.add_frame_task(TIP_FRAME, np.eye(4))

        # The jaw hinge axis expressed in the tip frame is fixed by the URDF: measure it once.
        self._set(np.zeros(5))
        T_tip = self.robot.get_T_world_frame(TIP_FRAME)
        T_hinge = self.robot.get_T_world_frame(HINGE_FRAME)
        hinge_in_tip = T_tip[:3, :3].T @ T_hinge[:3, 2]
        # Opening direction = the direction the moving jaw travels = hinge x approach (both in tip frame).
        opening = np.cross(hinge_in_tip, [0.0, 0.0, 1.0])
        self._opening_local = opening / np.linalg.norm(opening)

        # Arm frame origin = the pan axis, dropped to z = 0 of base_link.
        self.origin = self.robot.get_T_world_frame("shoulder_pan")[:3, 3].copy()
        self.origin[2] = 0.0

        lo, hi = zip(*(self.robot.get_joint_limits(j) for j in ARM_JOINTS))
        self.limits_deg = np.rad2deg(np.array([lo, hi]))  # shape (2, 5)

    def restrict_limits_deg(self, lo_deg: np.ndarray, hi_deg: np.ndarray) -> None:
        """Narrow the joint limits (e.g. to the real motor ranges from the calibration file)."""
        lo = np.maximum(self.limits_deg[0], lo_deg)
        hi = np.minimum(self.limits_deg[1], hi_deg)
        for name, a, b in zip(ARM_JOINTS, np.deg2rad(lo), np.deg2rad(hi)):
            self.robot.set_joint_limits(name, float(a), float(b))
        self.limits_deg = np.array([lo, hi])

    # ------------------------------------------------------------------ FK
    def _set(self, joints_deg: np.ndarray) -> None:
        for name, q in zip(ARM_JOINTS, np.deg2rad(np.asarray(joints_deg, float)[:5])):
            self.robot.set_joint(name, float(q))
        self.robot.update_kinematics()

    def _get(self) -> np.ndarray:
        return np.rad2deg([self.robot.get_joint(j) for j in ARM_JOINTS])

    def fk(self, joints_deg: np.ndarray) -> np.ndarray:
        """4x4 pose of the gripper tip in the arm frame. Tip z axis = approach direction (out of the jaws)."""
        self._set(joints_deg)
        T = self.robot.get_T_world_frame(TIP_FRAME).copy()
        T[:3, 3] -= self.origin
        return T

    def jaw_opening_dir(self, T_tip: np.ndarray) -> np.ndarray:
        return T_tip[:3, :3] @ self._opening_local

    def jaw_yaw(self, joints_deg: np.ndarray) -> float:
        """Yaw (radians, base frame) of the jaw opening direction at this configuration."""
        o = self.jaw_opening_dir(self.fk(joints_deg))
        return float(np.arctan2(o[1], o[0]))

    # ------------------------------------------------------------------ IK
    def _target_rotation(self, approach: np.ndarray, yaw_rad: float) -> np.ndarray:
        """Rotation whose z axis is `approach` and whose jaw-opening axis points as close as possible to yaw."""
        a = approach / np.linalg.norm(approach)
        o = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0.0])
        o = o - a * (o @ a)  # make it perpendicular to the approach
        if np.linalg.norm(o) < 1e-6:
            o = np.cross(a, [0.0, 0.0, 1.0])
        o /= np.linalg.norm(o)
        world = np.column_stack([o, np.cross(a, o), a])
        ol = self._opening_local
        local = np.column_stack([ol, np.cross([0.0, 0.0, 1.0], ol), [0.0, 0.0, 1.0]])
        return world @ local.T

    def _solve(self, T_target: np.ndarray, seed_deg: np.ndarray, iters: int, ori_weight: float) -> np.ndarray:
        self._set(seed_deg)
        self.tip_task.T_world_frame = T_target
        self.tip_task.configure("tip", "soft", 1.0, ori_weight)
        for _ in range(iters):
            self.solver.solve(True)
            self.robot.update_kinematics()
        return self._get()

    def _evaluate(self, q: np.ndarray, pos: np.ndarray, approach: np.ndarray, yaw: float, pitch: float) -> IKResult:
        T = self.fk(q)
        a = approach / np.linalg.norm(approach)
        approach_err = np.degrees(np.arccos(np.clip(T[:3, 2] @ a, -1, 1)))
        o = self.jaw_opening_dir(T)
        o_h = o[:2] / max(np.linalg.norm(o[:2]), 1e-9)
        # Jaws are symmetric: yaw and yaw + 180 deg grasp the same way (flips are prevented by the seed/target).
        yaw_err = np.degrees(np.arccos(np.clip(abs(o_h @ [np.cos(yaw), np.sin(yaw)]), 0, 1)))
        return IKResult(q, float(np.linalg.norm(T[:3, 3] - pos)), float(approach_err), float(yaw_err), pitch)

    def solve_grasp(
        self,
        pos: np.ndarray,
        yaw_rad: float | None,
        seed_deg: np.ndarray,
        pitches_deg: tuple[float, ...] = (90, 80, 70, 60, 50, 40),
        iters: int = 400,
        allow_flip: bool = True,
    ) -> IKResult:
        """IK for the gripper tip at `pos` (arm frame), approaching from above.

        Tries straight down first (pitch 90), then tilts the approach toward the base for points that are too
        far for a vertical approach. yaw_rad=None means "any yaw" (e.g. a round object).
        allow_flip: also try yaw + 180 deg (same grasp, symmetric jaws). Set False to keep the wrist roll
        continuous between waypoints (pass the yaw the previous waypoint actually reached).
        Returns the first candidate that passes IKResult.ok(), else the best one (check .ok() yourself).
        """
        pos = np.asarray(pos, float)
        radial = pos[:2] / max(np.linalg.norm(pos[:2]), 1e-9)
        best: IKResult | None = None
        for pitch in pitches_deg:
            p = np.radians(pitch)
            approach = np.array([radial[0] * np.cos(p), radial[1] * np.cos(p), -np.sin(p)])
            yaws = [None] if yaw_rad is None else [yaw_rad, yaw_rad + np.pi] if allow_flip else [yaw_rad]
            for yaw in yaws:
                y = np.arctan2(radial[1], radial[0]) + np.pi / 2 if yaw is None else yaw
                T = np.eye(4)
                T[:3, :3] = self._target_rotation(approach, y)
                T[:3, 3] = pos + self.origin
                q = self._solve(T, seed_deg, iters, ori_weight=0.05)
                res = self._evaluate(q, pos, approach, y, pitch)
                if yaw is None:
                    res.yaw_error_deg = 0.0
                if res.ok():
                    return res
                if best is None or (res.pos_error_m, res.approach_error_deg) < (
                    best.pos_error_m,
                    best.approach_error_deg,
                ):
                    best = res
        assert best is not None
        return best
