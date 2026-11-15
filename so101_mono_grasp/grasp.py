"""Grasp planning (floor point -> joint waypoints) and execution on the follower.

Joint-angle convention: the IK works in URDF angles. The follower speaks lerobot calibrated degrees
(0 = middle of the range recorded during motor calibration). If those don't line up, set
`urdf_offset_deg` in config/grasp.yaml so that  urdf_deg = motor_deg + offset  (check with pick.py --where).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .arms import REPO_ROOT, make_follower
from .kinematics import ARM_JOINTS, IKResult, SO101Kinematics

DEFAULT_GRASP_CONFIG = REPO_ROOT / "config" / "grasp.yaml"
CONTROL_HZ = 50


def resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def load_grasp_config(path: str | Path = DEFAULT_GRASP_CONFIG) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def calibrated_limits_deg(calib_json: str | Path, margin_deg: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Joint range in lerobot degrees from a motor calibration file (0 = middle of the recorded range)."""
    with open(calib_json) as f:
        cal = json.load(f)
    half = np.array([(cal[j]["range_max"] - cal[j]["range_min"]) / 2 * 360 / 4095 for j in ARM_JOINTS])
    return -half + margin_deg, half - margin_deg


# ---------------------------------------------------------------------------- planning
@dataclass
class Waypoint:
    name: str
    joints_deg: np.ndarray  # motor degrees, ARM_JOINTS order
    gripper: float  # 0-100
    tip_xyz: np.ndarray | None = None


@dataclass
class GraspPlan:
    target_xyz: np.ndarray
    yaw_deg: float | None
    ik: IKResult
    waypoints: list[Waypoint] = field(default_factory=list)


class GraspPlanner:
    def __init__(self, gcfg: dict[str, Any], follower_calib_json: str | Path | None = None):
        self.cfg = gcfg
        self.kin = SO101Kinematics(resolve(gcfg["urdf"]))
        off = gcfg.get("urdf_offset_deg") or {}
        self.offset = np.array([float(off.get(j, 0.0)) for j in ARM_JOINTS])
        if follower_calib_json is not None:
            lo, hi = calibrated_limits_deg(follower_calib_json)
            self.kin.restrict_limits_deg(lo + self.offset, hi + self.offset)

    # motor <-> urdf degrees
    def to_urdf(self, motor_deg: np.ndarray) -> np.ndarray:
        return np.asarray(motor_deg, float) + self.offset

    def to_motor(self, urdf_deg: np.ndarray) -> np.ndarray:
        return np.asarray(urdf_deg, float) - self.offset

    def pose(self, name: str) -> np.ndarray | None:
        p = (self.cfg.get("poses") or {}).get(name)
        return None if p is None else np.array([float(p[j]) for j in ARM_JOINTS])

    def tip_xyz(self, motor_deg: np.ndarray) -> np.ndarray:
        return self.kin.fk(self.to_urdf(motor_deg))[:3, 3]

    def check_workspace(self, xyz: np.ndarray) -> str | None:
        ws = self.cfg["workspace"]
        r = float(np.linalg.norm(xyz[:2]))
        if r < ws["min_radius_m"]:
            return f"target is {r:.3f} m from the base, closer than workspace.min_radius_m={ws['min_radius_m']}"
        if r > ws["max_radius_m"]:
            return f"target is {r:.3f} m from the base, farther than workspace.max_radius_m={ws['max_radius_m']}"
        if xyz[0] < ws["min_x_m"]:
            return f"target x={xyz[0]:.3f} m is behind workspace.min_x_m={ws['min_x_m']}"
        return None

    def _ik(
        self, xyz: np.ndarray, yaw_rad: float | None, seed_motor: np.ndarray, pitches, allow_flip: bool = True
    ) -> IKResult:
        res = self.kin.solve_grasp(
            xyz, yaw_rad, self.to_urdf(seed_motor), pitches_deg=pitches, allow_flip=allow_flip
        )
        res.joints_deg = self.to_motor(res.joints_deg)
        return res

    def plan(self, floor_xyz: np.ndarray, yaw_deg: float | None, seed_motor: np.ndarray) -> GraspPlan:
        """floor_xyz: object position on the floor, base frame. yaw_deg: jaw OPENING direction, or None."""
        g = self.cfg["grasp"]
        target = np.asarray(floor_xyz, float) + [0.0, 0.0, g["grasp_height_m"]]
        if (err := self.check_workspace(target)) is not None:
            raise ValueError(err)
        yaw = None if yaw_deg is None else np.radians(yaw_deg)

        grasp = self._ik(target, yaw, seed_motor, pitches=(90, 80, 70, 60, 50, 40))
        if not grasp.ok():
            raise ValueError(
                f"IK failed for the grasp at {np.round(target, 3)}: position error "
                f"{grasp.pos_error_m * 1000:.0f} mm, approach error {grasp.approach_error_deg:.0f} deg, "
                f"yaw error {grasp.yaw_error_deg:.0f} deg. The point is probably out of reach."
            )
        # Keep the jaw orientation the grasp actually reached for the other waypoints: no wrist spin.
        held_yaw = self.kin.jaw_yaw(self.to_urdf(grasp.joints_deg))

        # Pre-grasp: back off along the approach direction so the descent is (nearly) a straight line.
        # Higher up, a vertical gripper can need more wrist_flex than the joint has, so allow the
        # approach to tilt a bit more there (and back off along that tilted direction).
        pre, pre_xyz = None, target
        for pitch in np.arange(grasp.approach_pitch_deg, 29.0, -10.0):
            p = np.radians(pitch)
            radial = target[:2] / np.linalg.norm(target[:2])
            approach = np.array([radial[0] * np.cos(p), radial[1] * np.cos(p), -np.sin(p)])
            pre_xyz = target - approach * g["approach_dist_m"]
            pre = self._ik(pre_xyz, held_yaw, grasp.joints_deg, pitches=(float(pitch),), allow_flip=False)
            if pre.ok():
                break
        lift_xyz = target + [0.0, 0.0, g["lift_height_m"]]
        lift = self._ik(
            lift_xyz,
            held_yaw,
            grasp.joints_deg,
            pitches=tuple(np.arange(grasp.approach_pitch_deg, 29.0, -10.0)),
            allow_flip=False,
        )
        for name, r in (("pre-grasp", pre), ("lift", lift)):
            if r is None or not r.ok(pos_tol_m=0.01, approach_tol_deg=15, yaw_tol_deg=25):
                err = "?" if r is None else f"{r.pos_error_m * 1000:.0f} mm off"
                raise ValueError(f"IK failed for the {name} point ({err}).")

        o, c = float(g["gripper_open"]), float(g["gripper_closed"])
        wps = [
            Waypoint("pre-grasp", pre.joints_deg, o, pre_xyz),
            Waypoint("descend", grasp.joints_deg, o, target),
            Waypoint("close", grasp.joints_deg, c, target),
            Waypoint("lift", lift.joints_deg, c, lift_xyz),
        ]
        place = self.pose("place")
        if place is not None:
            wps += [Waypoint("place", place, c), Waypoint("release", place, o)]
        home = self.pose("look")
        if home is not None:
            wps.append(Waypoint("home", home, c if place is None else o))
        return GraspPlan(target, yaw_deg, grasp, wps)


# ---------------------------------------------------------------------------- execution
def connect_follower(cfg):
    """Connect the follower (no camera, never starts an interactive calibration)."""
    follower = make_follower(cfg, with_camera=False)
    follower.connect(calibrate=False)
    if not follower.is_calibrated:
        follower.disconnect()
        raise SystemExit("Follower is not calibrated (or the calibration doesn't match). Run scripts/calibrate_arms.py.")
    return follower


def read_follower(follower) -> tuple[np.ndarray, float]:
    obs = follower.bus.sync_read("Present_Position")
    return np.array([obs[j] for j in ARM_JOINTS]), float(obs["gripper"])


def move_to(follower, joints_deg: np.ndarray, gripper: float, speed_deg_s: float, min_time_s: float = 0.4) -> None:
    """Smooth (min-jerk) joint-space move from the current position."""
    q0, g0 = read_follower(follower)
    q1 = np.asarray(joints_deg, float)
    span = max(np.max(np.abs(q1 - q0)), abs(gripper - g0) * 0.9)  # gripper units ~ degrees-ish
    duration = max(min_time_s, span / speed_deg_s)
    n = max(1, int(duration * CONTROL_HZ))
    for i in range(1, n + 1):
        t0 = time.perf_counter()
        s = i / n
        s = 10 * s**3 - 15 * s**4 + 6 * s**5
        q = q0 + (q1 - q0) * s
        g = g0 + (gripper - g0) * s
        action = {f"{j}.pos": float(v) for j, v in zip(ARM_JOINTS, q)}
        action["gripper.pos"] = float(g)
        follower.send_action(action)
        time.sleep(max(0.0, 1 / CONTROL_HZ - (time.perf_counter() - t0)))


def execute(follower, plan: GraspPlan, gcfg: dict[str, Any], log=print) -> bool:
    """Run the waypoints. Returns False (and stops after lifting the empty gripper) if nothing was grasped."""
    g = gcfg["grasp"]
    margin = float(g.get("empty_grasp_margin", 3.0))
    for i, wp in enumerate(plan.waypoints):
        speed = g["slow_speed_deg_s"] if wp.name in ("descend", "close") else g["speed_deg_s"]
        log(f"  -> {wp.name}")
        move_to(follower, wp.joints_deg, wp.gripper, speed)
        if wp.name in ("close", "release"):
            time.sleep(g["gripper_wait_s"])
        if wp.name == "close":
            # An object between the jaws stops the gripper before its closed target.
            _, grip = read_follower(follower)
            if abs(grip - float(g["gripper_closed"])) <= margin:
                log(f"  gripper closed all the way ({grip:.1f}): missed the object. Lifting and stopping.")
                lift = next((w for w in plan.waypoints[i + 1 :] if w.name == "lift"), None)
                if lift is not None:
                    move_to(follower, lift.joints_deg, float(g["gripper_open"]), g["speed_deg_s"])
                return False
            log(f"  gripper stopped at {grip:.1f}: holding something.")
    return True
