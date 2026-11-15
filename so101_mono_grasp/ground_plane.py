"""Depth from one camera: intersect the pixel's viewing ray with the floor, then move it into the arm base frame.

Frames
------
Ground frame G: origin on the floor directly under the camera lens, X forward (the camera's viewing
direction projected on the floor), Y left, Z up.
Arm frame B: origin below the shoulder rotation axis, x forward, y left, z up (see kinematics.py).

Both of these are written by scripts/calibrate_ground.py (fit_camera_pose below): click >= 4 floor
markers whose (x, y) in the arm frame you measured with a tape.

calib/ground.yaml:

    h: 0.45          # camera lens height above the floor [m]
    tilt_deg: 35.0   # how far the optical axis points below horizontal
    roll_deg: 0.0    # rotation about the optical axis (positive = camera turned clockwise, seen from behind it)

The G -> B transform lives in config/grasp.yaml (`camera_ground_in_base`): where the floor point under
the camera is in the arm frame, and which way the camera looks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from .camera import Intrinsics

# Camera axes (x right, y down, z forward) expressed in G when tilt = roll = 0.
_R0 = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@dataclass
class GroundCalib:
    h: float
    tilt_deg: float
    roll_deg: float = 0.0

    @classmethod
    def load(cls, path: str | Path) -> GroundCalib:
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(h=float(d["h"]), tilt_deg=float(d["tilt_deg"]), roll_deg=float(d.get("roll_deg", 0.0)))

    def save(self, path: str | Path, **extra) -> None:
        d = {"h": round(self.h, 5), "tilt_deg": round(self.tilt_deg, 3), "roll_deg": round(self.roll_deg, 3)}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump({**d, **extra}, f, sort_keys=False)

    def R_ground_cam(self, extra_pitch_deg: float = 0.0) -> np.ndarray:
        """Rotation from the camera frame to G. extra_pitch_deg: e.g. the quadruped's IMU pitch change."""
        tilt = np.radians(self.tilt_deg + extra_pitch_deg)
        return _rot_y(tilt) @ _R0 @ _rot_z(np.radians(self.roll_deg))


@dataclass
class GroundToBase:
    """Pose of G in the arm base frame: floor point under the camera at (x, y, z), camera looking along yaw."""

    x: float
    y: float
    z: float  # floor height in the base frame (negative if the arm base is above the floor)
    yaw_deg: float

    @classmethod
    def from_dict(cls, d: dict) -> GroundToBase:
        return cls(float(d["x"]), float(d["y"]), float(d["z"]), float(d.get("yaw_deg", 0.0)))

    def apply(self, p_ground: np.ndarray) -> np.ndarray:
        return _rot_z(np.radians(self.yaw_deg)) @ np.asarray(p_ground, float) + [self.x, self.y, self.z]


class FloorProjector:
    def __init__(self, intr: Intrinsics, ground: GroundCalib, ground_to_base: GroundToBase):
        self.intr = intr
        self.ground = ground
        self.g2b = ground_to_base

    def pixel_to_ground(
        self, u: float, v: float, plane_height: float = 0.0, extra_pitch_deg: float = 0.0
    ) -> np.ndarray | None:
        """Where the pixel's ray hits the horizontal plane `plane_height` m above the floor, in G.

        Returns (X, Y, plane_height), or None if the ray never reaches that plane (at/above the horizon).
        Use plane_height = half the object's height with the blob's centroid: the centroid of an object
        in the image is (roughly) its middle, not a point on the floor.
        """
        d = self.ground.R_ground_cam(extra_pitch_deg) @ self.intr.pixel_to_ray(u, v)
        drop = self.ground.h - plane_height
        if d[2] > -1e-3 or drop <= 0:
            return None
        return np.array([0.0, 0.0, self.ground.h]) + (drop / -d[2]) * d

    def pixel_to_base(
        self, u: float, v: float, plane_height: float = 0.0, extra_pitch_deg: float = 0.0
    ) -> np.ndarray | None:
        p = self.pixel_to_ground(u, v, plane_height, extra_pitch_deg)
        return None if p is None else self.g2b.apply(p)

    def camera_forward_in_base(self) -> np.ndarray:
        """Horizontal unit vector of the camera's viewing direction, in the base frame."""
        return _rot_z(np.radians(self.g2b.yaw_deg)) @ np.array([1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------- calibration
@dataclass
class PoseFit:
    ground: GroundCalib
    g2b: GroundToBase
    errors_m: np.ndarray  # per marker: |clicked pixel projected to the floor - measured position|
    reproj_px: np.ndarray  # per marker: |measured position projected into the image - clicked pixel|


def fit_camera_pose(
    intr: Intrinsics, pixels: np.ndarray, floor_xy: np.ndarray, floor_z: float = 0.0
) -> PoseFit:
    """Camera pose from >= 4 floor markers: pixel (u, v) <-> measured (x, y) in the arm frame, on z = floor_z.

    Spread the markers out (near/far AND left/right) or the tilt/roll are poorly determined.
    """
    pixels = np.asarray(pixels, np.float64).reshape(-1, 2)
    floor_xy = np.asarray(floor_xy, np.float64).reshape(-1, 2)
    n = len(pixels)
    if n < 4:
        raise ValueError("Need at least 4 markers.")
    centered = floor_xy - floor_xy.mean(axis=0)
    if np.linalg.svd(centered, compute_uv=False)[-1] < 0.02 * np.sqrt(n):
        raise ValueError("Markers are (nearly) on one line. Spread them out in both x and y.")

    # Solve with the floor at z = 0 (IPPE wants the plane there), then shift by floor_z at the end.
    obj = np.column_stack([floor_xy, np.zeros(n)])
    # A planar target has two mirror-image PnP solutions; keep the one with the camera above the floor.
    n_sol, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, pixels, intr.K, intr.dist, flags=cv2.SOLVEPNP_IPPE)
    best = None
    for rv, tv in zip(rvecs[:n_sol], tvecs[:n_sol]):
        rv, tv = cv2.solvePnPRefineLM(obj, pixels, intr.K, intr.dist, rv, tv)
        R = cv2.Rodrigues(rv)[0]
        if (-R.T @ tv).ravel()[2] <= 0.0:
            continue
        err = np.linalg.norm(cv2.projectPoints(obj, rv, tv, intr.K, intr.dist)[0].reshape(-1, 2) - pixels)
        if best is None or err < best[0]:
            best = (err, rv, tv)
    if best is None:
        raise ValueError(
            "No solution with the camera above the floor: check the measured coordinates (x forward, y LEFT)."
        )
    _, rvec, tvec = best
    R_cam_arm, _ = cv2.Rodrigues(rvec)
    R_arm_cam = R_cam_arm.T
    C = (-R_arm_cam @ tvec).ravel()  # camera center, arm frame (floor at z = 0 for now)

    fwd = R_arm_cam[:, 2]  # optical axis
    tilt = np.degrees(np.arcsin(np.clip(-fwd[2], -1, 1)))
    # Looking exactly straight down: the optical axis has no heading; use the image "up" direction.
    heading = fwd if np.linalg.norm(fwd[:2]) > 1e-4 else -R_arm_cam[:, 1]
    yaw = np.degrees(np.arctan2(heading[1], heading[0]))
    R_g_cam = _rot_z(-np.radians(yaw)) @ R_arm_cam
    M = _R0.T @ _rot_y(np.radians(tilt)).T @ R_g_cam  # = Rz(roll)
    roll = np.degrees(np.arctan2(M[1, 0], M[0, 0]))

    ground = GroundCalib(h=float(C[2]), tilt_deg=float(tilt), roll_deg=float(roll))
    g2b = GroundToBase(float(C[0]), float(C[1]), float(floor_z), float(yaw))

    proj = FloorProjector(intr, ground, g2b)
    errs = []
    for (u, v), xy in zip(pixels, floor_xy):
        p = proj.pixel_to_base(u, v)
        errs.append(np.inf if p is None else np.linalg.norm(p[:2] - xy))
    img_pts, _ = cv2.projectPoints(obj, rvec, tvec, intr.K, intr.dist)
    reproj = np.linalg.norm(img_pts.reshape(-1, 2) - pixels, axis=1)
    return PoseFit(ground, g2b, np.array(errs), reproj)


def arm_to_camera(proj: FloorProjector, pts_arm: np.ndarray) -> np.ndarray:
    """Arm-frame 3D points -> OpenCV camera frame (x right, y down, z forward)."""
    pts_arm = np.asarray(pts_arm, float).reshape(-1, 3)
    g = proj.g2b
    pts_g = (_rot_z(-np.radians(g.yaw_deg)) @ (pts_arm - [g.x, g.y, g.z]).T).T
    R_g_cam = proj.ground.R_ground_cam()
    return (R_g_cam.T @ (pts_g - [0.0, 0.0, proj.ground.h]).T).T


def in_view(proj: FloorProjector, pts_arm: np.ndarray, margin: float = 1.3) -> np.ndarray:
    """Boolean per point: in front of the camera and within `margin` x the image's field of view.

    Lens-distortion formulas give nonsense far outside the calibrated field of view, so overlays
    must not draw such points.
    """
    c = arm_to_camera(proj, pts_arm)
    z = c[:, 2]
    i = proj.intr
    with np.errstate(divide="ignore", invalid="ignore"):
        ok_x = np.abs(c[:, 0] / z - (i.width / 2 - i.cx) / i.fx) < margin * i.width / (2 * i.fx)
        ok_y = np.abs(c[:, 1] / z - (i.height / 2 - i.cy) / i.fy) < margin * i.height / (2 * i.fy)
    return (z > 1e-3) & ok_x & ok_y


def project_arm_points(proj: FloorProjector, pts_arm: np.ndarray) -> np.ndarray | None:
    """Arm-frame 3D points -> pixels (with distortion), for drawing overlays. None if behind the camera."""
    pts_cam = arm_to_camera(proj, pts_arm)
    if np.any(pts_cam[:, 2] <= 1e-3):
        return None
    px, _ = cv2.projectPoints(pts_cam, np.zeros(3), np.zeros(3), proj.intr.K, proj.intr.dist)
    return px.reshape(-1, 2)


def projector_from_config(gcfg: dict) -> FloorProjector:
    """FloorProjector from config/grasp.yaml (intrinsics + ground files and camera_ground_in_base).

    Raises FileNotFoundError naming the missing calibration file(s).
    """
    from .arms import REPO_ROOT

    def resolve(p: str) -> Path:
        q = Path(p).expanduser()
        return q if q.is_absolute() else REPO_ROOT / q

    paths = {k: resolve(gcfg[k]) for k in ("intrinsics", "ground")}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Camera calibration missing: " + ", ".join(missing) + "\n"
            "Run scripts/calibrate_intrinsics.py, then scripts/calibrate_ground.py."
        )
    return FloorProjector(
        Intrinsics.load(paths["intrinsics"]),
        GroundCalib.load(paths["ground"]),
        GroundToBase.from_dict(gcfg["camera_ground_in_base"]),
    )
