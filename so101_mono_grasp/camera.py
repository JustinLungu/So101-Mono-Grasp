"""Single RGB camera: opening it, intrinsics, undistortion, pixel -> viewing ray.

calib/intrinsics.yaml (written by scripts/calibrate_intrinsics.py):

    width: 640            # resolution the calibration was done at
    height: 480
    fx: 600.0
    fy: 600.0
    cx: 320.0
    cy: 240.0
    dist: [k1, k2, p1, p2, k3]   # OpenCV distortion coefficients (optional, default all zeros)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import cv2
import numpy as np
import yaml


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @classmethod
    def load(cls, path: str | Path) -> Intrinsics:
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(
            width=int(d["width"]),
            height=int(d["height"]),
            fx=float(d["fx"]),
            fy=float(d["fy"]),
            cx=float(d["cx"]),
            cy=float(d["cy"]),
            dist=np.asarray(d.get("dist", [0, 0, 0, 0, 0]), float),
        )

    def save(self, path: str | Path, **extra: Any) -> None:
        d = {
            "width": self.width,
            "height": self.height,
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "dist": [float(x) for x in np.ravel(self.dist)],
            **extra,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(d, f, sort_keys=False)

    def check_frame(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if (w, h) != (self.width, self.height):
            raise ValueError(
                f"Frame is {w}x{h} but the intrinsics were calibrated at {self.width}x{self.height}. "
                "Use the same resolution or re-run the intrinsics calibration."
            )

    def pixel_to_ray(self, u: float, v: float) -> np.ndarray:
        """Unit viewing ray in the OpenCV camera frame (x right, y down, z forward), distortion removed."""
        pt = np.array([[[u, v]]], dtype=np.float64)
        x, y = cv2.undistortPoints(pt, self.K, self.dist)[0, 0]
        ray = np.array([x, y, 1.0])
        return ray / np.linalg.norm(ray)


class Camera:
    """OpenCV capture with a fixed resolution and (when the driver allows it) autofocus OFF.

    Autofocus changes the focal length, which silently invalidates the intrinsics: keep it off for
    calibration AND for picking. read() returns RGB.
    """

    def __init__(self, index_or_path: int | str, width: int, height: int, fps: int = 30, warmup: int = 10):
        src = int(index_or_path) if str(index_or_path).isdigit() else str(index_or_path)
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2) if isinstance(src, int) else cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Can't open camera {index_or_path!r}. List cameras with: lerobot-find-cameras opencv "
                "(or ls /dev/video*), then set camera.index_or_path in config/arms.yaml."
            )
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (self.width, self.height) != (width, height):
            self.release()
            raise RuntimeError(f"Camera gives {self.width}x{self.height}, not the requested {width}x{height}.")
        for _ in range(warmup):  # let auto-exposure settle
            self.cap.read()

    @classmethod
    def from_config(cls, cam_cfg: dict[str, Any]) -> Camera:
        return cls(cam_cfg["index_or_path"], int(cam_cfg["width"]), int(cam_cfg["height"]), int(cam_cfg["fps"]))

    def read(self) -> np.ndarray:
        for _ in range(3):
            ok, bgr = self.cap.read()
            if ok:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            time.sleep(0.05)
        raise RuntimeError("Camera stopped delivering frames.")

    def release(self) -> None:
        self.cap.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.release()
