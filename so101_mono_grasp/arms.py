"""Build lerobot's SO-101 follower and leader from config/arms.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "arms.yaml"

# Motor names and IDs on both arms, as lerobot defines them.
MOTORS = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}


@dataclass
class ArmsConfig:
    raw: dict[str, Any]

    @property
    def follower(self) -> dict[str, Any]:
        return self.raw["follower"]

    @property
    def leader(self) -> dict[str, Any]:
        return self.raw["leader"]

    @property
    def camera(self) -> dict[str, Any]:
        return self.raw.get("camera", {"enabled": False})

    @property
    def teleop_fps(self) -> int:
        return int(self.raw.get("teleop", {}).get("fps", 60))


def _resolve(path: str | None) -> Path | None:
    if path is None:
        return None
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def load_config(path: str | Path = DEFAULT_CONFIG) -> ArmsConfig:
    with open(path) as f:
        return ArmsConfig(yaml.safe_load(f))


def make_follower(cfg: ArmsConfig, with_camera: bool = True) -> SO101Follower:
    f = cfg.follower
    cameras = {}
    cam = cfg.camera
    if with_camera and cam.get("enabled", False):
        cameras["front"] = OpenCVCameraConfig(
            index_or_path=cam["index_or_path"],
            width=cam["width"],
            height=cam["height"],
            fps=cam["fps"],
        )
    return SO101Follower(
        SO101FollowerConfig(
            port=f["port"],
            id=f["id"],
            calibration_dir=_resolve(f.get("calibration_dir")),
            max_relative_target=f.get("max_relative_target"),
            cameras=cameras,
        )
    )


def make_leader(cfg: ArmsConfig) -> SO101Leader:
    l = cfg.leader  # noqa: E741
    return SO101Leader(
        SO101LeaderConfig(
            port=l["port"],
            id=l["id"],
            calibration_dir=_resolve(l.get("calibration_dir")),
        )
    )
