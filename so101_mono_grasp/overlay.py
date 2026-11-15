"""Drawing helpers for the calibration / debug views (all images RGB, drawn in place)."""

from __future__ import annotations

import cv2
import numpy as np

from .ground_plane import FloorProjector, in_view, project_arm_points

GREEN, RED, YELLOW, CYAN, MAGENTA, WHITE = (0, 220, 0), (255, 40, 40), (255, 230, 0), (0, 220, 255), (255, 0, 255), (255, 255, 255)


def text(img: np.ndarray, s: str, org: tuple[int, int], color=YELLOW, scale: float = 0.55) -> None:
    for i, line in enumerate(s.splitlines()):
        p = (int(org[0]), int(org[1] + 20 * i))
        cv2.putText(img, line, p, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, p, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _polyline(img, proj: FloorProjector, pts_arm: np.ndarray, color, thickness=1) -> None:
    px = project_arm_points(proj, pts_arm)
    if px is None or not np.all(np.isfinite(px)):
        return
    h, w = img.shape[:2]
    if np.all((px[:, 0] < -w) | (px[:, 0] > 2 * w) | (px[:, 1] < -h) | (px[:, 1] > 2 * h)):
        return
    cv2.polylines(img, [np.round(px).astype(np.int32)], False, color, thickness, cv2.LINE_AA)


def floor_grid(
    img: np.ndarray,
    proj: FloorProjector,
    step: float = 0.05,
    x_range: tuple[float, float] = (0.0, 0.45),
    y_range: tuple[float, float] = (-0.25, 0.25),
    reach: tuple[float, float] | None = None,
) -> None:
    """Arm-frame grid on the floor (5 cm squares by default), axes highlighted, optional reach ring."""
    z = proj.g2b.z
    xs = np.arange(x_range[0], x_range[1] + 1e-9, step)
    ys = np.arange(y_range[0], y_range[1] + 1e-9, step)
    # Sample each line densely so lens distortion bends it correctly, and keep only points in front of the camera.
    for x in xs:
        _segments(img, proj, np.column_stack([np.full(40, x), np.linspace(*y_range, 40), np.full(40, z)]),
                  CYAN if abs(x) < 1e-9 else WHITE)
    for y in ys:
        _segments(img, proj, np.column_stack([np.linspace(*x_range, 40), np.full(40, y), np.full(40, z)]),
                  RED if abs(y) < 1e-9 else WHITE)
    if reach is not None:
        a = np.linspace(-np.pi / 2, np.pi / 2, 60)
        for r in reach:
            _segments(img, proj, np.column_stack([r * np.cos(a), r * np.sin(a), np.full(60, z)]), GREEN, 2)
    for x in xs[1::2]:
        p = np.array([[x, 0.0, z]])
        if in_view(proj, p, margin=1.0)[0]:
            text(img, f"{x * 100:.0f}", tuple(project_arm_points(proj, p)[0] + [4, -4]), WHITE, 0.4)


def _segments(img, proj, pts, color, thickness=1) -> None:
    """Draw a polyline, splitting it where points leave the camera's field of view."""
    visible = in_view(proj, pts)
    good: list[np.ndarray] = []
    for p, ok in zip(pts, visible):
        if ok:
            good.append(p)
            continue
        if len(good) > 1:
            _polyline(img, proj, np.array(good), color, thickness)
        good = []
    if len(good) > 1:
        _polyline(img, proj, np.array(good), color, thickness)


def cross(img: np.ndarray, uv, color=RED, size: int = 14, thickness: int = 2) -> None:
    cv2.drawMarker(img, (round(float(uv[0])), round(float(uv[1]))), color, cv2.MARKER_CROSS, size, thickness)


class Layer:
    """A static overlay (e.g. the floor grid) rendered once and pasted onto every frame."""

    def __init__(self, shape: tuple[int, ...], draw_fn):
        self.img = np.zeros((shape[0], shape[1], 3), np.uint8)
        draw_fn(self.img)
        self.mask = self.img.any(axis=2)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        frame[self.mask] = self.img[self.mask]
        return frame
