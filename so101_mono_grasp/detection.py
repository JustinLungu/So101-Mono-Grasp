"""Object detection, baseline: HSV color thresholding on a brightly colored object. No training needed.

Returns everything the grasp needs in pixels:
  - centroid_px: the blob's center, i.e. roughly the object's middle (pick.py intersects its ray with a
                 plane at half the object's height to get the grasp point)
  - axis_px:     two pixels along the object's long axis (from the min-area rectangle), for the jaw yaw
  - contact_px:  bottom-center, the point closest to where the object touches the floor (debug only)
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    contour: np.ndarray
    bbox: tuple[int, int, int, int]  # x, y, w, h
    area_px: float
    centroid_px: tuple[float, float]
    contact_px: tuple[float, float]
    axis_px: tuple[tuple[float, float], tuple[float, float]]  # long-axis endpoints
    elongation: float  # long side / short side of the min-area rectangle (1 = round / square)


def hsv_mask(frame_rgb: np.ndarray, ranges: list[dict]) -> np.ndarray:
    """ranges: [{lower: [h, s, v], upper: [h, s, v]}, ...] in OpenCV HSV (H 0-179). Red needs two ranges."""
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for r in ranges:
        mask |= cv2.inRange(hsv, np.array(r["lower"], np.uint8), np.array(r["upper"], np.uint8))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect(frame_rgb: np.ndarray, ranges: list[dict], min_area_px: float = 300) -> Detection | None:
    """The largest blob matching the color, or None."""
    mask = hsv_mask(frame_rgb, ranges)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area_px]
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    m = cv2.moments(c)
    centroid = (m["m10"] / m["m00"], m["m01"] / m["m00"])

    # Bottom-center: lowest contour points (largest v), averaged in u.
    pts = c.reshape(-1, 2)
    bottom = pts[pts[:, 1] >= pts[:, 1].max() - 2]
    contact = (float(bottom[:, 0].mean()), float(pts[:, 1].max()))

    (rcx, rcy), (rw, rh), angle = cv2.minAreaRect(c)
    long_len, ang = (rw, angle) if rw >= rh else (rh, angle + 90.0)
    d = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))]) * long_len / 2
    axis = ((rcx - d[0], rcy - d[1]), (rcx + d[0], rcy + d[1]))
    elong = max(rw, rh) / max(min(rw, rh), 1e-6)

    return Detection(c, (x, y, w, h), float(cv2.contourArea(c)), centroid, contact, axis, float(elong))


def draw(frame_rgb: np.ndarray, det: Detection | None, text: str = "") -> np.ndarray:
    """Annotated BGR copy for saving with cv2.imwrite."""
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if det is not None:
        cv2.drawContours(img, [det.contour], -1, (0, 255, 0), 2)
        x, y, w, h = det.bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 200, 0), 1)
        (a0, a1) = det.axis_px
        cv2.line(img, tuple(map(int, a0)), tuple(map(int, a1)), (255, 0, 255), 2)
        cv2.circle(img, tuple(map(int, det.contact_px)), 4, (0, 165, 255), -1)
        cv2.drawMarker(img, tuple(map(int, det.centroid_px)), (0, 0, 255), cv2.MARKER_CROSS, 18, 3)
    if text:
        for i, line in enumerate(text.splitlines()):
            cv2.putText(img, line, (10, 25 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return img
