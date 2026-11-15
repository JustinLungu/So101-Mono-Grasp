"""Task 2.1: camera intrinsics (fx, fy, cx, cy, distortion) from a printed checkerboard.

    python scripts/calibrate_intrinsics.py --cols 9 --rows 6 --square 0.025

--cols / --rows count INNER corners (a board of 10x7 squares has 9x6 inner corners).
--square is the side of one square in meters: measure the PRINT with a ruler, printers rescale.
Glue the print on something flat (cardboard, a book cover): a bent board ruins the calibration.

Live window:
  - the board is detected live (colored corners). With auto-capture on, a view is taken when the board
    is steady and in a NEW place/angle; SPACE takes one by hand.
  - fill the whole image: corners, edges, near, far, and tilt the board up to ~45 deg in every direction.
    The dots show where you already have corners.
  - C: calibrate (needs --min-views, default 15) and save calib/intrinsics.yaml
  - A: auto-capture on/off   U: undo last view   Q / Esc: quit

Re-run offline on the saved frames:  python scripts/calibrate_intrinsics.py --images "outputs/intrinsics_frames/*.png"

Uses the resolution in config/arms.yaml (camera.width/height): calibrate at the resolution you pick at.
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from so101_mono_grasp import overlay
from so101_mono_grasp.arms import DEFAULT_CONFIG, REPO_ROOT, load_config
from so101_mono_grasp.camera import Camera, Intrinsics

FRAMES_DIR = REPO_ROOT / "outputs" / "intrinsics_frames"


def find_board(gray: np.ndarray, size: tuple[int, int]) -> np.ndarray | None:
    found, corners = cv2.findChessboardCornersSB(gray, size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    return corners.reshape(-1, 2) if found else None


def board_points(cols: int, rows: int, square: float) -> np.ndarray:
    obj = np.zeros((rows * cols, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return obj


def calibrate(views: list[np.ndarray], obj: np.ndarray, wh: tuple[int, int]):
    """Calibrate; drop views whose error is > 3x the median once, then recalibrate."""
    def run(vs):
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera([obj] * len(vs), [v.astype(np.float32) for v in vs], wh, None, None)
        per_view = []
        for v, rv, tv in zip(vs, rvecs, tvecs):
            proj, _ = cv2.projectPoints(obj, rv, tv, K, dist)
            per_view.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - v) ** 2, axis=1)))))
        return rms, K, dist, np.array(per_view)

    rms, K, dist, per_view = run(views)
    keep = per_view <= max(3 * np.median(per_view), 0.5)
    dropped = int((~keep).sum())
    if dropped and keep.sum() >= 10:
        views = [v for v, k in zip(views, keep) if k]
        rms, K, dist, per_view = run(views)
    else:
        dropped = 0
    return rms, K, dist, per_view, dropped


def save(K, dist, wh, rms, n_views, dropped, args) -> Intrinsics:
    intr = Intrinsics(wh[0], wh[1], K[0, 0], K[1, 1], K[0, 2], K[1, 2], dist.ravel())
    out = Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out
    intr.save(
        out,
        rms_px=round(float(rms), 4),
        views=n_views,
        dropped_views=dropped,
        board={"cols": args.cols, "rows": args.rows, "square_m": args.square},
        date=datetime.now().isoformat(timespec="seconds"),
    )
    hfov = np.degrees(2 * np.arctan(wh[0] / (2 * K[0, 0])))
    vfov = np.degrees(2 * np.arctan(wh[1] / (2 * K[1, 1])))
    quality = "good" if rms < 0.5 else "OK" if rms < 1.0 else "POOR: re-do it (flat board? sharp images? right --square?)"
    print(
        f"\nSaved {out}\n  RMS reprojection error {rms:.3f} px ({quality}), {n_views} views"
        f"{f', {dropped} bad view(s) dropped' if dropped else ''}\n"
        f"  fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f}  FOV {hfov:.0f}x{vfov:.0f} deg\n"
        f"  dist={np.round(dist.ravel(), 4).tolist()}"
    )
    return intr


def offline(args, size, obj) -> None:
    files = sorted(glob.glob(args.images))
    views, wh = [], None
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        wh = wh or (img.shape[1], img.shape[0])
        c = find_board(img, size)
        print(f"{'ok  ' if c is not None else 'MISS'} {f}")
        if c is not None:
            views.append(c)
    if len(views) < args.min_views:
        sys.exit(f"Only {len(views)} usable images, need {args.min_views}.")
    rms, K, dist, _, dropped = calibrate(views, obj, wh)
    save(K, dist, wh, rms, len(views) - dropped, dropped, args)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--cols", type=int, default=9, help="inner corners per row")
    p.add_argument("--rows", type=int, default=6, help="inner corners per column")
    p.add_argument("--square", type=float, default=0.025, help="square side in meters (measure the print!)")
    p.add_argument("--min-views", type=int, default=15)
    p.add_argument("--out", default="calib/intrinsics.yaml")
    p.add_argument("--images", help="glob of saved images: calibrate offline, no camera")
    args = p.parse_args()

    size = (args.cols, args.rows)
    obj = board_points(args.cols, args.rows, args.square)
    if args.images:
        offline(args, size, obj)
        return

    from so101_mono_grasp.viewer import Viewer

    cfg = load_config(args.config)
    cam = Camera.from_config(cfg.camera)
    wh = (cam.width, cam.height)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for old in FRAMES_DIR.glob("*.png"):
        old.unlink()

    view = Viewer("Intrinsics calibration")
    views: list[np.ndarray] = []
    auto = True
    prev: np.ndarray | None = None
    still_since = 0.0
    last_shot = 0.0
    msg = f"Show the {args.cols}x{args.rows} board. Fill the whole image, tilt it."
    try:
        while view.alive:
            frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            corners = find_board(gray, size)
            show = frame.copy()
            for v in views:  # coverage so far
                for u, w in v[:: max(1, len(v) // 12)]:
                    cv2.circle(show, (int(u), int(w)), 2, overlay.GREEN, -1)

            new_place = False
            now = time.time()
            if corners is not None:
                cv2.drawChessboardCorners(show, size, corners.reshape(-1, 1, 2).astype(np.float32), True)
                moving = prev is None or np.mean(np.linalg.norm(corners - prev, axis=1)) > 1.5
                still_since = now if moving else still_since
                diffs = [np.mean(np.linalg.norm(corners - v, axis=1)) for v in views]
                new_place = not diffs or min(diffs) > 0.06 * wh[0]
            prev = corners

            keys = view.keys()
            take = "space" in keys or (
                auto and corners is not None and new_place and now - still_since > 0.4 and now - last_shot > 1.0
            )
            if take and corners is not None:
                if new_place or "space" in keys:
                    views.append(corners)
                    cv2.imwrite(str(FRAMES_DIR / f"{len(views):03d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    last_shot = now
                    msg = f"view {len(views)} captured"
                    show[:] = 255 - show  # flash
            elif "space" in keys:
                msg = "no board detected"
            if "a" in keys:
                auto = not auto
                msg = f"auto-capture {'ON' if auto else 'OFF'}"
            if "u" in keys and views:
                views.pop()
                msg = f"removed last view ({len(views)} left)"
            if "c" in keys:
                if len(views) < args.min_views:
                    msg = f"need {args.min_views} views, have {len(views)}"
                else:
                    view.show(show, "calibrating...")
                    rms, K, dist, _, dropped = calibrate(views, obj, wh)
                    save(K, dist, wh, rms, len(views) - dropped, dropped, args)
                    msg = f"SAVED: RMS {rms:.3f} px. Q to quit, or add views and C again."
            if "q" in keys or "Escape" in keys:
                break

            status = "board OK" if corners is not None else "no board"
            view.show(
                show,
                f"{status} | views {len(views)}/{args.min_views} | auto {'ON' if auto else 'OFF'} | "
                f"SPACE capture  A auto  U undo  C calibrate+save  Q quit\n{msg}",
            )
    finally:
        cam.release()
        view.close()


if __name__ == "__main__":
    main()
