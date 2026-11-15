"""Task 2.2: where the camera is relative to the arm and the floor (h, tilt, roll + mount position/yaw).

Needs calib/intrinsics.yaml first (scripts/calibrate_intrinsics.py).

1. Put the camera where it will be when picking. Camera on the arm: the arm must be in the `look` pose
   of config/grasp.yaml (--move-to-look does that and holds it there).
2. Put 5-8 small markers (tape crosses, coins) on the floor, SPREAD OUT: near and far, left and right,
   covering the area where you will pick.
3. Measure each one with a tape in the ARM FRAME:
      origin = the floor point directly below the shoulder rotation axis (shoulder_pan)
      x = forward (the direction the arm points with shoulder_pan = 0), y = to the LEFT, in cm
4. Click a marker in the window, type its "x y" in cm. After 4 markers the fit runs on every click:
   the floor grid (5 cm squares, red line = x axis, cyan = y axis, green = reach) should lie on the
   real floor, and the per-marker errors should be a few mm.
5. S saves calib/ground.yaml and `camera_ground_in_base` in config/grasp.yaml.   U undo   G grid   Q quit

If the arm stands on a different surface than the objects (quadruped), pass --floor-z: the floor height
in the arm frame, e.g. --floor-z -0.30 when the floor is 30 cm below the arm's base.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from so101_mono_grasp import overlay
from so101_mono_grasp.arms import DEFAULT_CONFIG, load_config
from so101_mono_grasp.camera import Camera, Intrinsics
from so101_mono_grasp.grasp import DEFAULT_GRASP_CONFIG, load_grasp_config, resolve
from so101_mono_grasp.ground_plane import FloorProjector, PoseFit, fit_camera_pose


def parse_xy_cm(s: str | None) -> np.ndarray | None:
    if not s:
        return None
    nums = re.findall(r"[-+]?\d+(?:[.,]\d+)?", s)
    if len(nums) != 2:
        return None
    return np.array([float(n.replace(",", ".")) for n in nums]) / 100.0


def write_camera_mount(grasp_cfg_path: Path, fit: PoseFit) -> None:
    """Replace the camera_ground_in_base line in grasp.yaml, keeping every comment."""
    g = fit.g2b
    line = f"camera_ground_in_base: {{x: {g.x:.4f}, y: {g.y:.4f}, z: {g.z:.4f}, yaw_deg: {g.yaw_deg:.2f}}}"
    text = grasp_cfg_path.read_text()
    new, n = re.subn(r"^camera_ground_in_base:.*$", line, text, count=1, flags=re.MULTILINE)
    if n != 1:
        raise RuntimeError(f"No camera_ground_in_base line in {grasp_cfg_path}")
    grasp_cfg_path.write_text(new)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG, help="arms config (ports, camera)")
    p.add_argument("--grasp-config", default=DEFAULT_GRASP_CONFIG)
    p.add_argument("--floor-z", type=float, help="floor height in the arm frame [m] (default: from grasp.yaml)")
    p.add_argument("--move-to-look", action="store_true", help="move the follower to poses.look and hold it there")
    args = p.parse_args()

    cfg = load_config(args.config)
    gcfg = load_grasp_config(args.grasp_config)
    intr_path = resolve(gcfg["intrinsics"])
    if not intr_path.exists():
        sys.exit(f"{intr_path} is missing: run scripts/calibrate_intrinsics.py first.")
    intr = Intrinsics.load(intr_path)
    floor_z = args.floor_z if args.floor_z is not None else float(gcfg["camera_ground_in_base"]["z"])
    ws = gcfg["workspace"]

    follower = None
    if args.move_to_look:
        from so101_mono_grasp.grasp import GraspPlanner, connect_follower, move_to

        look = GraspPlanner(gcfg).pose("look")
        if look is None:
            sys.exit("poses.look is not set in grasp.yaml.")
        follower = connect_follower(cfg)
        input(f"The follower will move to the look pose {look.tolist()}. Stay clear. [Enter] ")
        move_to(follower, look, float(gcfg["grasp"]["gripper_open"]), gcfg["grasp"]["speed_deg_s"])
        time.sleep(0.5)

    from so101_mono_grasp.viewer import Viewer

    cam = Camera.from_config(cfg.camera)
    try:
        intr.check_frame(np.zeros((cam.height, cam.width, 3)))
    except ValueError as e:
        cam.release()
        sys.exit(str(e))

    view = Viewer("Ground calibration: click a floor marker, type its x y in cm")
    pixels: list[tuple[float, float]] = []
    points: list[np.ndarray] = []
    fit: PoseFit | None = None
    grid: overlay.Layer | None = None
    show_grid = True
    msg = "Click the first marker."
    try:
        while view.alive:
            frame = cam.read()
            keys = view.keys()
            for u, v in view.clicks():
                s = view.ask("Marker position", f"Pixel ({u:.0f}, {v:.0f})\nx y in cm (x forward, y LEFT), e.g.  25 -10")
                xy = parse_xy_cm(s)
                if xy is None:
                    msg = "Skipped (expected two numbers in cm)."
                    continue
                pixels.append((u, v))
                points.append(xy)
                msg = f"marker {len(points)}: ({xy[0] * 100:.1f}, {xy[1] * 100:.1f}) cm"
                fit, grid = None, None
            if "u" in keys and points:
                pixels.pop()
                points.pop()
                fit, grid = None, None
                msg = f"removed last marker ({len(points)} left)"
            if "g" in keys:
                show_grid = not show_grid

            if fit is None and len(points) >= 4:
                try:
                    fit = fit_camera_pose(intr, np.array(pixels), np.array(points), floor_z)
                    proj = FloorProjector(intr, fit.ground, fit.g2b)
                    grid = overlay.Layer(
                        frame.shape,
                        lambda img, proj=proj: overlay.floor_grid(
                            img, proj, reach=(ws["min_radius_m"], ws["max_radius_m"])
                        ),
                    )
                except ValueError as e:
                    msg = f"fit failed: {e}"

            if "s" in keys:
                if fit is None:
                    msg = "Need at least 4 markers and a successful fit before saving."
                else:
                    gpath = resolve(gcfg["ground"])
                    fit.ground.save(
                        gpath,
                        markers=[
                            {"px": [round(u, 1), round(v, 1)], "xy_cm": [round(x * 100, 1), round(y * 100, 1)]}
                            for (u, v), (x, y) in zip(pixels, points)
                        ],
                        max_error_mm=round(float(fit.errors_m.max() * 1000), 1),
                        floor_z=floor_z,
                        date=datetime.now().isoformat(timespec="seconds"),
                    )
                    write_camera_mount(Path(args.grasp_config), fit)
                    print(f"Saved {gpath} and camera_ground_in_base in {args.grasp_config}")
                    msg = "SAVED. Q to quit."
            if "q" in keys or "Escape" in keys:
                break

            show = frame.copy()
            if grid is not None and show_grid:
                grid.apply(show)
            for i, ((u, v), xy) in enumerate(zip(pixels, points)):
                err = "" if fit is None else f" {fit.errors_m[i] * 1000:.0f}mm"
                overlay.cross(show, (u, v), overlay.MAGENTA)
                overlay.text(show, f"{i + 1}:({xy[0] * 100:.0f},{xy[1] * 100:.0f}){err}", (u + 8, v - 8), overlay.MAGENTA)
            if fit is not None:
                g, t = fit.ground, fit.g2b
                summary = (
                    f"h={g.h * 100:.1f}cm tilt={g.tilt_deg:.1f} roll={g.roll_deg:.1f} | camera floor point "
                    f"({t.x * 100:.1f}, {t.y * 100:.1f}) cm, yaw {t.yaw_deg:.1f} | marker error "
                    f"mean {fit.errors_m.mean() * 1000:.1f} max {fit.errors_m.max() * 1000:.1f} mm"
                )
            else:
                summary = f"{len(points)} markers (need 4+)"
            view.show(show, f"{summary}\nclick = add marker  U undo  G grid  S save  Q quit | {msg}")
    finally:
        cam.release()
        view.close()
        if follower is not None:
            print("Torque off: hold the arm if it isn't resting.")
            follower.disconnect()


if __name__ == "__main__":
    main()
