"""Task 2.3: check the camera calibration live, and tune the object detection.

    python scripts/click_to_floor.py

- Click anywhere on the floor: it prints and draws that point in the arm frame (cm). Put a tape there:
  the numbers should match within ~1 cm in the area where you pick.
- The 5 cm floor grid (red = x axis, cyan = y axis, green = reach of the arm) should lie on the real floor.
- D: object detection on/off (HSV colors from config/grasp.yaml): shows where pick.py would grasp,
  whether it is in reach, and the jaw direction.
- M: show the color mask instead of the image (white = "object color"), to tune detection.hsv_ranges.
- G: grid on/off   C: clear clicks   S: save a snapshot to outputs/   Q / Esc: quit

The camera must be where the ground calibration was done (camera on the arm: arm in the `look` pose;
pass --move-to-look to put it there).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
from pick import OUT_DIR, locate

from so101_mono_grasp import overlay
from so101_mono_grasp.arms import DEFAULT_CONFIG, load_config
from so101_mono_grasp.camera import Camera
from so101_mono_grasp.detection import hsv_mask
from so101_mono_grasp.grasp import DEFAULT_GRASP_CONFIG, GraspPlanner, connect_follower, load_grasp_config, move_to
from so101_mono_grasp.ground_plane import project_arm_points, projector_from_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG, help="arms config (ports, camera)")
    p.add_argument("--grasp-config", default=DEFAULT_GRASP_CONFIG)
    p.add_argument("--move-to-look", action="store_true", help="move the follower to poses.look and hold it there")
    args = p.parse_args()

    cfg = load_config(args.config)
    gcfg = load_grasp_config(args.grasp_config)
    try:
        proj = projector_from_config(gcfg)
    except FileNotFoundError as e:
        sys.exit(str(e))
    ws = gcfg["workspace"]
    half_h = gcfg["detection"]["object_half_height_m"]

    follower = None
    if args.move_to_look:
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
        proj.intr.check_frame(np.zeros((cam.height, cam.width, 3)))
    except ValueError as e:
        cam.release()
        sys.exit(str(e))
    grid = overlay.Layer(
        (cam.height, cam.width), lambda img: overlay.floor_grid(img, proj, reach=(ws["min_radius_m"], ws["max_radius_m"]))
    )
    view = Viewer("Click the floor: arm-frame coordinates")
    clicks: list[tuple[tuple[float, float], np.ndarray | None]] = []
    show_grid, detect_on, mask_on = True, False, False
    msg = "Click the floor."
    try:
        while view.alive:
            frame = cam.read()
            keys = view.keys()
            for u, v in view.clicks():
                pt = proj.pixel_to_base(u, v)
                clicks.append(((u, v), pt))
                msg = "above the horizon" if pt is None else f"floor point ({pt[0] * 100:.1f}, {pt[1] * 100:.1f}) cm"
                print(f"pixel ({u:.0f}, {v:.0f}) -> {msg}")
            if "g" in keys:
                show_grid = not show_grid
            if "d" in keys:
                detect_on = not detect_on
            if "m" in keys:
                mask_on = not mask_on
            if "c" in keys:
                clicks.clear()
            if "q" in keys or "Escape" in keys:
                break

            if mask_on:
                show = cv2.cvtColor(hsv_mask(frame, gcfg["detection"]["hsv_ranges"]), cv2.COLOR_GRAY2RGB)
            else:
                show = frame.copy()
            if show_grid:
                grid.apply(show)
            for (u, v), pt in clicks:
                overlay.cross(show, (u, v), overlay.MAGENTA)
                if pt is not None:
                    overlay.text(show, f"({pt[0] * 100:.1f}, {pt[1] * 100:.1f})", (u + 8, v - 8), overlay.MAGENTA)

            det_status = ""
            if detect_on:
                xyz, yaw, det, _ = locate(frame, gcfg, proj)
                if det is None:
                    det_status = " | no object"
                else:
                    cv2.drawContours(show, [det.contour], -1, overlay.GREEN, 2)
                if xyz is not None:
                    r = float(np.hypot(xyz[0], xyz[1]))
                    ok = ws["min_radius_m"] <= r <= ws["max_radius_m"] and xyz[0] >= ws["min_x_m"]
                    c = project_arm_points(proj, np.array([[xyz[0], xyz[1], xyz[2] + half_h]]))
                    color = overlay.GREEN if ok else overlay.RED
                    if c is not None:
                        overlay.cross(show, c[0], color, 22, 3)
                        if yaw is not None:  # jaw opening direction, 4 cm each side
                            d = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw)), 0.0]) * 0.04
                            ends = project_arm_points(proj, np.array([xyz - d, xyz + d]) + [0, 0, half_h])
                            if ends is not None:
                                cv2.line(show, tuple(map(int, ends[0])), tuple(map(int, ends[1])), color, 2)
                    det_status = (
                        f" | object ({xyz[0] * 100:.1f}, {xyz[1] * 100:.1f}) cm, r={r * 100:.0f} cm "
                        f"{'IN REACH' if ok else 'OUT OF REACH'}"
                        + (f", jaws {yaw:.0f} deg" if yaw is not None else ", round")
                    )
            if "s" in keys:
                OUT_DIR.mkdir(exist_ok=True)
                path = OUT_DIR / f"click_{time.strftime('%H%M%S')}.jpg"
                cv2.imwrite(str(path), cv2.cvtColor(show, cv2.COLOR_RGB2BGR))
                msg = f"saved {path.name}"
            view.show(
                show,
                f"{msg}{det_status}\nclick = floor point  D detection  M mask  G grid  C clear  S snapshot  Q quit",
            )
    finally:
        cam.release()
        view.close()
        if follower is not None:
            print("Torque off: hold the arm if it isn't resting.")
            follower.disconnect()


if __name__ == "__main__":
    main()
