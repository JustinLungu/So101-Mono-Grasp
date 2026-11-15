"""Task 3: detect an object on the floor with the camera and pick it up with the follower.

Pipeline: arm to the `look` pose -> camera frame -> HSV detection -> blob centroid -> ray hits the plane at
          the object's half height (Task 2 calibration) -> arm frame -> IK (placo + SO-101 URDF)
          -> pre-grasp -> descend -> close -> lift -> place

Coordinates: ARM FRAME, origin on the floor below the shoulder rotation axis, x forward, y left, meters.

Recommended order the first time:

  python scripts/pick.py --where                     # torque OFF: move the arm by hand, compare tip xyz with a tape
  python scripts/pick.py --target 0.25 0.0 --dry-run # IK only, no motion
  python scripts/pick.py --target 0.25 0.0           # real grasp at a tape-measured point (no camera)
  python scripts/pick.py --detect-only               # camera + detection only, saves outputs/detection.jpg
  python scripts/pick.py --dry-run                   # camera -> floor point -> IK, no motion (arm must already be in `look`)
  python scripts/pick.py                             # the full thing: moves to `look`, looks, grasps

Every command takes --config (ports, as for teleop.py) and --grasp-config (config/grasp.yaml).
Before any motion the plan is printed and you confirm with Enter (skip with --yes).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from so101_mono_grasp.arms import DEFAULT_CONFIG, REPO_ROOT, load_config
from so101_mono_grasp.camera import Camera
from so101_mono_grasp.detection import detect, draw, hsv_mask
from so101_mono_grasp.grasp import (
    DEFAULT_GRASP_CONFIG,
    GraspPlanner,
    connect_follower,
    execute,
    load_grasp_config,
    move_to,
    read_follower,
    resolve,
)
from so101_mono_grasp.ground_plane import FloorProjector, projector_from_config
from so101_mono_grasp.kinematics import ARM_JOINTS

OUT_DIR = REPO_ROOT / "outputs"


def fmt(v) -> str:
    return "[" + ", ".join(f"{x:+.3f}" for x in v) + "]"


# ---------------------------------------------------------------------------- camera
def grab_frame(cam_cfg: dict, image: str | None) -> np.ndarray:
    """One RGB frame, from a file or from the camera in config/arms.yaml."""
    if image:
        bgr = cv2.imread(image)
        if bgr is None:
            sys.exit(f"Could not read image {image}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with Camera.from_config(cam_cfg) as cam:
        return cam.read()


def locate(frame: np.ndarray, gcfg: dict, projector: FloorProjector | None):
    """Detection -> (floor xyz in the arm frame, jaw yaw in degrees or None, detection, text for the image)."""
    dcfg = gcfg["detection"]
    det = detect(frame, dcfg["hsv_ranges"], dcfg["min_area_px"])
    if det is None:
        return None, None, None, "no object found"
    text = f"area {det.area_px:.0f}px  elong {det.elongation:.2f}"
    if projector is None:
        return None, None, det, text
    projector.intr.check_frame(frame)

    # The blob's centroid is roughly the object's middle: intersect its ray with the plane at the object's
    # half height (not the floor), then drop the point to the floor.
    half_h = dcfg["object_half_height_m"]
    p = projector.pixel_to_base(*det.centroid_px, plane_height=half_h)
    if p is None:
        return None, None, det, text + "\nobject is above the horizon"
    p[2] -= half_h

    yaw = None
    if det.elongation >= dcfg["min_elongation_for_yaw"]:
        a0, a1 = (projector.pixel_to_base(*pt, plane_height=half_h) for pt in det.axis_px)
        if a0 is not None and a1 is not None:
            long_axis = np.degrees(np.arctan2(a1[1] - a0[1], a1[0] - a0[0]))
            yaw = (long_axis + 90.0 + 90.0) % 180.0 - 90.0  # jaws open across the long axis, in [-90, 90)
    text += f"\nfloor xyz {fmt(p)}" + (f"  jaw yaw {yaw:.0f}deg" if yaw is not None else "  (round)")
    return p, yaw, det, text


# ---------------------------------------------------------------------------- arm
def confirm(msg: str, yes: bool) -> None:
    if yes:
        return
    try:
        input(f"{msg} [Enter = go, Ctrl+C = abort] ")
    except KeyboardInterrupt:
        print("\nAborted.")
        raise SystemExit(1) from None


def where(cfg, planner: GraspPlanner) -> None:
    follower = connect_follower(cfg)
    follower.bus.disable_torque()
    print("Torque OFF: move the follower by hand. Tip = middle between the jaw tips. Ctrl+C to stop.\n")
    try:
        while True:
            q, grip = read_follower(follower)
            T = planner.kin.fk(planner.to_urdf(q))
            joints = "  ".join(f"{j}={v:+6.1f}" for j, v in zip(ARM_JOINTS, q))
            print(f"tip xyz {fmt(T[:3, 3])} m  pointing {fmt(T[:3, 2])} | {joints}  gripper={grip:5.1f}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        follower.disconnect()


def print_plan(plan) -> None:
    ik = plan.ik
    yaw = "any (round object)" if plan.yaw_deg is None else f"{plan.yaw_deg:.0f} deg"
    print(
        f"\nGrasp at {fmt(plan.target_xyz)} m | jaw yaw {yaw} | approach {ik.approach_pitch_deg:.0f} deg below "
        f"horizontal | IK error {ik.pos_error_m * 1000:.1f} mm"
    )
    for wp in plan.waypoints:
        xyz = fmt(wp.tip_xyz) if wp.tip_xyz is not None else "(joint pose)       "
        print(f"  {wp.name:9s} tip {xyz}  joints {np.round(wp.joints_deg, 1)}  gripper {wp.gripper:.0f}")


# ---------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG, help="arms config (ports, camera)")
    p.add_argument("--grasp-config", default=DEFAULT_GRASP_CONFIG)
    p.add_argument("--where", action="store_true", help="torque off, print the tip position live (FK check)")
    p.add_argument("--target", nargs=2, type=float, metavar=("X", "Y"), help="grasp at this floor point, no camera")
    p.add_argument("--yaw", type=float, help="with --target: jaw opening direction in degrees (default: any)")
    p.add_argument("--detect-only", action="store_true", help="camera + detection only, no arm")
    p.add_argument("--image", help="use this image file instead of the camera")
    p.add_argument("--dry-run", action="store_true", help="plan and print, never move the arm")
    p.add_argument("--yes", action="store_true", help="don't ask for confirmation before moving")
    args = p.parse_args()

    cfg = load_config(args.config)
    gcfg = load_grasp_config(args.grasp_config)
    f = cfg.follower
    calib_json = resolve(f.get("calibration_dir", "calib/motors")) / f"{f['id']}.json"
    planner = GraspPlanner(gcfg, calib_json if calib_json.exists() else None)

    if args.where:
        where(cfg, planner)
        return

    look = planner.pose("look")
    gripper_open = float(gcfg["grasp"]["gripper_open"])
    speed = gcfg["grasp"]["speed_deg_s"]
    use_camera = not args.target
    live = use_camera and not args.image
    will_move = not (args.dry_run or args.detect_only)

    projector: FloorProjector | None = None
    if use_camera:
        try:
            projector = projector_from_config(gcfg)
        except FileNotFoundError as e:
            if not args.detect_only:
                sys.exit(f"{e}\nUntil then, test the grasp with --target X Y (a tape-measured point).")
            print("(No camera calibration yet: detection only, no floor position.)")

    follower = None
    try:
        # ---- 0. camera on the arm: look from the pose the ground calibration was done in
        if will_move:
            follower = connect_follower(cfg)
            q, _ = read_follower(follower)
            print(f"Follower at {np.round(q, 1)}, tip {fmt(planner.tip_xyz(q))}")
            print("Stay clear of the arm. Torque turns OFF at the end or on error: hold it if it's not folded.")
            if live and look is not None:
                confirm(f"Move to the look pose {look.tolist()}?", args.yes)
                move_to(follower, look, gripper_open, speed)
                time.sleep(0.5)  # let the camera stop shaking
        elif live and not args.detect_only:
            print("--dry-run: the camera looks from the arm's CURRENT pose; it must be the `look` pose.")

        # ---- 1. where is the object?
        yaw = args.yaw
        if args.target:
            floor_xyz = np.array([args.target[0], args.target[1], float(gcfg["camera_ground_in_base"]["z"])])
            print(f"Target from the command line: {fmt(floor_xyz)} m")
        else:
            frame = grab_frame(cfg.camera, args.image)
            floor_xyz, yaw, det, text = locate(frame, gcfg, projector)
            OUT_DIR.mkdir(exist_ok=True)
            cv2.imwrite(str(OUT_DIR / "detection.jpg"), draw(frame, det, text))
            cv2.imwrite(str(OUT_DIR / "mask.png"), hsv_mask(frame, gcfg["detection"]["hsv_ranges"]))
            print(text.replace("\n", " | "))
            print(f"Saved {OUT_DIR / 'detection.jpg'} and {OUT_DIR / 'mask.png'}")
            if args.detect_only:
                return
            if floor_xyz is None:
                sys.exit("Nothing to grasp.")

        # ---- 2. plan
        seed = look if look is not None else np.zeros(5)
        try:
            plan = planner.plan(floor_xyz, yaw, seed)
        except ValueError as e:
            sys.exit(f"Can't grasp: {e}")
        print_plan(plan)
        if follower is None:
            print("\n--dry-run: not moving.")
            return

        # ---- 3. execute
        confirm("Run the grasp?", args.yes)
        if not live and look is not None:
            print("  -> look")
            move_to(follower, look, gripper_open, speed)
        if not execute(follower, plan, gcfg):
            print("Nothing grasped. Check outputs/detection.jpg and the FK (--where), then try again.")
    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        if follower is not None:
            follower.disconnect()


if __name__ == "__main__":
    main()
