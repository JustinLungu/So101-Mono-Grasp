"""Leader -> follower teleoperation, with an optional live camera view.

    uv run python scripts/teleop.py --no-camera     # arms only
    uv run python scripts/teleop.py                 # with the camera window (needs an OpenCV build with GUI)

Once a second it prints the real loop rate and how far each follower joint is from the leader ("lag"),
so you can see whether the follower is keeping up.

Ctrl+C (or q in the camera window) stops it. Torque is turned off on exit, so hold the follower
or make sure it's resting before you stop.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from lerobot.utils.robot_utils import precise_sleep

from so101_mono_grasp.arms import DEFAULT_CONFIG, load_config, make_follower, make_leader


def opencv_has_gui() -> bool:
    info = cv2.getBuildInformation()
    gui_line = next((line for line in info.splitlines() if line.strip().startswith("GUI:")), "")
    return "NONE" not in gui_line.upper()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--no-camera", action="store_true")
    p.add_argument("--fps", type=int, help="override teleop.fps from the config")
    p.add_argument("--print-every", type=float, default=1.0, help="seconds between status lines (0 = off)")
    args = p.parse_args()

    cfg = load_config(args.config)
    fps = args.fps or cfg.teleop_fps
    use_camera = not args.no_camera and cfg.camera.get("enabled", False)
    if use_camera and not opencv_has_gui():
        print("OpenCV here has no GUI (opencv-python-headless), so no camera window. Running without the camera.")
        use_camera = False

    # lerobot logs a warning on EVERY loop when max_relative_target clamps a goal; that floods the
    # terminal and slows the loop. The status line below shows the lag instead.
    logging.getLogger("lerobot.robots.utils").setLevel(logging.ERROR)

    follower = make_follower(cfg, with_camera=use_camera)
    leader = make_leader(cfg)

    # calibrate=False: never start an interactive calibration by surprise; run calibrate_arms.py first.
    follower.connect(calibrate=False)
    try:
        leader.connect(calibrate=False)
    except Exception:
        follower.disconnect()
        raise
    if not (follower.is_calibrated and leader.is_calibrated):
        follower.disconnect()
        leader.disconnect()
        sys.exit("Arms are not calibrated (or the calibration doesn't match). Run scripts/calibrate_arms.py first.")

    f = cfg.follower
    print(
        f"Teleop target {fps} Hz | step cap: {f.get('max_relative_target')} | "
        f"P={f.get('p_coefficient', 16)} D={f.get('d_coefficient', 32)}. Ctrl+C to stop."
    )
    last_print = time.perf_counter()
    loops = 0
    try:
        while True:
            t0 = time.perf_counter()

            action = leader.get_action()
            follower.send_action(action)
            loops += 1

            if use_camera:
                obs = follower.get_observation()
                frame = cv2.cvtColor(obs["front"], cv2.COLOR_RGB2BGR)
                cv2.imshow("follower camera", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.print_every and t0 - last_print >= args.print_every:
                hz = loops / (t0 - last_print)
                last_print, loops = t0, 0
                present = follower.bus.sync_read("Present_Position", num_retry=2)
                lag = {k: action[f"{k}.pos"] - v for k, v in present.items()}
                worst = max(lag, key=lambda k: abs(lag[k]))
                print(
                    f"{hz:5.1f} Hz | lag (leader - follower): "
                    + "  ".join(f"{k}={v:+5.1f}" for k, v in lag.items())
                    + f"  | worst: {worst}"
                )

            precise_sleep(max(0.0, 1 / fps - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping, torque off.")
        follower.disconnect()
        leader.disconnect()
        if use_camera:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
