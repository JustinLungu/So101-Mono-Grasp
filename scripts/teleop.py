"""Leader -> follower teleoperation, with an optional live camera view.

    uv run python scripts/teleop.py                 # with the camera, if enabled in config
    uv run python scripts/teleop.py --no-camera     # arms only (do this first)

Ctrl+C (or q in the camera window) stops it. Torque is turned off on exit, so hold the follower
or make sure it's resting before you stop.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from lerobot.utils.robot_utils import precise_sleep

from so101_mono_grasp.arms import DEFAULT_CONFIG, load_config, make_follower, make_leader


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--no-camera", action="store_true")
    p.add_argument("--fps", type=int, help="override teleop.fps from the config")
    p.add_argument("--print-every", type=float, default=1.0, help="seconds between joint printouts (0 = off)")
    args = p.parse_args()

    cfg = load_config(args.config)
    fps = args.fps or cfg.teleop_fps
    use_camera = not args.no_camera and cfg.camera.get("enabled", False)

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

    print(f"Teleop running at {fps} Hz. Ctrl+C to stop.")
    last_print = 0.0
    try:
        while True:
            t0 = time.perf_counter()

            action = leader.get_action()
            sent = follower.send_action(action)

            if use_camera:
                obs = follower.get_observation()
                frame = cv2.cvtColor(obs["front"], cv2.COLOR_RGB2BGR)
                cv2.imshow("follower camera", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.print_every and t0 - last_print >= args.print_every:
                last_print = t0
                print("  ".join(f"{k.removesuffix('.pos')}={v:6.1f}" for k, v in sent.items()))

            precise_sleep(max(0.0, 1 / fps - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping, torque off.")
        follower.disconnect()
        leader.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
