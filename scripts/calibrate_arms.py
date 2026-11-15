"""Calibrate the follower and/or leader motors. Results are saved to calib/motors/<id>.json.

This is interactive: lerobot asks you to put the arm in the middle of its range, press Enter,
then move every joint through its full range and press Enter again.

    uv run python scripts/calibrate_arms.py              # both, follower first
    uv run python scripts/calibrate_arms.py --arm leader
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from so101_mono_grasp.arms import DEFAULT_CONFIG, load_config, make_follower, make_leader


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--arm", choices=["follower", "leader", "both"], default="both")
    args = p.parse_args()
    cfg = load_config(args.config)

    arms = []
    if args.arm in ("follower", "both"):
        arms.append(("follower", make_follower(cfg, with_camera=False)))
    if args.arm in ("leader", "both"):
        arms.append(("leader", make_leader(cfg)))

    for name, arm in arms:
        print(f"\n===== Calibrating {name} ({arm.config.port}) =====")
        arm.connect(calibrate=False)
        try:
            arm.calibrate()
        finally:
            arm.disconnect()


if __name__ == "__main__":
    main()
