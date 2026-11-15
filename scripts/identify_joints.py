"""Live, read-only view of every motor's raw position, to check which motor ID sits in which joint.

Move ONE joint at a time by hand and watch which row changes. Torque is never turned on.
A sudden jump of more than JUMP ticks between two reads is counted as a glitch (a bad reading),
which is different from a joint really moving through the 4095 -> 0 boundary.

    uv run python scripts/identify_joints.py --arm follower
    uv run python scripts/identify_joints.py --port /dev/ttyACM0

Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

from so101_mono_grasp.arms import DEFAULT_CONFIG, MOTORS, load_config

JUMP = 300  # ticks between two reads ~20 ms apart; a hand can't move a joint that fast (~26°)
BOUNDARY = 600  # a step from near 4095 to near 0 (or back) is a real wrap, not a glitch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--arm", choices=["follower", "leader"])
    g.add_argument("--port")
    args = p.parse_args()

    port = args.port or load_config(args.config).raw[args.arm]["port"]
    names = {i: n for n, i in MOTORS.items()}
    bus = SerialMotorsBus(port, {f"id{i}": Motor(i, "sts3215", MotorNormMode.DEGREES) for i in names})
    bus.connect(handshake=False)

    def read(i: int) -> int | None:
        try:
            return int(bus._read("Present_Position", i, 0))
        except Exception:  # noqa: BLE001
            return None

    start = {i: read(i) for i in names}
    last = dict(start)
    lo = {i: v for i, v in start.items()}
    hi = {i: v for i, v in start.items()}
    glitches = {i: 0 for i in names}
    wraps = {i: 0 for i in names}
    errors = {i: 0 for i in names}

    print(f"Reading {port}. Move one joint at a time. Ctrl+C to stop.\n")
    lines = len(names) + 2
    try:
        while True:
            for i in names:
                v = read(i)
                if v is None:
                    errors[i] += 1
                    continue
                prev = last[i]
                if prev is not None and abs(v - prev) > JUMP:
                    if (prev > 4095 - BOUNDARY and v < BOUNDARY) or (prev < BOUNDARY and v > 4095 - BOUNDARY):
                        wraps[i] += 1
                    else:
                        glitches[i] += 1
                        continue  # don't let a glitch move the tracked range
                last[i] = v
                lo[i] = v if lo[i] is None else min(lo[i], v)
                hi[i] = v if hi[i] is None else max(hi[i], v)

            out = [f"{'ID':>2}  {'lerobot name':<14} {'POS':>5} {'MOVED':>6}  {'MIN':>5} {'MAX':>5}  WRAPS GLITCH ERR"]
            for i, n in names.items():
                pos = last[i]
                moved = "" if pos is None or start[i] is None else f"{pos - start[i]:+d}"
                flag = "  <== moving" if moved and abs(int(moved)) > 50 else ""
                out.append(
                    f"{i:>2}  {n:<14} {pos if pos is not None else '--':>5} {moved:>6}  "
                    f"{lo[i] if lo[i] is not None else '--':>5} {hi[i] if hi[i] is not None else '--':>5}  "
                    f"{wraps[i]:>5} {glitches[i]:>6} {errors[i]:>3}{flag}"
                )
            print("\n".join(line + "\033[K" for line in out), flush=True)
            time.sleep(0.02)
            print(f"\033[{lines - 1}A", end="")
    except KeyboardInterrupt:
        print("\n" * lines)
    finally:
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
