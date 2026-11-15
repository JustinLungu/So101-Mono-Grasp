"""Read-only health check for one or both arms. It never moves a motor or turns torque on.

For each motor ID 1-6 it reports whether the motor answers, plus its voltage and temperature.
Run this before calibrating to confirm the wiring and the power supply.

    uv run python scripts/check_motors.py               # both arms, ports from config/arms.yaml
    uv run python scripts/check_motors.py --port /dev/ttyACM0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

from so101_mono_grasp.arms import DEFAULT_CONFIG, MOTORS, load_config


def check_port(name: str, port: str) -> bool:
    print(f"\n=== {name}: {port} ===")
    if not Path(port).exists():
        print(f"  ✗ {port} does not exist (arm not plugged in, or udev rule not installed yet)")
        return False

    bus = SerialMotorsBus(
        port=port,
        motors={m: Motor(i, "sts3215", MotorNormMode.DEGREES) for m, i in MOTORS.items()},
    )
    try:
        bus.connect(handshake=False)  # no handshake, so we can report each missing motor ourselves
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ could not open port: {e}")
        print("    permission denied? -> sudo usermod -aG dialout $USER  (then log out/in)")
        return False

    ok = True
    try:
        for motor, motor_id in MOTORS.items():
            model = bus.ping(motor, num_retry=2)
            if model is None:
                print(f"  ✗ id {motor_id} {motor:<14} no answer")
                ok = False
                continue
            try:
                volts = bus.read("Present_Voltage", motor, normalize=False, num_retry=2) / 10
                temp = bus.read("Present_Temperature", motor, normalize=False, num_retry=2)
            except (ConnectionError, RuntimeError) as e:
                print(f"  ⚠ id {motor_id} {motor:<14} answered, then garbled reply ({e.__class__.__name__}): loose contact?")
                ok = False
                continue
            print(f"  ✓ id {motor_id} {motor:<14} model {model}  {volts:4.1f} V  {temp:3d} °C")
    finally:
        bus.disconnect(disable_torque=False)  # leave torque as it was

    if not ok:
        print(
            "  -> If NO motor answers: check power, cables and the voltage first, then run motor setup\n"
            "     (new motors all ship with ID 1, so lerobot-setup-motors has to assign 1-6).\n"
            "  -> If only some don't answer: reseat the 3-pin cable going into the first missing motor."
        )
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--port", help="check a single port instead of both arms from the config")
    args = p.parse_args()

    if args.port:
        results = [check_port("arm", args.port)]
    else:
        cfg = load_config(args.config)
        results = [
            check_port("follower", cfg.follower["port"]),
            check_port("leader", cfg.leader["port"]),
        ]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
