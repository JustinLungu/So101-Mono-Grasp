# So101-Mono-Grasp

SO-101 arm control: leader-follower teleop, single RGB camera calibration, ground-plane depth estimation and pick-and-place.

The arm is mounted on the back of a quadruped and picks objects up off the floor. There is **one RGB camera** and no depth sensor. Depth comes from the camera intrinsics plus a few tape-measured distances: the camera's height and tilt above the ground.

This repo only holds the arm code. Robot drivers, motor calibration and kinematics come from [LeRobot](https://github.com/huggingface/lerobot).

---

## Roadmap

| # | Task | Status |
|---|------|--------|
| 1 | Leader-follower teleoperation | ⏳ In progress |
| 2 | Camera calibration (intrinsics + ground-plane extrinsics) | ⏳ Planned |
| 3 | Object detection → grasp (pick and place) | ⏳ Planned |

---

## Hardware

- **Follower arm:** SO-101 with 6× Feetech STS3215 (1/345 gearing)
- **Leader arm:** SO-101 leader with mixed gearing (1/191, 1/345, 1/147)
- **Camera:** 1× USB RGB camera (mount: wrist or quadruped body, TBD)
- **Host:** Linux laptop for development, later the quadruped's onboard computer
- **Quadruped:** the base the arm is mounted on

### ⚠️ Power supply voltage

The STS3215 comes in a **7.4 V** version and a **12 V** version. Using the wrong power supply puts the motors into an error state (blinking red LEDs) and can damage them.

| Motor sticker code | Voltage | Power supply |
|---|---|---|
| C001 / C044 / C046 | 7.4 V | 5 V brick |
| C018 (or the sticker says "12V") | 12 V | 12 V brick |

The leader is almost always the 7.4 V type. Check the follower's motor stickers and the label on each brick before powering on.

---

## Setup

This repo expects a LeRobot checkout **next to it** (`../lerobot`) and installs it as an editable dependency.

```bash
git clone https://github.com/huggingface/lerobot.git ../lerobot   # if it isn't there yet
uv sync                                                          # creates .venv with lerobot[feetech,kinematics]
```

Serial port permissions (Linux). Not needed once the udev rule from 1.2 is installed:

```bash
sudo usermod -aG dialout $USER   # then log out and back in
```

All the hardware settings (ports, arm IDs, camera, loop rate) are in [`config/arms.yaml`](config/arms.yaml). Every script reads that file.

---

## Task 1 — Leader-follower teleoperation

Moving the leader arm by hand makes the follower copy its joint angles.

| File | What it does |
|---|---|
| `config/arms.yaml` | ports, arm IDs, calibration folder, camera, safety cap |
| `scripts/check_motors.py` | read-only check: does each motor 1–6 answer, plus its voltage and temperature |
| `scripts/calibrate_arms.py` | motor calibration, saved to `calib/motors/<id>.json` (kept in the repo) |
| `scripts/teleop.py` | leader → follower loop with an optional camera window |
| `udev/` | stable port names `/dev/so101_follower` and `/dev/so101_leader` |
| `so101_mono_grasp/arms.py` | builds lerobot's follower/leader from the config (reused by later tasks) |

### 1.1 Find the ports

Plug in and power **one arm at a time**:

```bash
./udev/find_usb_serial.sh
```

Note which serial number belongs to which arm. Until the udev rule is installed, put the raw `/dev/ttyACM?` ports into `config/arms.yaml`.

### 1.2 Stable port names (udev)

`ttyACM0` and `ttyACM1` can swap between reboots, and the leader would then be read with the follower's calibration. To prevent that, fill both serials into [`udev/99-so101.rules`](udev/99-so101.rules), then:

```bash
sudo cp udev/99-so101.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
ls -l /dev/so101_*
```

### 1.3 Check the motors (read-only)

```bash
uv run python scripts/check_motors.py
```

Every motor should answer with a sensible voltage: about 5–8 V on the 7.4 V version, about 12 V on the 12 V version. If **no** motor answers on an arm, its motors probably still have the factory ID 1. Run motor setup once:

```bash
uv run lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/so101_follower
uv run lerobot-setup-motors --teleop.type=so101_leader  --teleop.port=/dev/so101_leader
```

### 1.4 Calibrate the motors

This is interactive. Put the arm in the middle of its range and press Enter. Then move **every** joint through its full range and press Enter again.

```bash
uv run python scripts/calibrate_arms.py                # both arms, follower first
uv run python scripts/calibrate_arms.py --arm leader   # just one
```

The calibrations are saved to `calib/motors/follower.json` and `calib/motors/leader.json`. They're kept in the repo so the quadruped's computer uses the same ones.

> This calibrates the **motors** so that the leader and follower agree on joint angles. Camera calibration is Task 2.

### 1.5 Teleoperate

```bash
uv run python scripts/teleop.py --no-camera   # arms only, do this first
uv run python scripts/teleop.py               # with the camera window (q to quit)
```

`follower.max_relative_target` in the config caps how far a joint can move in one step (10° by default). That avoids sudden jumps while the arm is new. Raise it or set it to `null` once teleop is smooth. Torque turns **off** when the script exits, so support the follower before stopping it.

To find the camera index: `uv run lerobot-find-cameras opencv`.

### Troubleshooting

Most failures are hardware problems, not code. Check the red motor LEDs along the daisy chain:

- **All steady red:** the wiring is fine.
- **Some dark / the chain stops partway:** reseat the 3-pin cables and check the power supply.
- **Blinking:** the motor is in an error state, either overloaded or getting the wrong voltage (see above).

---

## Task 2 — Camera calibration and depth from one camera

A single camera turns a pixel into a **direction** (a ray), not a distance. We get the distance by assuming the object is **on the floor** and working out where the ray hits the floor.

### 2.1 Intrinsics (once per camera and resolution)

Print a checkerboard and measure the size of one square with a tape. Take about 15 photos of it from different angles and distances, then run `cv2.calibrateCamera`. This gives:

- `fx, fy`: focal lengths in pixels
- `cx, cy`: the image center (principal point)
- distortion coefficients, used to undistort pixels before the ground-plane math

Re-run this if you change the camera, its resolution or its focus.

### 2.2 Ground-plane extrinsics (once per camera pose)

Measure with the tape:

- `h`: the camera's height above the floor

Then click markers on the floor and measure each one:

1. Put a marker on the floor and click it in the image (pixel `u, v`).
2. Measure its **horizontal** distance `D` along the floor from the point directly under the camera. This is not the straight-line distance from the lens.
3. Each marker gives one tilt estimate:

   ```
   angle_to_marker = atan(h / D)               # real angle below horizontal
   angle_in_image  = atan((v − cy) / fy)       # angle relative to the optical axis
   tilt            = angle_to_marker − angle_in_image
   ```

4. Use 3–5 markers at different distances and average the tilts, or fit them by least squares. Points spread across the image also let you solve for **roll**.

### 2.3 Pixel → floor position

For any new pixel `(u, v)`, after undistorting it:

```
α = tilt + atan((v − cy) / fy)     # angle below horizontal
D = h / tan(α)                     # forward distance along the floor
Y ≈ D · (u − cx) / fx              # sideways offset
```

`(D, Y, 0)` is the point on the floor in the camera's ground frame. A fixed transform, measured with the tape, then puts it in the **arm base frame**.

### Assumptions and limitations

- The object is **on the floor**. Height adds error, so aim for the object's bottom-center pixel, not its middle.
- `h` and `tilt` only hold for the **pose the calibration was done in**:
  - Camera on the wrist: always look from the same arm pose.
  - Quadruped: pick only when standing still. Body pitch changes the tilt, so correct it with the IMU pitch (`tilt_effective = tilt + Δpitch`).
- The error grows with distance because a small tilt error matters more at shallow angles. Calibrate with markers in the range where you will actually pick.

---

## Task 3 — Object detection → grasp

The pipeline:

```
camera frame
  → detect the object (bounding box / mask)
  → grasp pixel (bottom-center of the box, or centroid of the mask)
  → pixel → floor (x, y) in the camera ground frame       [Task 2]
  → transform into the arm base frame                     [measured mount offset]
  → IK (placo + SO-101 URDF) → pre-grasp above the object
  → descend → close gripper → lift → place
```

Plan for the detection step:

1. **Baseline:** color thresholding (HSV) on a brightly colored object. No training needed.
2. **Then:** a pretrained detector (YOLO, or an open-vocabulary model such as OWL-ViT / Grounding DINO) to pick objects by name.
3. **Orientation (optional):** angle of the mask's minimum-area rectangle → wrist roll.

The IK uses `lerobot.model.kinematics.RobotKinematics` with the [SO-101 URDF](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf).

---

## Repo layout

```
So101-Mono-Grasp/
├── README.md
├── pyproject.toml          # depends on ../lerobot (editable)
├── config/
│   └── arms.yaml           # ports, IDs, camera, teleop settings
├── calib/
│   ├── motors/             # follower.json, leader.json (Task 1)
│   └── …                   # intrinsics.yaml, ground.yaml (Task 2, planned)
├── udev/
│   ├── 99-so101.rules      # stable /dev/so101_* names
│   └── find_usb_serial.sh
├── scripts/
│   ├── check_motors.py     # Task 1
│   ├── calibrate_arms.py   # Task 1
│   ├── teleop.py           # Task 1
│   ├── calibrate_intrinsics.py   # Task 2 (planned): checkerboard → fx, fy, cx, cy, distortion
│   ├── calibrate_ground.py       # Task 2 (planned): click markers + tape → tilt, roll, h
│   ├── click_to_floor.py         # Task 2 (planned): click a pixel → floor (x, y)
│   └── pick.py                   # Task 3 (planned): detect → localize → IK → grasp
└── so101_mono_grasp/
    ├── arms.py             # Task 1: follower/leader from config
    ├── camera.py           # Task 2 (planned): pixel ↔ ray, undistortion
    ├── ground_plane.py     # Task 2 (planned): ray ∩ floor, tilt/roll fitting
    ├── detection.py        # Task 3 (planned): HSV baseline, then a learned detector
    └── grasp.py            # Task 3 (planned): IK and gripper sequencing
```
