# So101-Mono-Grasp

SO-101 arm control: leader-follower teleop, single RGB camera calibration, ground-plane depth estimation and pick-and-place.

The arm is mounted on the back of a quadruped and picks objects up off the floor. There is **one RGB camera** and no depth sensor. Depth comes from the camera intrinsics plus a few tape-measured distances: the camera's height and tilt above the ground.

This repo only holds the arm code. Robot drivers, motor calibration and kinematics come from [LeRobot](https://github.com/huggingface/lerobot).

---

## Roadmap

| # | Task | Status |
|---|------|--------|
| 1 | Leader-follower teleoperation | ⏳ In progress |
| 2 | Camera calibration (intrinsics + ground-plane extrinsics) | 🧪 Code written, tested on synthetic data, not yet on the real camera |
| 3 | Object detection → grasp (pick and place) | 🧪 Code written, tested on synthetic data, not yet on the arm |

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

Three scripts, all with a live window (tkinter: lerobot's OpenCV is headless, so `cv2.imshow` doesn't work):

| Script | Writes | What you do |
|---|---|---|
| `scripts/calibrate_intrinsics.py` | `calib/intrinsics.yaml` | wave a printed checkerboard in front of the camera |
| `scripts/calibrate_ground.py` | `calib/ground.yaml` + `camera_ground_in_base` in `config/grasp.yaml` | click 5–8 floor markers, type where they are |
| `scripts/click_to_floor.py` | nothing | check: click the floor → arm-frame coordinates; `D` shows the detected object and whether it's in reach |

The camera resolution is the one in `config/arms.yaml` (`camera:`). Calibrate at the resolution you pick at; if you change it, redo 2.1.

### 2.1 Intrinsics (once per camera and resolution)

Print a checkerboard (e.g. 10×7 squares = **9×6 inner corners**) and glue it on something flat. **Measure one square on the print** with a ruler, because printers rescale.

```bash
python scripts/calibrate_intrinsics.py --cols 9 --rows 6 --square 0.025
```

Move the board around: corners and edges of the image, near and far, tilted up to ~45° in every direction. Auto-capture takes a view whenever the board is steady somewhere new; the green dots show the coverage so far. After 15+ views press **C**: it calibrates, drops outlier views and saves. RMS < 0.5 px is good, and anything > 1 px means re-do it (bent board, blur, or a wrong `--square`). The frames are kept in `outputs/intrinsics_frames/` (re-run offline with `--images "outputs/intrinsics_frames/*.png"`).

### 2.2 Camera pose relative to the arm and the floor (once per camera mount)

No angle measuring: you click floor markers whose position you measured, and `cv2.solvePnP` finds the camera's height, tilt, roll **and** where it is relative to the arm.

1. Camera in its picking position. **Camera on the arm: the arm must be in `poses.look`** (`--move-to-look` moves it there and holds it). Pick a `look` pose where the camera sees the floor in front of the arm: move the arm by hand with `pick.py --where` and copy the joint values into `config/grasp.yaml`.
2. Put 5–8 markers (tape crosses, coins) on the floor, **spread out**: near/far and left/right, over the area you will pick from.
3. Measure each one in the **arm frame**: origin = the point directly below the shoulder rotation axis, **x forward, y to the LEFT**, in cm.
4. Run it, click each marker and type `x y` in cm:

   ```bash
   python scripts/calibrate_ground.py --move-to-look
   ```

   From the 4th marker on it fits on every click and draws the floor grid (5 cm squares, red = x axis, cyan = y axis, green = the arm's reach). The grid must lie on the real floor and the marker errors should be a few mm. One marker with a big error is probably mistyped: **U** removes the last one.
5. **S** saves `calib/ground.yaml` (h, tilt, roll and the markers) and rewrites the `camera_ground_in_base` line in `config/grasp.yaml`.

Arm on a different surface than the objects (quadruped)? Pass `--floor-z -0.30` (the floor is 30 cm below the arm frame origin).

### 2.3 Check it

```bash
python scripts/click_to_floor.py --move-to-look
```

Click the floor anywhere and compare with a tape: it should be within ~1 cm in the picking area. **D** turns on the detection (green cross = in reach, red = out of reach, line = jaw direction) and **M** shows the color mask for tuning `detection.hsv_ranges`.

### The math (`so101_mono_grasp/ground_plane.py`)

A pixel gives a ray (`camera.py`: undistort, then `[(u−cx)/fx, (v−cy)/fy, 1]`). Rotate it with the tilt and roll into the ground frame G (origin on the floor under the lens, X forward, Z up). Then intersect it with the horizontal plane at height `z0`: the floor, or half the object's height for a blob centroid:

```
d = R_ground_cam(tilt, roll) · ray
t = (h − z0) / (−d_z)            # needs d_z < 0: the ray goes down
P_G = (0, 0, h) + t·d
P_arm = Rz(yaw) · P_G + (x, y, z_floor)      # camera_ground_in_base
```

With `roll = 0` and the pixel on the image's center column this is the classic `D = h / tan(tilt + atan((v − cy)/fy))`.

### Assumptions and limitations

- The object is **on the floor**. For an object of known height, intersect the ray of the blob's **centroid** with the plane at **half the object's height** (that's what `pick.py` does). Using the floor plane with the centroid lands too far away; using the bottom-center pixel lands on the near edge.
- `h` and `tilt` only hold for the **pose the calibration was done in**:
  - Camera on the arm: always look from the same arm pose (`poses.look`). `pick.py` moves there before taking the picture.
  - Quadruped: pick only when standing still. Body pitch changes the tilt, so correct it with the IMU pitch (`tilt_effective = tilt + Δpitch`).
- The error grows with distance because a small tilt error matters more at shallow angles. Calibrate with markers in the range where you will actually pick.

---

## Task 3 — Object detection → grasp

```
camera frame
  → HSV color detection → blob centroid + long axis           so101_mono_grasp/detection.py
  → centroid ray ∩ plane at the object's half height          so101_mono_grasp/ground_plane.py  [Task 2 calib]
  → arm frame                                                 config/grasp.yaml: camera_ground_in_base
  → IK (placo + SO-101 URDF): top-down, jaws across the long axis
  → pre-grasp → descend → close → lift → (place) → home       so101_mono_grasp/grasp.py
```

| File | What it does |
|---|---|
| `config/grasp.yaml` | URDF, joint offsets, camera mount, HSV colors, grasp heights, gripper values, workspace, poses |
| `scripts/pick.py` | the whole pipeline, plus debug modes (below) |
| `so101_mono_grasp/kinematics.py` | FK and top-down grasp IK on placo; falls back to a tilted approach when vertical isn't reachable |
| `so101_mono_grasp/camera.py` | intrinsics (`calib/intrinsics.yaml`), undistortion, pixel → ray |
| `so101_mono_grasp/ground_plane.py` | `calib/ground.yaml` (h, tilt, roll), ray ∩ plane, ground → arm frame |
| `so101_mono_grasp/detection.py` | HSV mask, largest blob, centroid, long axis, annotated debug image |
| `so101_mono_grasp/grasp.py` | waypoint planning, joint-limit handling, smooth min-jerk moves on the follower |
| `assets/so101/` | SO-101 URDF + meshes (from the `lerobot/robot-urdfs` HF bucket) |

**Arm frame** (every coordinate in configs and scripts): origin on the surface the arm stands on, **directly below the shoulder rotation axis**; **x forward, y left, z up**, meters. (That is URDF `base_link` moved 3.9 cm forward. The `base_link` origin is under the back of the base, where you cannot measure from.)

### Bring-up, in this order

```bash
python scripts/pick.py --where                       # 1. torque OFF, move the arm by hand: is the printed tip xyz right?
python scripts/pick.py --target 0.25 0.0 --dry-run   # 2. IK only, prints the waypoints
python scripts/pick.py --target 0.25 0.0             # 3. real grasp at a tape-measured point, no camera
python scripts/pick.py --detect-only                 # 4. camera + detection, writes outputs/detection.jpg and mask.png
python scripts/pick.py --dry-run                     # 5. camera → floor point → IK, no motion
python scripts/pick.py                               # 6. the full pick
```

Add `--config config/<your ports>.yaml` if the udev names aren't installed. Before moving, the plan is printed and you confirm with Enter (`--yes` skips that). Torque turns off at the end, the same as teleop.

1. **FK check (`--where`).** This is the one that matters. LeRobot's 0° is the *middle of the range recorded during motor calibration*; the IK assumes that matches the URDF's 0°. Put the tip on a few measured points. If one joint is consistently off, set `urdf_offset_deg` for it.
2. **Gripper values.** `gripper_open` / `gripper_closed` are lerobot's 0–100 units. Read the real values with `--where` while opening/closing by hand.
3. **Detection colors.** Tune `detection.hsv_ranges` until `outputs/mask.png` is white only on the object.
4. **Reach.** Straight-down grasps from 0.16 to about 0.32 m from the shoulder axis. Up to 0.36 m the gripper tilts (60°); closer than 0.16 m the elbow cannot fold enough. Arm base above the floor (quadruped): set `camera_ground_in_base.z` negative, and the reach on the floor shrinks.

### Not done yet

- Camera pose from the arm's FK at *any* pose (hand-eye calibration), so the arm doesn't have to go back to `look` every time.
- A learned detector instead of HSV.

---

## Repo layout

```
So101-Mono-Grasp/
├── README.md
├── pyproject.toml          # depends on ../lerobot (editable)
├── config/
│   ├── arms.yaml           # ports, IDs, camera, teleop settings
│   └── grasp.yaml          # Task 3: camera mount, detection, grasp, poses
├── assets/so101/           # SO-101 URDF + meshes (Task 3)
├── calib/
│   ├── motors/             # follower.json, leader.json (Task 1)
│   ├── intrinsics.yaml     # Task 2.1 (calibrate_intrinsics.py)
│   └── ground.yaml         # Task 2.2 (calibrate_ground.py)
├── udev/
│   ├── 99-so101.rules      # stable /dev/so101_* names
│   └── find_usb_serial.sh
├── scripts/
│   ├── check_motors.py     # Task 1
│   ├── calibrate_arms.py   # Task 1
│   ├── teleop.py           # Task 1
│   ├── calibrate_intrinsics.py   # Task 2: checkerboard → fx, fy, cx, cy, distortion
│   ├── calibrate_ground.py       # Task 2: click measured floor markers → h, tilt, roll, camera mount
│   ├── click_to_floor.py         # Task 2: click a pixel → floor (x, y); live detection check
│   └── pick.py                   # Task 3: detect → localize → IK → grasp (+ --where, --target, --detect-only)
└── so101_mono_grasp/
    ├── arms.py             # Task 1: follower/leader from config
    ├── kinematics.py       # Task 3: FK + top-down grasp IK (placo)
    ├── camera.py           # Task 2: Camera capture, intrinsics, pixel → ray, undistortion
    ├── ground_plane.py     # Task 2: ray ∩ plane, ground → arm frame, camera pose fit (solvePnP)
    ├── viewer.py           # Task 2: live tkinter window (keys, clicks)
    ├── overlay.py          # Task 2: floor grid / reach / marker drawing
    ├── detection.py        # Task 3: HSV baseline (learned detector: planned)
    └── grasp.py            # Task 3: waypoint planning and execution
```
