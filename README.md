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

### 1. Install LeRobot

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
uv sync --locked --extra core_scripts --extra feetech --extra kinematics
```

`feetech` is the motor bus driver. `kinematics` adds `placo`, which does the forward and inverse kinematics.

### 2. Serial port permissions (Linux)

```bash
sudo usermod -aG dialout $USER   # then log out and back in
```

The one-off alternative: `sudo chmod 666 /dev/ttyACM*` after each time you plug the arms in.

---

## Task 1 — Leader-follower teleoperation

Moving the leader arm by hand makes the follower copy its joint angles.

Run every command from the `lerobot/` checkout and prefix it with `uv run`.

### 1.1 Find the ports

Plug in and power one arm at a time:

```bash
uv run lerobot-find-port
```

Unplug the arm when it asks. The port that disappears belongs to that arm. On Linux it's usually `/dev/ttyACM0` / `/dev/ttyACM1`.

| Arm | Port |
|---|---|
| Follower | `/dev/ttyACM?` |
| Leader | `/dev/ttyACM?` |

### 1.2 Set motor IDs (one time, skip if the arms came pre-configured)

```bash
uv run lerobot-setup-motors --robot.type=so101_follower --robot.port=<FOLLOWER_PORT>
uv run lerobot-setup-motors --teleop.type=so101_leader  --teleop.port=<LEADER_PORT>
```

### 1.3 Calibrate the motors

Move the arm to the middle of its range and press Enter. Then move **every** joint through its full range. The `id` is the name the calibration is saved under, so use the same one every time.

```bash
uv run lerobot-calibrate --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=follower
uv run lerobot-calibrate --teleop.type=so101_leader  --teleop.port=<LEADER_PORT>  --teleop.id=leader
```

Calibrations are saved to `~/.cache/huggingface/lerobot/calibration/`.

> This calibrates the **motors** so that the leader and follower agree on joint angles. Camera calibration is Task 2.

### 1.4 Teleoperate

```bash
uv run lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=follower \
  --teleop.type=so101_leader  --teleop.port=<LEADER_PORT>  --teleop.id=leader \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --display_data=true
```

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

## Repo layout (planned)

```
So101-Mono-Grasp/
├── README.md
├── calib/                  # saved calibration results (intrinsics.yaml, ground.yaml)
├── scripts/
│   ├── calibrate_intrinsics.py   # checkerboard → fx, fy, cx, cy, distortion
│   ├── calibrate_ground.py       # click markers + tape distances → tilt, roll, h
│   ├── click_to_floor.py         # live check: click a pixel → floor (x, y)
│   └── pick.py                   # detect → localize → IK → grasp
└── so101_mono_grasp/
    ├── camera.py           # pixel ↔ ray, undistortion
    ├── ground_plane.py     # ray ∩ floor, tilt/roll fitting
    ├── detection.py        # HSV baseline, then a learned detector
    └── grasp.py            # IK and gripper sequencing
```
