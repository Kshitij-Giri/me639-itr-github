# Task 3: Buzz-Wire on the Addverb Heal (teach by teleop, replay with DLS-IK)

The Heal cobot holds a ring in its Robotiq 85 gripper and has to pass the ring along a bent wire without touching it.

1. **Teach.** We drive the arm along the wire with a keyboard teleop that is locked to a plane, and record the end-effector path.
2. **Process.** We clean the recording and turn it into one smooth path with a speed profile.
3. **Execute.** The arm replays the path on its own, using damped-least-squares IK and joint velocity control. It runs in reverse, starting where teaching stopped.

Everything runs on ROS 2 Humble with Python 3.10. Path processing needs only numpy.

---

## Team

| Name | Roll No |
|---|---|
| Kshitij Giri | 23110177 |
| Parshva Patel | 24110240 |
| Kush Mehta | 24110177 |

---

## Demonstration Video

Youtube Link: 

---

## 1. Repository layout

```
Heal_Buzzwire_Task3/
├── README.md
├── requirements.txt
├── src/
│   ├── heal_plane_teleop.py    # teleop + path recorder (ROS 2 node)
│   ├── plane_core.py           # pure-numpy core: plane control, DLS, alignment, IK simulation
│   ├── align_gripper_axis.py   # standalone tool: align a gripper axis/edge to a base axis
│   ├── plot_recording.py       # quick 3D / plane plot of a raw recording
│   ├── stitch_recordings.py    # join recording parts (after a robot fault) into one file
│   ├── prepare_path.py         # recording -> smooth path + speed profile (+ plot)
│   ├── replay_core.py          # pure-numpy path follower (governor, approach, resume)
│   └── heal_ik_replay.py       # autonomous replay: DLS-IK + velocity control (ROS 2 node)
├── recordings/
│   ├── wire_path_20261115_093151.csv / _meta.json   # teach run, part 1
│   ├── wire_path_20261115_094705.csv / _meta.json   # part 2 (after robot fault)
│   ├── wire_path_20261115_095857.csv / _meta.json   # part 3
│   ├── wire_path_stitched.csv / _meta.json          # parts 1-3 joined
│   └── wire_path_prepared.csv / _meta.json          # final replay path (input to heal_ik_replay.py)
├── demo-videos/                # original demonstration videos with compiled demo video
├── docs/
│   ├── wire_path_prepared.png  # path coloured by speed, zooms, speed + ring-rotation profiles
│   ├── tube_compare.png        # smoothing tube 3 / 5 / 8 mm comparison
│   └── raw_recording.png
└── legacy/
    └── heal_wire_teleop.py     # first free-Cartesian teleop (superseded, kept for reference)
```

All commands below are run from the **repository root**. The scripts import `plane_core.py` and `replay_core.py` from their own folder, so they work as `python3 src/<script>.py`.

---

## 2. System overview

```
 [heal_server on robot]  <--ssh/docker-->  [velocity_control.launch.py on PC]
                                                     |  /joint_states   ^ /velocity_controller/commands
                                                     v                  |
   TEACH:   heal_plane_teleop.py  --(K/G lock plane, w/s z/x a/d q/e, r record)-->  recordings/*.csv
   PROCESS: stitch_recordings.py -> prepare_path.py  -->  wire_path_prepared.csv (+ plot)
   EXECUTE: heal_ik_replay.py   --(DLS-IK + velocity control, reverse from end)-->  ring along wire
```

Both nodes command the arm only through joint velocities on `/velocity_controller/commands` (joint1..joint6). Forward kinematics and the Jacobian come from KDL, using the `/robot_description` that the launch publishes. The tool frame is `rq_tool_tip`.

---

## 3. Features

### 3.1 Teleop and recording (`heal_plane_teleop.py`)

**Plane locking: how you set up the motion.**
The wire is bent in roughly one vertical plane. Instead of free 6-DOF motion, the teleop locks the gripper orientation and a plane, so each key moves the ring along one meaningful direction.

| Key | Action |
|---|---|
| `G` | Go to the saved start pose: a slow joint-space quintic move, then an exact preset alignment and plane lock. This is the normal way to begin. |
| `S` | Save the current joint angles as the start pose. |
| `K` | Lock the current orientation exactly as it is and start plane control in the current plane. |
| `Y` | Align to the preset orientation (pointing axis, finger edge and target axis are parameters), then lock. |
| `H` | Square the gripper up to the vertical plane in front of it with minimal tilt. The move is simulated first and refused if it would come near a joint limit. |
| `T` | Choose which tool axis is the plane normal. |
| `L` / `R` | Mark two points on the wire to define the plane direction. |
| `m` | JOINT mode (`1`–`6` select a joint, `=`/`-` jog it). Used to rough-position before locking. |

**Moving in the locked plane.**

| Key | Motion |
|---|---|
| `w` / `s` | Up / down in the plane |
| `z` / `x` | Left / right along the plane |
| `a` / `d` | Plane offset: the whole plane moves in/out and stays parallel |
| `q` / `e` | Ring yaw. Joint 6 is locked and moves only while q/e is held. |
| `[` `]` / `{` `}` | Speed: 8 levels from 0.2 to 10 cm/s (fine moves near the wire, fast moves in free space) |
| `g` / `o` | Close / open the gripper (force 40) |
| `space` | Stop |

**How the motion is computed.** Each cycle, the key input sets a Cartesian twist in the plane frame. `PlaneCore.step()` solves it with DLS-IK:
- `dq = Jᵀ(JJᵀ + λ²I)⁻¹ e`, where the damping λ increases near singularities (adaptive: ε = 0.015, λ_max = 0.06).
- It uses feed-forward plus proportional correction on the pose error, so the orientation does not drift during long translations.
- When q/e is not pressed, the task is 5-D (position plus pointing direction) with joint 6 removed from the Jacobian. This makes in-plane moves clean and stops joint 6 from wandering into its limit.

**Recording.**
- `r` starts and stops recording. `p` marks a keyframe and `b` marks a buzzer contact.
- Each sample stores time, flag, the 6 joint angles, the tool-tip pose (xyz + quaternion in `base_link`), the in-plane coordinates (u, v, d) and the ring angle ψ.
- A `_meta.json` stores frames, joint names and the plane definition.

**Safety layer (always on).**
- Joint velocity cap of 0.30 rad/s with uniform scaling (the driver rejects anything above 0.5 rad/s); acceleration cap of 0.6 rad/s².
- Predictive joint limits: if any joint would reach its soft limit (±170°, joint4 ±165°) within the look-ahead time, the **whole** arm stops instead of moving in a distorted way.
- Singularity guard: σ_min of the Jacobian is shown live, there is a NEAR SINGULARITY warning, and the arm holds below `sing_stop`.
- Only complete `/joint_states` messages are accepted; stale states give zero velocity.
- Following watchdog: if motion is commanded but the joints don't move for 1.5 s (robot in error), commands go to zero.
- `0` RECOVER brings joints back inside their limits; `9` sends the arm HOME.

### 3.2 From waypoints to a path (`stitch_recordings.py`, `prepare_path.py`)

The teleop recording is not a good trajectory as it stands. It contains:
- pauses
- one-axis-at-a-time staircase moves
- back-and-forth corrections
- in our case, two robot faults that split the teach run into 3 files

The path is turned into a single replayable path in these steps:

| Step | Logic | Our data |
|---|---|---|
| 1. Stitch | Join the parts in order and check the tip and orientation jump at each join. Drop fault rows: all joints exactly 0, or jumps > 5 mm. | Joins 0.35 mm / 0.05° and 0.54 mm / 0.10°; 1,668 + 537 fault rows dropped |
| 2. Remove pauses | Keep a sample only if the tip moved ≥ `dwell-mm` or ψ changed ≥ `dwell-deg`. | 68,171 → 5,233 samples |
| 3. Remove back-and-forth | If the path comes back within `loop-mm` (and `loop-deg`) of an earlier point, cut the detour in between. Operator corrections and overshoots disappear. | 9 detours cut |
| 4. One smooth path | Arc-length resampling, then elastic-band smoothing constrained to a **3 mm tube** around the cleaned recording. This straightens the staircase while never moving further than the tube from where the ring actually passed. A light Gaussian rounding (5 mm) follows. | Max deviation 2.46 mm (95%: 1.75 mm) |
| 5. Ring rotation ψ | The pointing axis stays constant (spread 0.02°), so orientation reduces to one angle ψ about it. ψ is smoothed over 10 mm. | ψ range 173° |
| 6. Speed profile | Fast on straight stretches (radius ≥ 80 mm → 3 cm/s) and slow at bends (radius ≤ 30 mm → 1 cm/s). Ring-rotation speed is limited (`w-max` 0.25 rad/s), a look-ahead slows the arm before each bend, and acceleration is limited. | 96.3 cm path, 143 s at scale 1.0 |
| 7. Output | Resample every 1 mm: `s, x, y, z, qx, qy, qz, qw, psi, v_ref, radius`. `_meta.json` stores the start and end joint angles and poses (for go-to-start/end) and statistics. | 965 points |

`--plot` writes `docs/wire_path_prepared.png`: the path coloured by speed, zooms at the largest deviation and the tightest bend, and the speed and ψ profiles. Use it to inspect the path before running anything on the robot.

### 3.3 Executing the path with velocity control (`heal_ik_replay.py` + `replay_core.py`)

**Control loop (100 Hz).**
1. Read the joints and compute the tool pose and Jacobian with KDL.
2. The `PathFollower` advances the path parameter: `ṡ = speed_scale · v_ref(s) · governor`.
3. Reference twist = path feed-forward (tangent × ṡ, ψ̇) + P correction (`kp_lin` 3, `kp_rot` 2) on the position and orientation error to the reference point.
4. DLS-IK gives joint velocities. These are capped per joint (uniform scaling keeps the direction), acceleration-limited, and checked against joint limits and the singularity guard.
5. Publish to `/velocity_controller/commands`.

**Features.**
- **Reverse replay from where teaching stopped.** With `direction auto`, the replay starts from whichever path end the arm is nearest. After teaching, the arm is at the end of the wire, so it traces back without re-gripping.
- **Approach phase.** If the arm is a few mm or degrees off the path start (within `start_tol`, 10 mm / 15°), it first moves slowly (5 mm/s) onto the path.
- **Clearance governor.** This protects the no-contact score. `clearance_mm` = (ring inner diameter − wire diameter) / 2.
  - Tracking error ≥ clearance / 3: slow down.
  - ≥ clearance / 2: hold the path parameter until the arm catches up.
  - ≥ 0.8 × clearance: abort.
  - The same scheme is applied to rotation error (3° / 6° / 12°).
- **Recovery commands.** `E` / `S` go to the recorded end/start pose (slow joint-space move, confirm with `y`) for re-gripping. `c` continues from the nearest path point after an abort.
- **Single-publisher check.** The replay refuses to start, or aborts, if another node (for example the teleop) is publishing velocity commands.
- **FK self-check at startup.** FK of the recorded end joints must match the recorded tip within 2 mm, or the node refuses to run (see Learnings: KDL objects being garbage-collected).
- **Trial log.** Every run is appended to `recordings/trials.csv` with direction, scale, duration, max/mean error, hold time and result. This is our repeatability evidence.

| Key | Action |
|---|---|
| `Enter` | Start from the nearest end |
| `space` | Pause / resume |
| `x` | Abort |
| `+` / `-` | Speed scale ±0.1 |
| `c` | Continue from the nearest point |
| `E` / `S` | Go to recorded end / start |
| `f` | Flip direction |
| `g` / `o` | Close / open gripper |

### 3.4 Results

| speed_scale | Full trace time |
|---|---|
| 0.5 | 4 min 45 s |
| 0.8 | 2 min 42 s |
| 1.5 | 1 min 42 s |

- Offline simulation of the same controller on the prepared path gave a max tracking error of 0.81 mm at scale 1.0 and 0.41 mm at 0.5, with peak joint speed 0.27 rad/s.
- **Speed limit analysis:**
  - About half the run is ring rotation. The ring turns about 150° back and forth nine times, about 1,160° (20.3 rad) in total, and almost all of that is joint 6.
  - With joint speed capped at 0.30 rad/s, rotation alone takes at least about 68 s, which is why times flatten out above scale 1.5.
  - Raising the cap to 0.40 rad/s (`-p max_joint_vel:=0.40 -p speed_scale:=1.6`) gives about 90 s, with 20% margin to the 0.5 rad/s fault.
  - Getting under 60 s would need joint 6 at about 0.47 rad/s or more, right at the driver's fault speed. We judged that not worth the risk to the no-contact and safety scores.

---

## 4. What we learned

**Choosing the right plane and start pose matters more than the controller.**
- The Heal has a wrist singularity: when the joint4/5/6 axes line up, σ_min of the Jacobian goes to zero.
- Our first start pose had σ_min ≈ 0.035. It looked fine, but during the in-plane moves the arm drifted into the singularity: σ_min hit 0.000, DLS damping froze the motion, and joint 6 spun toward its limit.
- We searched poses near the operator's preferred one and saved one with the same gripper orientation but σ_min ≈ 0.133 (`G` start pose: `[0.01, 37.93, 27.50, -90.08, 0.09, -0.70]°`). That is about 4× further from the singularity.
- With the plane chosen so the whole wire can be traced from there without passing through the singular configuration, the arm didn't stall again during teleop or replay.
- **Lesson:** in a velocity-controlled IK system, the workspace and plane you pick decide whether IK is well-conditioned. Check σ_min along the expected motion before teaching, not after.

**Lock what doesn't need to move.**
- Pointing the ring needs only 5 DOF. Joint 6 adds nothing to "position + pointing", so leaving it free lets DLS move it at random, which made the a/d and z/x keys feel erratic and drove it into its limit.
- Locking joint 6 unless q/e is pressed made the teleop predictable.

**Align to how the gripper is actually mounted.**
- The plane is derived from the tool frame (`rq_tool_tip`: x = finger open/close, y = flat-face normal, z = approach), not from a guessed world direction.
- An early default faced the robot's side (+X) instead of its front (+Y).

**Use the joint limits that are actually enforced.**
- The limits in the vendor config headers (`default_joint_limits`) are dead code, and URDF limits aren't enforced. The real limits live in `heal_server`.
- Using the wrong ones blocked joints far too early. We set soft limits at ±170° (joint4 ±165°, since −178° was rejected) and the 0.5 rad/s driver velocity check.

**Teach in segments and stitch.**
- The robot faulted twice during teaching. Because we recorded the joints as well as the poses, we could restart, re-lock (`K`) without moving, and stitch the parts with sub-mm joins.

**Process the recording; don't replay it raw.**
- Raw teleop is 161 cm of staircase moves, pauses and corrections. The cleaned path is 96 cm.
- Smoothing within a tube bound is what lets us run fast while still guaranteeing the path stays where the ring actually passed.

**Engineering lessons.**
- PyKDL solvers keep only a reference to the chain. If the chain is a local variable, it gets garbage-collected and FK silently returns zeros. Hence the startup FK self-check.
- Only one node may publish velocity commands. A second publisher makes the controller stop.
- Accept only complete `/joint_states` messages. With partial messages, another robot on the same ROS domain once contaminated the joint states.

---

## 5. Setup and run

### 5.1 Requirements

- Lab PC: Ubuntu 22.04, ROS 2 Humble, Python 3.10, the Addverb workspace (`~/Debojit_WS/addverb_ws`, built).
- Python packages: `numpy` and `matplotlib` (plots only). See `requirements.txt`.
- ROS Python packages: `PyKDL`, `urdf_parser_py` and `kdl_parser_py`. These are already present on the lab PC.
- Robot: Addverb Heal at `192.168.1.25`, with the Robotiq 85 gripper holding the ring.

```bash
pip3 install -r requirements.txt
echo 'export ROS_DOMAIN_ID=37' >> ~/.bashrc && source ~/.bashrc   # isolates us from other lab robots
```

### 5.2 Bring-up (three terminals, start in this order and stop in reverse)

**Terminal 1: robot server (SSH into the Heal and enter the docker container)**
```bash
ssh cobot@192.168.1.25
sudo docker ps                                              # container must be "Up"
sudo docker exec -it cobot_safety3_gripperfix_20260908 bash # use exec, NOT attach
cd /tests/build
./heal_server                                               # wait until it is listening
```

**Terminal 2: ROS 2 control on the PC (Debojit workspace)**
```bash
source /opt/ros/humble/setup.bash
cd ~/Debojit_WS/addverb_ws
source install/setup.bash
ros2 launch addverb_cobot_control velocity_control.launch.py
# check in another terminal:
ros2 control list_controllers              # velocity_controller must be active
ros2 topic hz /joint_states
```

**Terminal 3: our scripts (from this repository's root)**
```bash
source /opt/ros/humble/setup.bash
cd Heal_Buzzwire_Task3
```

### 5.3 Teach: teleop and record

```bash
python3 src/heal_plane_teleop.py --ros-args -p gripper_force:=40.0
```
1. `o` opens the gripper. Place the ring in it, then press `g` to close.
2. `G` goes to the start pose and locks the plane. Alternatively, position in JOINT mode (`m`) and lock with `K`.
3. Bring the ring onto the start of the wire with `w/s`, `z/x`, `a/d` and `q/e`; use `[` `]` for speed.
4. `r` starts recording. Trace the whole wire slowly, keeping a gap to the wire; press `b` if the buzzer sounds. Press `r` again to stop.
5. Recordings are saved to `recordings/wire_path_<timestamp>.csv` plus `_meta.json`. **Don't move the arm afterwards:** the replay starts from here.

Check a recording (optional):
```bash
python3 src/plot_recording.py recordings/wire_path_<timestamp>.csv
```
If the robot faults mid-trace: recover, restart the teleop, press `K` without moving, press `r`, and continue. You'll stitch the parts in the next step.

Standalone orientation tool (optional, used during setup):
```bash
python3 src/align_gripper_axis.py --point +Y --edge x --to X --dry-run   # preview, then run without --dry-run
```

### 5.4 Process: build the replay path

```bash
# only if the trace was recorded in several parts (in order along the wire):
python3 src/stitch_recordings.py recordings/wire_path_20261115_093151.csv \
        recordings/wire_path_20261115_094705.csv recordings/wire_path_20261115_095857.csv \
        -o recordings/wire_path_stitched.csv

python3 src/prepare_path.py recordings/wire_path_stitched.csv \
        -o recordings/wire_path_prepared.csv --plot docs/wire_path_prepared.png
```
The included `recordings/wire_path_prepared.csv` was produced by exactly these commands, and the result is reproducible bit for bit. Useful options:
- `--tube-mm` (default 3): maximum corner cutting. Keep it below the ring clearance.
- `--v-straight` / `--v-corner`
- `--w-max`
- `--a-max`

### 5.5 Execute: autonomous replay

Close the teleop first (Ctrl-C), so that only one node publishes velocity commands.

```bash
# first run: slow
python3 src/heal_ik_replay.py --ros-args -p path:=recordings/wire_path_prepared.csv \
        -p clearance_mm:=5.0 -p speed_scale:=0.5
```
- At startup, check for `FK self-check OK`.
- `Enter` starts the run (reverse, from the end where teaching stopped). `space` pauses and `x` aborts. Keep one person on the E-stop.

Faster runs, after clean slow runs:
```bash
python3 src/heal_ik_replay.py --ros-args -p path:=recordings/wire_path_prepared.csv \
        -p clearance_mm:=5.0 -p max_joint_vel:=0.40 -p speed_scale:=1.6
```
Repeatability: run 3 trials and compare max error and duration in `recordings/trials.csv`.

Set `clearance_mm` to (ring inner diameter − wire diameter) / 2.

### 5.6 Recovery

| Problem | Fix |
|---|---|
| `Robot gone into error` | `ros2 service call /cobot_services/error_recovery_srv std_srvs/srv/Trigger "{}"` (this homes the arm) |
| `Joint pos lower/upper error for joint N` (N counted from 0) | Stop `heal_server`, then run `./base_rigid_unsafe` in `/tests/build`. Run it again if it stops on the 0.5 rad/s check. |
| `Large Velocity for Joint` | Make sure only one node publishes `/velocity_controller/commands` (`ros2 topic info -v ...`) |
| Gripper `RxPacketError` | Power-cycle the controller box; use `gripper_force` 40 |
| Replay refuses: "far from start" | Press `E` / `S` to go to the recorded end/start, or `c` to continue from the nearest point |
| `docker attach` seems stuck | heal_server is running in the foreground; use `docker exec -it ... bash` |

---

## 6. Rubric mapping

| Criterion | Where |
|---|---|
| No-contact trace | Tube-bounded smoothing (§3.2), clearance governor and abort (§3.3) |
| Smooth trajectory | Elastic-band + Gaussian path, curvature-based, acceleration-limited speed profile, acceleration-limited joint commands |
| Repeatability (3 trials) | Same prepared path every time, `trials.csv` log |
| Safety | Velocity/acceleration caps, predictive limits, singularity guard, watchdogs, single-publisher check, FK self-check, pause/abort |
| Fastest (bonus) | `speed_scale` / `max_joint_vel`; analysis in §3.4 |
