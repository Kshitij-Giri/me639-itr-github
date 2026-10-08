import os

# On a Wayland desktop session, pyGLFW loads its native-Wayland GLFW build, so the
# MuJoCo viewer becomes a native Wayland window and pynput (X11-based) never
# receives any key from it. Force the X11 build (runs via XWayland on Wayland
# desktops). This MUST be set before mujoco.viewer / glfw are imported.
os.environ["PYGLFW_LIBRARY_VARIANT"] = "x11"

import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from pynput import keyboard


# ============================================================
# FILE PATH
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
LAB_DIR = SCRIPT_DIR.parent

XML_PATH = (
    LAB_DIR
    / "robot-descriptions"
    / "Quadrotor"
    / "scene.xml"
)

# Fallback: scene.xml sitting next to this script
if not XML_PATH.exists():
    XML_PATH = SCRIPT_DIR / "scene.xml"

print(f"Loading: {XML_PATH}")


# ============================================================
# LOAD MUJOCO MODEL
# ============================================================

model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)


# ============================================================
# QUADROTOR BODY
# ============================================================

body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "quadrotor"
)

if body_id == -1:
    raise RuntimeError("Could not find body 'quadrotor'")

# Index of the first DOF of the free joint.
# qvel[root_dof + 0:3] = linear velocity (world frame)
# qvel[root_dof + 3:6] = angular velocity (body/local frame)
root_dof = int(model.jnt_dofadr[model.body_jntadr[body_id]])


# ============================================================
# MOTOR ACTUATORS
# ============================================================
#
# From quadrotor.xml (sites are in the body frame):
#
#   thrust1 -> site motor1 at (+0.55,  0.00)  -> +X
#   thrust2 -> site motor2 at ( 0.00, +0.55)  -> +Y
#   thrust3 -> site motor3 at (-0.55,  0.00)  -> -X
#   thrust4 -> site motor4 at ( 0.00, -0.55)  -> -Y
#
# Every motor has gear="0 0 1 0 0 0": a pure force along the
# body +Z axis, ctrlrange 0..5 N, and NO yaw-torque term.

motor_names = [
    "thrust1",
    "thrust2",
    "thrust3",
    "thrust4"
]

motor_ids = []

for name in motor_names:

    actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        name
    )

    if actuator_id == -1:
        raise RuntimeError(
            f"Could not find actuator '{name}'"
        )

    motor_ids.append(actuator_id)


motor1_id = motor_ids[0]
motor2_id = motor_ids[1]
motor3_id = motor_ids[2]
motor4_id = motor_ids[3]

print(
    "Motor IDs:",
    motor1_id,
    motor2_id,
    motor3_id,
    motor4_id
)

# Verify the geometry assumed by the mixing below, straight from the model
site_names = ["motor1", "motor2", "motor3", "motor4"]
expected_dirs = [(+1, 0), (0, +1), (-1, 0), (0, -1)]

for site_name, (ex, ey), act_id in zip(site_names, expected_dirs, motor_ids):

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name
    )

    if site_id == -1:
        raise RuntimeError(f"Could not find site '{site_name}'")

    sx, sy = model.site_pos[site_id][:2]

    if not (np.sign(sx) == ex and np.sign(sy) == ey):
        raise RuntimeError(
            f"Site '{site_name}' is not where the motor mixing "
            f"expects it (found x={sx:.2f}, y={sy:.2f})"
        )

    if model.actuator_trnid[act_id][0] != site_id:
        raise RuntimeError(
            f"Actuator {act_id} is not attached to site '{site_name}'"
        )

motor_ctrl_min = float(model.actuator_ctrlrange[motor_ids[0]][0])
motor_ctrl_max = float(model.actuator_ctrlrange[motor_ids[0]][1])


# ============================================================
# INITIAL POSITION
# ============================================================

data.qpos[2] = 0.7

mujoco.mj_forward(model, data)


# ============================================================
# PHYSICAL PARAMETERS (read from the model)
# ============================================================

mass = float(np.sum(model.body_mass))
gravity = float(-model.opt.gravity[2])

total_hover_thrust = mass * gravity
motor_hover = total_hover_thrust / 4.0

print("Mass:", mass)
print("Total hover thrust:", total_hover_thrust)
print("Motor hover thrust:", motor_hover)


# ============================================================
# CONTROL PARAMETERS
# ============================================================

# ---- Horizontal flight (attitude) -------------------------

# Maximum commanded roll/pitch angle while an arrow is held (radians).
# 0.30 rad = about 17 degrees  ->  horizontal accel of roughly 3 m/s^2
max_angle = 0.30

# Attitude proportional gain
attitude_gain = 3.0

# Attitude rate (damping) gain
rate_gain = 0.5

# Maximum attitude correction (per-motor thrust difference, N)
attitude_control_limit = 0.5

# Horizontal air drag (N per m/s) on world X and Y motion.
# The XML has almost no drag, so without this the quadrotor would keep
# coasting forever after an arrow is released. With 1.0 the quad settles
# at roughly 2 m/s while tilted and slows to a stop in about 2-3 s after
# release. Set to 0.1 to get back the original (almost no drag) behaviour.
linear_drag = 1.0

model.dof_damping[root_dof + 0] = linear_drag
model.dof_damping[root_dof + 1] = linear_drag

# ---- Altitude hold ----------------------------------------

# SPACE held -> climb at this speed, SHIFT held -> descend at this speed
climb_speed = 1.0           # m/s

# Altitude PD gains (hold mode)
alt_kp = 6.0                # 1/s^2
alt_kd = 4.0                # 1/s

# Velocity-tracking gain used while SPACE / SHIFT is held
alt_kv = 4.0                # 1/s

# When SPACE/SHIFT is released, hold the height the quad will reach
# after this much extra coasting (prevents overshoot)
alt_lead = 0.25             # s

# Limits on commanded vertical acceleration (m/s^2)
alt_acc_min = -6.0
alt_acc_max = 8.0

# Lowest altitude the controller will command (m)
min_altitude = 0.25

# Altitude controller state
alt_target = float(data.qpos[root_dof + 2])
alt_was_moving = False


# ============================================================
# KEYBOARD STATE
# ============================================================

key_lock = threading.Lock()

# Set once any key event reaches the program (used for a startup self-check)
key_events_seen = False

forward_pressed = False
backward_pressed = False
left_pressed = False
right_pressed = False

# SPACE held -> climb,  SHIFT held -> descend
climb_pressed = False
descend_pressed = False

# Any of these keys counts as SHIFT (depends on platform / key layout)
SHIFT_KEYS = (
    keyboard.Key.shift,
    keyboard.Key.shift_l,
    keyboard.Key.shift_r,
)


# ============================================================
# KEY PRESS CALLBACK
# ============================================================

def on_press(key):

    global forward_pressed
    global backward_pressed
    global left_pressed
    global right_pressed
    global climb_pressed
    global descend_pressed
    global key_events_seen

    with key_lock:

        key_events_seen = True

        if key == keyboard.Key.space:

            climb_pressed = True

        elif key in SHIFT_KEYS:

            descend_pressed = True

        elif key == keyboard.Key.up:

            forward_pressed = True

        elif key == keyboard.Key.down:

            backward_pressed = True

        elif key == keyboard.Key.left:

            left_pressed = True

        elif key == keyboard.Key.right:

            right_pressed = True


# ============================================================
# KEY RELEASE CALLBACK
# ============================================================

def on_release(key):

    global forward_pressed
    global backward_pressed
    global left_pressed
    global right_pressed
    global climb_pressed
    global descend_pressed
    global key_events_seen

    with key_lock:

        key_events_seen = True

        if key == keyboard.Key.space:

            climb_pressed = False

        elif key in SHIFT_KEYS:

            descend_pressed = False

        elif key == keyboard.Key.up:

            forward_pressed = False

        elif key == keyboard.Key.down:

            backward_pressed = False

        elif key == keyboard.Key.left:

            left_pressed = False

        elif key == keyboard.Key.right:

            right_pressed = False


# ============================================================
# ATTITUDE / ALTITUDE / MOTOR CONTROLLER
# ============================================================

def compute_motor_commands(target_roll, target_pitch, climb_cmd):
    """
    Returns (f1, f2, f3, f4) motor thrust commands.

    climb_cmd: +1 = climb, -1 = descend, 0 = hold current altitude.

    Sign conventions (verified against MuJoCo's xmat):
      roll  = atan2(R[2,1], R[2,2])  -> rotation about body +X
      pitch = atan2(-R[2,0], ...)    -> rotation about body +Y

      +pitch tilts the thrust vector toward +X  (forward)
      +roll  tilts the thrust vector toward -Y  (right, with X forward, Z up)

    Torque of a motor force F (along +Z) at position (x, y):
      tau = r x (0, 0, F) = ( y*F, -x*F, 0 )

      M1 (+X): tau_y = -0.55*F1      M3 (-X): tau_y = +0.55*F3
      M2 (+Y): tau_x = +0.55*F2      M4 (-Y): tau_x = -0.55*F4

      +pitch torque (about +Y): F3 up, F1 down
      +roll  torque (about +X): F2 up, F4 down
    """

    global alt_target
    global alt_was_moving

    R = data.xmat[body_id].reshape(3, 3)

    roll = np.arctan2(
        R[2, 1],
        R[2, 2]
    )

    pitch = np.arctan2(
        -R[2, 0],
        np.sqrt(
            R[2, 1] ** 2 +
            R[2, 2] ** 2
        )
    )

    # Body-frame angular velocity (about body X and body Y)
    roll_rate = data.qvel[root_dof + 3]
    pitch_rate = data.qvel[root_dof + 4]

    # World-frame height and vertical speed
    z = data.qpos[root_dof + 2]
    vz = data.qvel[root_dof + 2]

    # ---------------- Attitude PD ----------------

    roll_control = (
        attitude_gain * (target_roll - roll)
        - rate_gain * roll_rate
    )

    pitch_control = (
        attitude_gain * (target_pitch - pitch)
        - rate_gain * pitch_rate
    )

    roll_control = np.clip(
        roll_control,
        -attitude_control_limit,
        attitude_control_limit
    )

    pitch_control = np.clip(
        pitch_control,
        -attitude_control_limit,
        attitude_control_limit
    )

    # ---------------- Altitude hold ----------------

    # Do not keep descending below the minimum altitude
    if climb_cmd < 0 and z <= min_altitude:
        climb_cmd = 0
        alt_was_moving = False
        alt_target = min_altitude

    if climb_cmd != 0:

        # Track a commanded vertical speed
        vz_cmd = climb_cmd * climb_speed
        az = alt_kv * (vz_cmd - vz)

        alt_target = z
        alt_was_moving = True

    else:

        if alt_was_moving:
            # Key just released: hold where we will naturally stop
            alt_target = max(z + alt_lead * vz, min_altitude)
            alt_was_moving = False

        # Hold the target altitude
        az = alt_kp * (alt_target - z) - alt_kd * vz

    az = np.clip(az, alt_acc_min, alt_acc_max)

    # Total vertical thrust needed (gravity + desired acceleration),
    # shared by 4 motors
    thrust = mass * (gravity + az) / 4.0

    # Keep the vertical thrust component correct while tilted
    compensation = np.cos(roll) * np.cos(pitch)
    compensation = max(compensation, 0.7)
    thrust = thrust / compensation

    # ---------------- Motor mixing (from the XML geometry) ----------------

    f1 = thrust - pitch_control   # +X
    f2 = thrust + roll_control    # +Y
    f3 = thrust + pitch_control   # -X
    f4 = thrust - roll_control    # -Y

    f1 = np.clip(f1, motor_ctrl_min, motor_ctrl_max)
    f2 = np.clip(f2, motor_ctrl_min, motor_ctrl_max)
    f3 = np.clip(f3, motor_ctrl_min, motor_ctrl_max)
    f4 = np.clip(f4, motor_ctrl_min, motor_ctrl_max)

    return f1, f2, f3, f4


def targets_from_keys(forward, backward, left, right, cam_azimuth_deg, yaw):
    """
    Arrow keys -> target roll/pitch, relative to the CAMERA view.

      UP    : fly away from the camera   (up the screen)
      DOWN  : fly toward the camera
      RIGHT : fly to the right of the screen
      LEFT  : fly to the left of the screen

    No keys -> level.

    MuJoCo camera convention (azimuth az, measured about world +Z):
      horizontal viewing direction = ( cos(az), sin(az) )
      screen-right direction       = ( sin(az), -cos(az) )

    cam_azimuth_deg : current viewer camera azimuth (degrees)
    yaw             : current quadrotor heading about world Z (radians)
    """

    az = np.radians(cam_azimuth_deg)

    view_dir = np.array([np.cos(az), np.sin(az)])
    screen_right = np.array([np.sin(az), -np.cos(az)])

    up_down = float(forward) - float(backward)
    right_left = float(right) - float(left)

    # Desired horizontal acceleration direction in the WORLD frame
    world_dir = up_down * view_dir + right_left * screen_right

    norm = np.linalg.norm(world_dir)

    if norm < 1e-9:
        return 0.0, 0.0

    # Same tilt magnitude for straight and diagonal flight
    world_dir = world_dir / norm

    # Rotate the world direction into the quadrotor body frame
    # (body x-axis = (cos yaw, sin yaw), body y-axis = (-sin yaw, cos yaw))
    body_x = np.cos(yaw) * world_dir[0] + np.sin(yaw) * world_dir[1]
    body_y = -np.sin(yaw) * world_dir[0] + np.cos(yaw) * world_dir[1]

    # +pitch -> thrust toward body +X ;  +roll -> thrust toward body -Y
    target_pitch = +max_angle * body_x
    target_roll = -max_angle * body_y

    return target_roll, target_pitch

# ============================================================
# BODY FRAME OVERLAY
# ============================================================

def update_body_frame_overlay(viewer):
    """
    Display the quadrotor body-frame position and
    rotation matrix in the top-left corner of the
    MuJoCo viewer.
    """

    # --------------------------------------------------------
    # Body position in WORLD frame
    # --------------------------------------------------------

    x, y, z = data.xpos[body_id]


    # --------------------------------------------------------
    # Body rotation matrix in WORLD frame
    # --------------------------------------------------------

    R = data.xmat[body_id].reshape(3, 3)


    # --------------------------------------------------------
    # Text shown in the viewer
    # --------------------------------------------------------

    text = (
        f"x = {x: .3f} m\n"
        f"y = {y: .3f} m\n"
        f"z = {z: .3f} m\n\n"
        f"ROTATION MATRIX R\n"
        f"[{R[0,0]: .3f} {R[0,1]: .3f} {R[0,2]: .3f}]\n"
        f"[{R[1,0]: .3f} {R[1,1]: .3f} {R[1,2]: .3f}]\n"
        f"[{R[2,0]: .3f} {R[2,1]: .3f} {R[2,2]: .3f}]"
    )


    # --------------------------------------------------------
    # Set overlay text
    # --------------------------------------------------------

    viewer.set_texts(
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "BODY FRAME",
            text
        )
    )

# ============================================================
# MAIN SIMULATION
# ============================================================

def main():

    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    )

    listener.daemon = True
    listener.start()

    try:

        with mujoco.viewer.launch_passive(
            model,
            data
        ) as viewer:

            print()
            print("========================================")
            print("QUADROTOR CONTROLLER STARTED")
            print("========================================")
            print("Arrows are relative to the camera view:")
            print("UP ARROW    : fly away from camera / up the screen (hold)")
            print("DOWN ARROW  : fly toward the camera                (hold)")
            print("LEFT ARROW  : fly to screen left                   (hold)")
            print("RIGHT ARROW : fly to screen right                  (hold)")
            print("SPACE       : climb         (hold)")
            print("SHIFT       : descend       (hold)")
            print("Release SPACE/SHIFT -> holds current altitude")
            print("Click the MuJoCo window once so it has keyboard focus.")
            print("Close the viewer window to exit.")
            print("========================================")
            print()

            warned = False
            loop_start = time.time()
            was_moving_printed = False

            while viewer.is_running():

                if (
                    not warned
                    and not key_events_seen
                    and time.time() - loop_start > 15.0
                ):
                    warned = True
                    print(
                        "WARNING: no keyboard events received yet. "
                        "Click the MuJoCo window to focus it. "
                        "Session type:",
                        os.environ.get("XDG_SESSION_TYPE", "unknown")
                    )

                step_start = time.perf_counter()

                # Copy keyboard state
                with key_lock:

                    forward = forward_pressed
                    backward = backward_pressed
                    left = left_pressed
                    right = right_pressed

                    climb = climb_pressed
                    descend = descend_pressed

                # SPACE alone -> +1, SHIFT alone -> -1, both/none -> 0
                if climb and not descend:
                    climb_cmd = +1
                elif descend and not climb:
                    climb_cmd = -1
                else:
                    climb_cmd = 0

                with viewer.lock():

                    # Live camera azimuth and quadrotor heading
                    cam_azimuth = float(viewer.cam.azimuth)

                    Rb = data.xmat[body_id].reshape(3, 3)
                    yaw = np.arctan2(Rb[1, 0], Rb[0, 0])

                    target_roll, target_pitch = targets_from_keys(
                        forward,
                        backward,
                        left,
                        right,
                        cam_azimuth,
                        yaw
                    )

                    f1, f2, f3, f4 = compute_motor_commands(
                        target_roll,
                        target_pitch,
                        climb_cmd
                    )

                    data.ctrl[motor1_id] = f1
                    data.ctrl[motor2_id] = f2
                    data.ctrl[motor3_id] = f3
                    data.ctrl[motor4_id] = f4

                    mujoco.mj_step(
                        model,
                        data
                    )

                    update_body_frame_overlay(viewer)

                # Print the held altitude once when SPACE/SHIFT is released
                if climb_cmd != 0:
                    was_moving_printed = True
                elif was_moving_printed:
                    was_moving_printed = False
                    print(f"Holding altitude: {alt_target:.2f} m")

                viewer.sync()

                # Real-time pacing
                elapsed = time.perf_counter() - step_start

                if elapsed < model.opt.timestep:

                    time.sleep(
                        model.opt.timestep - elapsed
                    )

    finally:

        listener.stop()


if __name__ == "__main__":
    main()