import os
os.environ["GLFW_PLATFORM"] = "x11"

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# 2. PATH TO FRANKA MUJOCO MODEL
# ============================================================

# Get the directory containing this Python file.
#
# Current structure:
#
# lab02-fk-heal-franka/
# ├── comparison-task2-3/
# │   └── franka_FK.py          <- this file
# │
# └── robot-descriptions/
#     └── franka/
#         └── mjx_panda.xml
#
# Therefore:
#
# CODE_DIR.parent = lab02-fk-heal-franka/

CODE_DIR = Path(__file__).resolve().parent

LAB_DIR = CODE_DIR.parent

ROBOT_DESCRIPTIONS_DIR = (
    LAB_DIR / "robot-descriptions"
)

# Supplied Franka MuJoCo model:
#
# robot-descriptions/
# └── franka/
#     └── mjx_panda.xml

XML_PATH = (
    ROBOT_DESCRIPTIONS_DIR
    / "franka"
    / "mjx_panda.xml"
)



# ============================================================
# TARGET JOINT ANGLES
# ============================================================
#
# Enter the seven Franka joint angles here in DEGREES.
#
# q1 -> joint1
# q2 -> joint2
# q3 -> joint3
# q4 -> joint4
# q5 -> joint5
# q6 -> joint6
# q7 -> joint7
#
# Change these values whenever you want to test
# another configuration.

TARGET_Q_DEG = np.array([
     20.0,      # q1
    -20.0,      # q2
     20.0,      # q3
    -90.0,      # q4
     20.0,      # q5
     90.0,      # q6
     20.0       # q7
])

TARGET_Q = np.deg2rad(TARGET_Q_DEG)


# ============================================================
# FRANKA ARM JOINT NAMES
# ============================================================

JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7"
]


# ============================================================
# FRAME 7 / END-EFFECTOR BODY
# ============================================================
#
# In the supplied mjx_panda.xml:
#
#     <body name="link7" ...>
#         ...
#         <joint name="joint7" .../>
#
# Therefore the origin of the "link7" body frame is the
# point corresponding to our seventh DH frame.
#
# We DO NOT use:
#
#     hand
#     gripper
#     left_finger
#     right_finger
#
# because those are downstream of frame 7.

FRAME_7_BODY_NAME = "link7"


# ============================================================
# MOTION SETTINGS
# ============================================================

MOVE_TIME = 5.0
DT = 0.01


# ============================================================
# LOAD FRANKA MODEL
# ============================================================

if not XML_PATH.exists():
    raise FileNotFoundError(
        f"Franka XML file not found:\n{XML_PATH}"
    )

model = mujoco.MjModel.from_xml_path(
    str(XML_PATH)
)

data = mujoco.MjData(model)


# ============================================================
# FIND THE SEVEN ARM JOINTS
# ============================================================

qpos_addresses = []

for joint_name in JOINT_NAMES:

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name
    )

    if joint_id == -1:
        raise RuntimeError(
            f"Joint '{joint_name}' was not found."
        )

    qpos_addresses.append(
        model.jnt_qposadr[joint_id]
    )


# ============================================================
# INITIAL CONFIGURATION
# ============================================================
#
# Required starting configuration:
#
# q1 = q2 = ... = q7 = 0
#
# The two gripper joints are not part of our 7-DOF
# kinematic chain and are left at their model defaults.

initial_q = np.zeros(7)

for address, q in zip(
    qpos_addresses,
    initial_q
):
    data.qpos[address] = q

data.qvel[:] = 0.0

mujoco.mj_forward(
    model,
    data
)


# ============================================================
# FIND FRAME 7
# ============================================================

frame_7_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    FRAME_7_BODY_NAME
)

if frame_7_body_id == -1:
    raise RuntimeError(
        "Body 'link7' was not found in the Franka model."
    )


# ============================================================
# INITIAL FRAME-7 POSITION
# ============================================================

initial_position = (
    data.xpos[frame_7_body_id].copy()
)


# ============================================================
# CREATE MUJOCO VIEWER
# ============================================================

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    viewer.cam.azimuth = 135
    viewer.cam.elevation = -20
    viewer.cam.distance = 2.5
    viewer.cam.lookat[:] = [
        0.0,
        0.0,
        0.5
    ]

    # --------------------------------------------------------
    # Move robot from zero configuration to target
    # --------------------------------------------------------

    start_time = time.time()

    while viewer.is_running():

        elapsed_time = (
            time.time() - start_time
        )

        # Normalized time:
        #
        # 0 -> starting position
        # 1 -> target position

        s = min(
            elapsed_time / MOVE_TIME,
            1.0
        )

        # Smooth start and stop:
        #
        # s_smooth = 3s^2 - 2s^3

        s_smooth = (
            3.0 * s**2
            - 2.0 * s**3
        )

        # Interpolate all seven joints.

        current_q = (
            initial_q
            + s_smooth
            * (TARGET_Q - initial_q)
        )

        # Set joint positions.

        for address, q in zip(
            qpos_addresses,
            current_q
        ):
            data.qpos[address] = q

        # We are directly specifying the configuration,
        # so velocities remain zero.

        data.qvel[:] = 0.0

        # Recalculate forward kinematics.

        mujoco.mj_forward(
            model,
            data
        )

        # Update viewer.

        viewer.sync()

        # ----------------------------------------------------
        # Keep final configuration fixed.
        # ----------------------------------------------------

        if s >= 1.0:

            for address, q in zip(
                qpos_addresses,
                TARGET_Q
            ):
                data.qpos[address] = q

            data.qvel[:] = 0.0

            mujoco.mj_forward(
                model,
                data
            )

            viewer.sync()

        time.sleep(DT)


# ============================================================
# FINAL FRAME-7 POSITION
# ============================================================
#
# IMPORTANT:
#
# data.xpos[frame_7_body_id]
#
# gives the WORLD-FRAME position of the origin of the
# "link7" body frame.
#
# This is the point corresponding to our seventh DH frame.
#
# It is NOT:
#
# - the hand position
# - the gripper position
# - the gripper site
# - the fingertip position

final_position = (
    data.xpos[frame_7_body_id].copy()
)


# ============================================================
# PRINT ONLY FINAL X, Y, Z
# ============================================================

print(
    f"x = {final_position[0]:.6f} m"
)

print(
    f"y = {final_position[1]:.6f} m"
)

print(
    f"z = {final_position[2]:.6f} m"
)