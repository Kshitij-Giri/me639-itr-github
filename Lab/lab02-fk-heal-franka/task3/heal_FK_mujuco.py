import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np



# ============================================================
# 2. MUJOCO MODEL PATH
# ============================================================

# This file is located at:
#
# lab02-fk-heal-franka/
# └── comparison-task2&3/
#     └── heal_FK.py
#
# The HEAL MuJoCo model is located at:
#
# lab02-fk-heal-franka/
# └── robot-descriptions/
#     └── single_arm_heal_effort_actuation_rs_mj_2.xml
#
# Therefore:
#
#     CODE_DIR.parent
#         -> lab02-fk-heal-franka/
#
#     LAB_DIR / "robot-descriptions"
#         -> lab02-fk-heal-franka/robot-descriptions/
#
# No absolute path or user-specific directory is used.

CODE_DIR = Path(__file__).resolve().parent

LAB_DIR = CODE_DIR.parent

ROBOT_DESCRIPTIONS_DIR = (
    LAB_DIR / "robot-descriptions"
)

XML_PATH = (
    ROBOT_DESCRIPTIONS_DIR
    / "single_arm_heal_effort_actuation_rs_mj_2.xml"
)
# ============================================================
# 2. TARGET JOINT CONFIGURATION
# ============================================================
#
# Enter the desired joint angles here in DEGREES.
#
# Order:
# q1 -> joint_1
# q2 -> joint_2
# q3 -> joint_3
# q4 -> joint_4
# q5 -> joint_5
# q6 -> joint_6
#
# You can change these values for different experiments.

TARGET_Q_DEG = np.array([
    30.0,    # q1
   -20.0,    # q2
    25.0,    # q3
    20.0,    # q4
   -15.0,    # q5
    20.0     # q6
])

# Convert degrees to radians because MuJoCo model uses radians.
TARGET_Q = np.deg2rad(TARGET_Q_DEG)


# ============================================================
# 3. JOINT NAMES
# ============================================================

JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6"
]


# ============================================================
# 4. MOTION SETTINGS
# ============================================================

# Time taken to move from zero configuration
# to the desired configuration.
MOVE_TIME = 5.0       # seconds

# Simulation/visualization update interval.
DT = 0.01             # seconds


# ============================================================
# 5. LOAD MUJOCO MODEL
# ============================================================

if not XML_PATH.exists():
    raise FileNotFoundError(
        f"Could not find HEAL XML file:\n{XML_PATH}"
    )

model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)


# ============================================================
# 6. FIND THE JOINT QPOS ADDRESSES
# ============================================================

joint_ids = []
qpos_addresses = []

for joint_name in JOINT_NAMES:

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name
    )

    if joint_id == -1:
        raise RuntimeError(
            f"Joint '{joint_name}' was not found in the model."
        )

    joint_ids.append(joint_id)

    # Address of this joint in data.qpos
    qpos_addresses.append(
        model.jnt_qposadr[joint_id]
    )


# ============================================================
# 7. CHECK NUMBER OF JOINTS
# ============================================================

if len(JOINT_NAMES) != 6:
    raise RuntimeError("HEAL should have 6 joints.")


# ============================================================
# 8. INITIAL CONFIGURATION = ALL JOINTS ZERO
# ============================================================

initial_q = np.zeros(6)

for address, q in zip(qpos_addresses, initial_q):
    data.qpos[address] = q

# Zero joint velocities.
data.qvel[:] = 0.0

# Recalculate all MuJoCo kinematics.
mujoco.mj_forward(model, data)


# ============================================================
# 9. FIND END-EFFECTOR SITE
# ============================================================

# The HEAL XML contains:
#
# <site name="right_center" pos="0 0 0" .../>
#
# inside the end_effector body.

SITE_NAME = "right_center"

site_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    SITE_NAME
)

if site_id == -1:
    raise RuntimeError(
        f"End-effector site '{SITE_NAME}' was not found."
    )

# ============================================================
# 11. OPEN MUJOCO VIEWER
# ============================================================

with mujoco.viewer.launch_passive(model, data) as viewer:

    # --------------------------------------------------------
    # Move from zero configuration to target configuration
    # --------------------------------------------------------

    start_time = time.time()

    while viewer.is_running():

        elapsed_time = time.time() - start_time

        # Normalized time from 0 to 1.
        s = min(elapsed_time / MOVE_TIME, 1.0)

        # Smooth interpolation.
        #
        # This gives a smooth start and smooth stop:
        #
        # s_smooth = 3s^2 - 2s^3
        #
        # rather than an abrupt linear movement.

        s_smooth = 3.0 * s**2 - 2.0 * s**3

        current_q = (
            initial_q
            + s_smooth * (TARGET_Q - initial_q)
        )

        # Set the six joint positions.
        for address, q in zip(
            qpos_addresses,
            current_q
        ):
            data.qpos[address] = q

        # Stop any joint velocity because we are directly
        # specifying the kinematic configuration.
        data.qvel[:] = 0.0

        # Recalculate MuJoCo forward kinematics.
        mujoco.mj_forward(model, data)

        # Update visualization.
        viewer.sync()

        # Once target is reached, keep it there.
        if s >= 1.0:

            for address, q in zip(
                qpos_addresses,
                TARGET_Q
            ):
                data.qpos[address] = q

            data.qvel[:] = 0.0

            mujoco.mj_forward(model, data)
            viewer.sync()

        time.sleep(DT)


# ============================================================
# 12. FINAL END-EFFECTOR POSITION
# ============================================================

final_position = data.site_xpos[site_id].copy()

print(
    f"x = {final_position[0]:.6f} m"
)

print(
    f"y = {final_position[1]:.6f} m"
)

print(
    f"z = {final_position[2]:.6f} m"
)