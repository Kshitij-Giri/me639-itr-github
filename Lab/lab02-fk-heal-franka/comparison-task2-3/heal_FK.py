import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# HEAL FORWARD KINEMATICS VALIDATION
# ============================================================
#
# This program calculates the HEAL end-effector position in
# TWO independent ways:
#
#   1. MuJoCo forward kinematics
#   2. Standard DH forward kinematics
#
# The two results are then compared.
#
#
# Overall flow:
#
#       Joint variables q1 ... q6
#                 |
#          ----------------
#          |              |
#          v              v
#       MuJoCo           DH
#          |              |
#          v              v
#       x,y,z          x,y,z
#          |              |
#          ------+--------
#                |
#                v
#          Difference
#
#
# The same joint variables are used in BOTH calculations.
# ============================================================


# ============================================================
# 1. JOINT VARIABLES
# ============================================================
#
# Same values as the original HEAL MuJoCo and HEAL DH codes.
#
# Units:
#       degrees
#
# These are converted to radians separately inside each
# calculation where required.

TARGET_Q_DEG = np.array([
     30.0,      # q1
    -20.0,      # q2
     25.0,      # q3
     20.0,      # q4
    -15.0,      # q5
     20.0       # q6
])


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
# 3. MUJOCO JOINT NAMES
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
# 4. MUJOCO SETTINGS
# ============================================================

MOVE_TIME = 5.0
DT = 0.01

SITE_NAME = "right_center"


# ============================================================
# 5. DH PARAMETERS
# ============================================================
#
# These are the same DH parameters used in heal_FK.py.
#
# Standard DH convention:
#
#       A_i =
#       Rot_z(theta_i)
#       Trans_z(d_i)
#       Trans_x(a_i)
#       Rot_x(alpha_i)
#
#
# The geometry parameters are fixed for the HEAL robot.
#
# The joint variables enter through:
#
#       theta_i = q_i + theta_offset_i
#
# ============================================================


# Link lengths a_i

DH_A = np.array([
     0.0,
    -0.3,
     0.0,
     0.0,
     0.0,
     0.0
])


# Link twists alpha_i

DH_ALPHA = np.array([
     np.pi / 2.0,
    -np.pi,
    -1.57,
     0.50951,
    -np.pi / 2.0,
     np.pi
])


# Link offsets d_i

DH_D = np.array([
     0.3208,
     1.1527768078,
     1.1526499530,
     0.3773557993,
     0.0001000926,
    -0.1227
])


# Constant angular offsets associated with our selected
# DH coordinate-frame convention.

DH_THETA_OFFSET = np.array([
    -np.pi,
    -np.pi / 2.0,
     0.0007963268,
     0.0,
     1.57,
     0.0
])


# ============================================================
# FUNCTION 1:
# STANDARD DH HOMOGENEOUS TRANSFORMATION
# ============================================================

def dh_matrix(a_i, alpha_i, d_i, theta_i):
    """
    Construct the standard DH homogeneous transformation.

    A_i =
        Rot_z(theta_i)
        Trans_z(d_i)
        Trans_x(a_i)
        Rot_x(alpha_i)
    """

    ct = np.cos(theta_i)
    st = np.sin(theta_i)

    ca = np.cos(alpha_i)
    sa = np.sin(alpha_i)

    A_i = np.array([
        [
            ct,
            -st * ca,
             st * sa,
             a_i * ct
        ],

        [
            st,
             ct * ca,
            -ct * sa,
             a_i * st
        ],

        [
            0.0,
             sa,
             ca,
             d_i
        ],

        [
            0.0,
             0.0,
             0.0,
             1.0
        ]
    ])

    return A_i


# ============================================================
# FUNCTION 2:
# DH FORWARD KINEMATICS
# ============================================================

def run_dh_fk(q_deg):
    """
    Calculate HEAL end-effector position using only
    standard DH forward kinematics.

    No MuJoCo is used here.

    Input:
        q_deg = [q1, q2, ..., q6] in degrees

    Output:
        [x, y, z] in metres
    """

    # Convert joint angles from degrees to radians.

    q = np.deg2rad(q_deg)

    # DH joint angles.

    theta = q + DH_THETA_OFFSET

    # --------------------------------------------------------
    # Construct A1 ... A6
    # --------------------------------------------------------

    A1 = dh_matrix(
        DH_A[0],
        DH_ALPHA[0],
        DH_D[0],
        theta[0]
    )

    A2 = dh_matrix(
        DH_A[1],
        DH_ALPHA[1],
        DH_D[1],
        theta[1]
    )

    A3 = dh_matrix(
        DH_A[2],
        DH_ALPHA[2],
        DH_D[2],
        theta[2]
    )

    A4 = dh_matrix(
        DH_A[3],
        DH_ALPHA[3],
        DH_D[3],
        theta[3]
    )

    A5 = dh_matrix(
        DH_A[4],
        DH_ALPHA[4],
        DH_D[4],
        theta[4]
    )

    A6 = dh_matrix(
        DH_A[5],
        DH_ALPHA[5],
        DH_D[5],
        theta[5]
    )

    # --------------------------------------------------------
    # Complete transformation:
    #
    #       T06 = A1 A2 A3 A4 A5 A6
    # --------------------------------------------------------

    T06 = (
        A1
        @ A2
        @ A3
        @ A4
        @ A5
        @ A6
    )

    # --------------------------------------------------------
    # End-effector is the origin of frame 6.
    #
    # Coordinates in frame 6:
    #
    #       [0]
    #       [0]
    #       [0]
    #       [1]
    #
    # --------------------------------------------------------

    end_effector_vector = np.array([
        0.0,
        0.0,
        0.0,
        1.0
    ])

    # Transform the point into the base frame.

    position = T06 @ end_effector_vector

    # Return only x, y, z.

    return position[:3]


# ============================================================
# FUNCTION 3:
# MUJOCO FORWARD KINEMATICS
# ============================================================

def run_mujoco_fk(q_deg):
    """
    Calculate HEAL end-effector position using MuJoCo.

    The robot starts at:
        q = [0, 0, 0, 0, 0, 0]

    It then moves smoothly to q_deg.

    The MuJoCo viewer remains open until the user closes it.

    The final position of the HEAL 'right_center' site is
    returned after the viewer is closed.

    Output:
        [x, y, z] in metres
    """

    # --------------------------------------------------------
    # Check XML
    # --------------------------------------------------------

    if not XML_PATH.exists():

        raise FileNotFoundError(
            f"\nCould not find HEAL XML file:\n"
            f"{XML_PATH}\n"
        )

    print("\nLoading HEAL MuJoCo model...")
    print(f"XML: {XML_PATH}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = mujoco.MjModel.from_xml_path(
        str(XML_PATH)
    )

    data = mujoco.MjData(model)

    # --------------------------------------------------------
    # Convert target angles to radians
    # --------------------------------------------------------

    target_q = np.deg2rad(q_deg)

    # --------------------------------------------------------
    # Find joint qpos addresses
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Initial configuration = all zeros
    # --------------------------------------------------------

    initial_q = np.zeros(6)

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

    # --------------------------------------------------------
    # Find end-effector site
    # --------------------------------------------------------

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        SITE_NAME
    )

    if site_id == -1:

        raise RuntimeError(
            f"End-effector site '{SITE_NAME}' "
            f"was not found."
        )

    # --------------------------------------------------------
    # Open MuJoCo viewer
    # --------------------------------------------------------

    print("\nOpening MuJoCo viewer...")
    print(
        "Robot will move from "
        "[0, 0, 0, 0, 0, 0]"
    )

    print(
        "to:",
        q_deg,
        "degrees"
    )

    print(
        "\nClose the MuJoCo window after "
        "the robot reaches the target."
    )

    with mujoco.viewer.launch_passive(
        model,
        data
    ) as viewer:

        start_time = time.time()

        while viewer.is_running():

            elapsed_time = (
                time.time() - start_time
            )

            # Normalized time 0 -> 1.

            s = min(
                elapsed_time / MOVE_TIME,
                1.0
            )

            # Smooth interpolation:
            #
            #       s_smooth = 3s² - 2s³

            s_smooth = (
                3.0 * s**2
                - 2.0 * s**3
            )

            # Current joint configuration.

            current_q = (
                initial_q
                + s_smooth
                * (target_q - initial_q)
            )

            # Set joint positions.

            for address, q in zip(
                qpos_addresses,
                current_q
            ):

                data.qpos[address] = q

            # Direct kinematic configuration:
            # velocities are zero.

            data.qvel[:] = 0.0

            # Recalculate forward kinematics.

            mujoco.mj_forward(
                model,
                data
            )

            # Update viewer.

            viewer.sync()

            # Once target is reached, hold it.

            if s >= 1.0:

                for address, q in zip(
                    qpos_addresses,
                    target_q
                ):

                    data.qpos[address] = q

                data.qvel[:] = 0.0

                mujoco.mj_forward(
                    model,
                    data
                )

                viewer.sync()

            time.sleep(DT)

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # The final position is obtained BEFORE the MuJoCo
        # data object is destroyed.
        # ----------------------------------------------------

        final_position = (
            data.site_xpos[site_id].copy()
        )

    return final_position


# ============================================================
# FUNCTION 4:
# COMPARE BOTH RESULTS
# ============================================================

def compare_results(mujoco_position, dh_position):
    """
    Compare the MuJoCo and DH positions.
    """

    # Coordinate-wise differences.

    difference = (
        mujoco_position
        - dh_position
    )

    # Absolute coordinate differences.

    absolute_difference = np.abs(
        difference
    )

    # Euclidean distance between the two
    # calculated positions.

    total_error = np.linalg.norm(
        difference
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("             HEAL FORWARD KINEMATICS")
    print("                 FINAL COMPARISON")
    print("=" * 65)

    print("\nJoint configuration:")
    print(
        "q =",
        TARGET_Q_DEG,
        "degrees"
    )

    print("\nMuJoCo end-effector position:")
    print(
        f"x = {mujoco_position[0]:.9f} m"
    )
    print(
        f"y = {mujoco_position[1]:.9f} m"
    )
    print(
        f"z = {mujoco_position[2]:.9f} m"
    )

    print("\nDH end-effector position:")
    print(
        f"x = {dh_position[0]:.9f} m"
    )
    print(
        f"y = {dh_position[1]:.9f} m"
    )
    print(
        f"z = {dh_position[2]:.9f} m"
    )

    print("\nDifference (MuJoCo - DH):")
    print(
        f"Δx = {difference[0]:+.9f} m"
    )
    print(
        f"Δy = {difference[1]:+.9f} m"
    )
    print(
        f"Δz = {difference[2]:+.9f} m"
    )

    print("\nAbsolute coordinate difference:")
    print(
        f"|Δx| = {absolute_difference[0]:.9f} m"
    )
    print(
        f"|Δy| = {absolute_difference[1]:.9f} m"
    )
    print(
        f"|Δz| = {absolute_difference[2]:.9f} m"
    )

    print("\nTotal position error:")
    print(
        f"||Δp|| = {total_error:.9f} m"
    )

    print("=" * 65)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # STEP 1:
    # Run MuJoCo
    # --------------------------------------------------------

    mujoco_position = run_mujoco_fk(
        TARGET_Q_DEG
    )

    # --------------------------------------------------------
    # STEP 2:
    # Run independent DH calculation
    #
    # This calculation uses NO MuJoCo.
    # --------------------------------------------------------

    dh_position = run_dh_fk(
        TARGET_Q_DEG
    )

    # --------------------------------------------------------
    # STEP 3:
    # Compare results
    # --------------------------------------------------------

    compare_results(
        mujoco_position,
        dh_position
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()