import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# FRANKA FORWARD KINEMATICS VALIDATION
# ============================================================
#
# This program calculates the position of the Franka frame-7
# end effector in TWO independent ways:
#
#       1. MuJoCo forward kinematics
#       2. Standard DH forward kinematics
#
# Then the two results are compared.
#
#
#                 Joint variables
#                       |
#                q1 ... q7
#                       |
#             -------------------
#             |                 |
#             v                 v
#          MuJoCo              DH
#             |                 |
#             v                 v
#           x,y,z             x,y,z
#             |                 |
#             --------+----------
#                      |
#                      v
#                  Difference
#
#
# IMPORTANT:
#
# The end effector for this lab is the ORIGIN OF FRAME 7.
#
# Therefore we DO NOT use:
#
#       hand
#       gripper
#       fingers
#       fingertips
#
# The MuJoCo reference point is the origin of the "link7"
# body frame.
#
# The DH calculation also terminates at frame 7.
# ============================================================


# ============================================================
# 1. TARGET JOINT CONFIGURATION
# ============================================================
#
# Joint angles are specified in DEGREES.
#
# q1 -> joint1
# q2 -> joint2
# q3 -> joint3
# q4 -> joint4
# q5 -> joint5
# q6 -> joint6
# q7 -> joint7
#
# These are exactly the same values used in the individual
# Franka MuJoCo and DH programs.
# ============================================================

TARGET_Q_DEG = np.array([
     20.0,       # q1
    -20.0,       # q2
     20.0,       # q3
    -90.0,       # q4
     20.0,       # q5
     90.0,       # q6
     20.0        # q7
])


# Convert degrees to radians for MuJoCo/DH calculations.

TARGET_Q = np.deg2rad(TARGET_Q_DEG)


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
# 3. FRANKA JOINT NAMES
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
# 4. FRAME-7 DEFINITION
# ============================================================
#
# We use the origin of the link7 body as the end effector.
#
# In the supplied MJCF:
#
#       link7
#           |
#           +-- joint7
#           |
#           +-- hand
#
# The hand/gripper is downstream of frame 7 and therefore
# must NOT be included.
# ============================================================

FRAME_7_BODY_NAME = "link7"


# ============================================================
# 5. MUJOCO MOTION SETTINGS
# ============================================================

MOVE_TIME = 5.0       # seconds
DT = 0.01             # viewer update interval


# ============================================================
# 6. STANDARD DH PARAMETERS
# ============================================================
#
# These are the same DH parameters used in the supplied
# Franka DH code.
#
#
# Standard DH transformation:
#
#       A_i =
#       Rot_z(theta_i)
#       Trans_z(d_i)
#       Trans_x(a_i)
#       Rot_x(alpha_i)
#
#
# The joint angle used by each DH matrix is:
#
#       theta_i = q_i + theta_offset_i
#
# ============================================================


# ------------------------------------------------------------
# Link lengths a_i
# ------------------------------------------------------------

DH_A = np.array([
    0.0,
    0.0,
    0.0825,
    0.0825,
    0.0,
    0.088,
    0.0
])


# ------------------------------------------------------------
# Link twists alpha_i
# ------------------------------------------------------------

DH_ALPHA = np.array([
    np.pi / 2.0,
    np.pi / 2.0,
    np.pi / 2.0,
    np.pi / 2.0,
    np.pi / 2.0,
    np.pi / 2.0,
    0.0
])


# ------------------------------------------------------------
# Link offsets d_i
# ------------------------------------------------------------

DH_D = np.array([
    0.333,
    0.0,
    0.316,
    0.0,
    0.384,
    0.0,
    0.0
])


# ------------------------------------------------------------
# Constant theta offsets
# ------------------------------------------------------------
#
# These are offsets caused by the particular DH frame
# convention used in the analytical model.
#
# They are NOT additional physical joint commands.
# ------------------------------------------------------------

DH_THETA_OFFSET = np.array([
    np.pi,
    np.pi,
    0.0,
    np.pi,
    np.pi,
    0.0,
    0.0
])


# ============================================================
# FUNCTION 1:
# STANDARD DH TRANSFORMATION MATRIX
# ============================================================

def dh_matrix(a_i, alpha_i, d_i, theta_i):
    """
    Return the standard DH homogeneous transformation matrix.

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
    Calculate the frame-7 position using standard DH.

    MuJoCo is NOT used in this function.

    Parameters
    ----------
    q_deg : array-like
        Seven joint angles in degrees.

    Returns
    -------
    numpy.ndarray
        [x, y, z] in metres.
    """

    # --------------------------------------------------------
    # Convert joint variables from degrees to radians.
    # --------------------------------------------------------

    q = np.deg2rad(q_deg)


    # --------------------------------------------------------
    # Calculate DH theta values.
    #
    #       theta_i = q_i + theta_offset_i
    # --------------------------------------------------------

    theta = q + DH_THETA_OFFSET


    # --------------------------------------------------------
    # Calculate A1 ... A7.
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

    A7 = dh_matrix(
        DH_A[6],
        DH_ALPHA[6],
        DH_D[6],
        theta[6]
    )


    # --------------------------------------------------------
    # Complete transformation:
    #
    #       T07 = A1 A2 A3 A4 A5 A6 A7
    # --------------------------------------------------------

    T01 = A1

    T02 = A1 @ A2

    T03 = A1 @ A2 @ A3

    T04 = A1 @ A2 @ A3 @ A4

    T05 = A1 @ A2 @ A3 @ A4 @ A5

    T06 = A1 @ A2 @ A3 @ A4 @ A5 @ A6

    T07 = A1 @ A2 @ A3 @ A4 @ A5 @ A6 @ A7


    # --------------------------------------------------------
    # Frame-7 origin in homogeneous coordinates.
    #
    #       p7 = [0 0 0 1]^T
    #
    # --------------------------------------------------------

    end_effector_vector = np.array([
        0.0,
        0.0,
        0.0,
        1.0
    ])


    # --------------------------------------------------------
    # Transform frame-7 origin into the base frame.
    #
    #       p0 = T07 p7
    # --------------------------------------------------------

    end_effector_position = (
        T07 @ end_effector_vector
    )


    # Return x, y, z.

    return end_effector_position[:3]


# ============================================================
# FUNCTION 3:
# MUJOCO FORWARD KINEMATICS
# ============================================================

def run_mujoco_fk(q_deg):
    """
    Calculate the frame-7 position using MuJoCo.

    The robot starts at:
        [0, 0, 0, 0, 0, 0, 0]

    and smoothly moves to q_deg.

    The MuJoCo viewer remains open until the user closes it.

    Returns
    -------
    numpy.ndarray
        [x, y, z] in metres.
    """

    # --------------------------------------------------------
    # Check that the XML exists.
    # --------------------------------------------------------

    if not XML_PATH.exists():

        raise FileNotFoundError(
            f"\nFranka XML file not found:\n"
            f"{XML_PATH}\n"
        )


    print("\nLoading Franka MuJoCo model...")
    print(f"XML: {XML_PATH}")


    # --------------------------------------------------------
    # Load MuJoCo model.
    # --------------------------------------------------------

    model = mujoco.MjModel.from_xml_path(
        str(XML_PATH)
    )

    data = mujoco.MjData(model)


    # --------------------------------------------------------
    # Convert target joint angles to radians.
    # --------------------------------------------------------

    target_q = np.deg2rad(q_deg)


    # --------------------------------------------------------
    # Find qpos address for every arm joint.
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
                f"Joint '{joint_name}' was not found "
                f"in the Franka model."
            )


        qpos_addresses.append(
            model.jnt_qposadr[joint_id]
        )


    # --------------------------------------------------------
    # Initial configuration:
    #
    #       q1 = q2 = ... = q7 = 0
    #
    # The gripper joints are not part of the 7-DOF chain.
    # --------------------------------------------------------

    initial_q = np.zeros(7)


    for address, q in zip(
        qpos_addresses,
        initial_q
    ):

        data.qpos[address] = q


    data.qvel[:] = 0.0


    # Recalculate MuJoCo kinematics.

    mujoco.mj_forward(
        model,
        data
    )


    # --------------------------------------------------------
    # Find link7 body.
    # --------------------------------------------------------

    frame_7_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        FRAME_7_BODY_NAME
    )


    if frame_7_body_id == -1:

        raise RuntimeError(
            "Body 'link7' was not found "
            "in the Franka model."
        )


    # --------------------------------------------------------
    # Initial frame-7 position.
    # --------------------------------------------------------

    initial_position = (
        data.xpos[frame_7_body_id].copy()
    )


    # Prevent unused-variable warnings.
    _ = initial_position


    # --------------------------------------------------------
    # Open MuJoCo viewer.
    # --------------------------------------------------------

    print("\nOpening MuJoCo viewer...")

    print(
        "Robot will move from:"
    )

    print(
        "[0, 0, 0, 0, 0, 0, 0] degrees"
    )

    print(
        "to:"
    )

    print(
        q_deg,
        "degrees"
    )

    print(
        "\nClose the MuJoCo window after "
        "the robot reaches the target."
    )


    # --------------------------------------------------------
    # Run viewer.
    # --------------------------------------------------------

    with mujoco.viewer.launch_passive(
        model,
        data
    ) as viewer:

        # ----------------------------------------------------
        # Camera settings.
        # ----------------------------------------------------

        viewer.cam.azimuth = 135

        viewer.cam.elevation = -20

        viewer.cam.distance = 2.5

        viewer.cam.lookat[:] = [
            0.0,
            0.0,
            0.5
        ]


        # ----------------------------------------------------
        # Start interpolation.
        # ----------------------------------------------------

        start_time = time.time()


        while viewer.is_running():

            elapsed_time = (
                time.time()
                - start_time
            )


            # ------------------------------------------------
            # Normalized time.
            #
            # 0 -> initial configuration
            # 1 -> target configuration
            # ------------------------------------------------

            s = min(
                elapsed_time / MOVE_TIME,
                1.0
            )


            # ------------------------------------------------
            # Smooth interpolation.
            #
            #       s_smooth = 3s² - 2s³
            #
            # Gives smooth acceleration and deceleration.
            # ------------------------------------------------

            s_smooth = (
                3.0 * s**2
                - 2.0 * s**3
            )


            # ------------------------------------------------
            # Current seven-joint configuration.
            # ------------------------------------------------

            current_q = (
                initial_q
                + s_smooth
                * (target_q - initial_q)
            )


            # ------------------------------------------------
            # Set the seven arm joint positions.
            # ------------------------------------------------

            for address, q in zip(
                qpos_addresses,
                current_q
            ):

                data.qpos[address] = q


            # We are directly specifying positions.

            data.qvel[:] = 0.0


            # ------------------------------------------------
            # Recalculate forward kinematics.
            # ------------------------------------------------

            mujoco.mj_forward(
                model,
                data
            )


            # ------------------------------------------------
            # Update viewer.
            # ------------------------------------------------

            viewer.sync()


            # ------------------------------------------------
            # Once target is reached, hold it there.
            # ------------------------------------------------

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
        # Take the position while model/data are still valid.
        #
        # data.xpos[frame_7_body_id] is the WORLD-frame
        # position of the origin of link7.
        # ----------------------------------------------------

        final_position = (
            data.xpos[
                frame_7_body_id
            ].copy()
        )


    # Return after viewer closes.

    return final_position


# ============================================================
# FUNCTION 4:
# COMPARE MUJOCO AND DH RESULTS
# ============================================================

def compare_results(
    mujoco_position,
    dh_position
):
    """
    Compare the MuJoCo and DH end-effector positions.
    """

    # --------------------------------------------------------
    # Difference:
    #
    #       MuJoCo - DH
    # --------------------------------------------------------

    difference = (
        mujoco_position
        - dh_position
    )


    # Absolute coordinate-wise difference.

    absolute_difference = np.abs(
        difference
    )


    # --------------------------------------------------------
    # Euclidean position error:
    #
    #       ||Delta p||
    #
    # = sqrt(dx^2 + dy^2 + dz^2)
    # --------------------------------------------------------

    total_error = np.linalg.norm(
        difference
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")

    print("=" * 70)

    print(
        "             FRANKA FORWARD KINEMATICS"
    )

    print(
        "                  FINAL COMPARISON"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Joint configuration
    # --------------------------------------------------------

    print("\nJoint configuration:")

    print(
        "q =",
        TARGET_Q_DEG,
        "degrees"
    )


    # --------------------------------------------------------
    # MuJoCo result
    # --------------------------------------------------------

    print(
        "\nMuJoCo frame-7 position:"
    )

    print(
        f"x = {mujoco_position[0]:.9f} m"
    )

    print(
        f"y = {mujoco_position[1]:.9f} m"
    )

    print(
        f"z = {mujoco_position[2]:.9f} m"
    )


    # --------------------------------------------------------
    # DH result
    # --------------------------------------------------------

    print(
        "\nDH frame-7 position:"
    )

    print(
        f"x = {dh_position[0]:.9f} m"
    )

    print(
        f"y = {dh_position[1]:.9f} m"
    )

    print(
        f"z = {dh_position[2]:.9f} m"
    )


    # --------------------------------------------------------
    # Signed difference
    # --------------------------------------------------------

    print(
        "\nDifference (MuJoCo - DH):"
    )

    print(
        f"Delta x = {difference[0]:+.9f} m"
    )

    print(
        f"Delta y = {difference[1]:+.9f} m"
    )

    print(
        f"Delta z = {difference[2]:+.9f} m"
    )


    # --------------------------------------------------------
    # Absolute difference
    # --------------------------------------------------------

    print(
        "\nAbsolute coordinate difference:"
    )

    print(
        f"|Delta x| = "
        f"{absolute_difference[0]:.9f} m"
    )

    print(
        f"|Delta y| = "
        f"{absolute_difference[1]:.9f} m"
    )

    print(
        f"|Delta z| = "
        f"{absolute_difference[2]:.9f} m"
    )


    # --------------------------------------------------------
    # Total Euclidean error
    # --------------------------------------------------------

    print(
        "\nTotal position error:"
    )

    print(
        f"||Delta p|| = "
        f"{total_error:.9f} m"
    )


    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # STEP 1:
    # Run MuJoCo FK
    # ========================================================

    mujoco_position = run_mujoco_fk(
        TARGET_Q_DEG
    )


    # ========================================================
    # STEP 2:
    # Run analytical DH FK
    #
    # This function does NOT use MuJoCo.
    # ========================================================

    dh_position = run_dh_fk(
        TARGET_Q_DEG
    )


    # ========================================================
    # STEP 3:
    # Compare both results
    # ========================================================

    compare_results(
        mujoco_position,
        dh_position
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()