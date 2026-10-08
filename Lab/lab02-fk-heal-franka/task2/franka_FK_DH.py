import numpy as np


# ============================================================
# FRANKA PANDA FORWARD KINEMATICS
# ============================================================
#
# This program calculates the forward kinematics of the
# 7-DOF Franka Panda arm WITHOUT using MuJoCo.
#
# Only:
#       NumPy
#       Standard Denavit-Hartenberg parameters
#       Homogeneous transformation matrices
#
# are used.
#
#
# ============================================================
# END-EFFECTOR DEFINITION
# ============================================================
#
# For this assignment, the end effector is defined as the
# ORIGIN OF FRAME 7.
#
# Therefore:
#
#       p_7 = [0, 0, 0, 1]^T
#
# in frame 7.
#
# We DO NOT use:
#
#       hand
#       gripper
#       finger
#       gripper site
#
# The supplied MJCF model has:
#
#       link7
#           |
#           +-- joint7
#           |
#           +-- hand at pos = (0, 0, 0.107)
#
# Therefore the 0.107 m hand offset is NOT included.
#
# Our forward kinematics ends at:
#
#       T_0_7
#
# and:
#
#       p_0 = T_0_7 [0, 0, 0, 1]^T
#
# ============================================================


# ============================================================
# 1. JOINT VARIABLES
# ============================================================
#
# These are the same joint variables used in the previous
# Franka MuJoCo experiment.
#
# Enter the joint angles in DEGREES here.
#
# The program converts them to radians automatically.
#
# q1 -> joint1
# q2 -> joint2
# q3 -> joint3
# q4 -> joint4
# q5 -> joint5
# q6 -> joint6
# q7 -> joint7

Q_DEG = np.array([
     20.0,       # q1
    -20.0,       # q2
     20.0,       # q3
    -90.0,       # q4
     20.0,       # q5
     90.0,       # q6
     20.0        # q7
])


# Convert degrees to radians.

Q = np.deg2rad(Q_DEG)


# ============================================================
# 2. DERIVATION OF THE DH PARAMETERS FROM THE XML
# ============================================================
#
# The supplied Franka MJCF contains the following fixed
# kinematic geometry:
#
# link1:
#       pos = (0, 0, 0.333)
#       joint1 axis = z
#
# link2:
#       quat = (1, -1, 0, 0)
#       joint2 is therefore perpendicular to joint1.
#
# link3:
#       pos = (0, -0.316, 0)
#       quat = (1, 1, 0, 0)
#       joint3 axis is again perpendicular to joint2.
#
# link4:
#       pos = (0.0825, 0, 0)
#       quat = (1, 1, 0, 0)
#
# link5:
#       pos = (-0.0825, 0.384, 0)
#       quat = (1, -1, 0, 0)
#
# link6:
#       quat = (1, 1, 0, 0)
#
# link7:
#       pos = (0.088, 0, 0)
#       quat = (1, 1, 0, 0)
#       joint7 is inside link7.
#
#
# The body quaternions rotate the local z-axis of each body.
# Consequently, the six consecutive joint-axis directions
# used for constructing the DH frames are:
#
#       joint 1 :  +Z
#       joint 2 :  +Y
#       joint 3 :  +Z
#       joint 4 :  -Y
#       joint 5 :  +Z
#       joint 6 :  -Y
#       joint 7 :  -Z
#
#
# The standard DH x-axes are chosen along the common normals
# between consecutive joint axes.
#
# From the XML geometry, the resulting distances are:
#
#       0.333 m  -> base offset
#       0.316 m  -> shoulder/elbow offset
#       0.0825 m -> first transverse link offset
#       0.0825 m -> second transverse link offset
#       0.384 m  -> forearm offset
#       0.088 m  -> distance from joint 6 to joint 7
#
#
# This gives the following STANDARD DH table:
#
#
#     i       a_i       alpha_i       d_i       theta_i
#
#     1       0         +pi/2         0.333     q1 + pi
#     2       0         +pi/2         0         q2 + pi
#     3       0.0825    +pi/2         0.316     q3
#     4       0.0825    +pi/2         0         q4 + pi
#     5       0         +pi/2         0.384     q5 + pi
#     6       0.088     +pi/2         0         q6
#     7       0         0             0         q7
#
#
# IMPORTANT:
#
# DH frames are not unique.
#
# Therefore these parameters can look different from another
# Panda DH table found online while describing exactly the
# same physical robot.
#
# The important requirement is that the chosen DH frames and
# transformations reproduce the actual kinematic chain.
#
# The constant pi terms in theta are frame-orientation
# offsets resulting from our particular frame assignment.
#
# They are NOT additional physical joint rotations.
# ============================================================


# Link lengths a_i.

a = np.array([
    0.0,
    0.0,
    0.0825,
    0.0825,
    0.0,
    0.088,
    0.0
])


# Link twists alpha_i.

alpha = np.array([
    np.pi / 2.0,
    np.pi / 2.0,
    np.pi / 2.0,
    np.pi / 2.0,
    np.pi / 2.0,
    np.pi / 2.0,
    0.0
])


# Link offsets d_i.

d = np.array([
    0.333,
    0.0,
    0.316,
    0.0,
    0.384,
    0.0,
    0.0
])


# Constant angular offsets introduced by the chosen DH frames.

theta_offset = np.array([
    np.pi,
    np.pi,
    0.0,
    np.pi,
    np.pi,
    0.0,
    0.0
])


# ============================================================
# 3. STANDARD DH TRANSFORMATION
# ============================================================
#
# We use the STANDARD DH convention:
#
#
#       A_i =
#       Rot_z(theta_i)
#       Trans_z(d_i)
#       Trans_x(a_i)
#       Rot_x(alpha_i)
#
#
# Therefore:
#
#
#       | cos(theta)  -sin(theta)cos(alpha)
#       |              sin(theta)sin(alpha)
#       |
#       | sin(theta)   cos(theta)cos(alpha)
#       |             -cos(theta)sin(alpha)
#       |
#       | 0            sin(alpha)
#       | 0            0
#
#
#       | a*cos(theta) |
#       | a*sin(theta) |
#       | d             |
#       | 1             |
#
# ============================================================

def dh_matrix(a_i, alpha_i, d_i, theta_i):

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
# 4. JOINT ANGLES USED IN THE DH MATRICES
# ============================================================
#
# The actual user-supplied joint variables are q1 ... q7.
#
# The DH theta values are:
#
#       theta_i = q_i + theta_offset_i
#
# ============================================================

theta = Q + theta_offset


# ============================================================
# 5. CALCULATE A1 ... A7
# ============================================================

A1 = dh_matrix(
    a[0],
    alpha[0],
    d[0],
    theta[0]
)

A2 = dh_matrix(
    a[1],
    alpha[1],
    d[1],
    theta[1]
)

A3 = dh_matrix(
    a[2],
    alpha[2],
    d[2],
    theta[2]
)

A4 = dh_matrix(
    a[3],
    alpha[3],
    d[3],
    theta[3]
)

A5 = dh_matrix(
    a[4],
    alpha[4],
    d[4],
    theta[4]
)

A6 = dh_matrix(
    a[5],
    alpha[5],
    d[5],
    theta[5]
)

A7 = dh_matrix(
    a[6],
    alpha[6],
    d[6],
    theta[6]
)


# ============================================================
# 6. COMPLETE FORWARD TRANSFORMATION
# ============================================================
#
# The complete transformation is:
#
#       T_0_7 =
#
#       A1 A2 A3 A4 A5 A6 A7
#
# This gives the pose of frame 7 relative to the base frame.
#
# ============================================================

T01 = A1

T02 = A1 @ A2

T03 = A1 @ A2 @ A3

T04 = A1 @ A2 @ A3 @ A4

T05 = A1 @ A2 @ A3 @ A4 @ A5

T06 = A1 @ A2 @ A3 @ A4 @ A5 @ A6

T07 = A1 @ A2 @ A3 @ A4 @ A5 @ A6 @ A7


# ============================================================
# 7. END-EFFECTOR VECTOR
# ============================================================
#
# Our end effector is the ORIGIN OF FRAME 7.
#
# Therefore, expressed in frame 7:
#
#       p_7 =
#
#       [ 0 ]
#       [ 0 ]
#       [ 0 ]
#       [ 1 ]
#
# ============================================================

end_effector_vector = np.array([
    0.0,
    0.0,
    0.0,
    1.0
])


# ============================================================
# 8. TRANSFORM END-EFFECTOR POINT TO BASE FRAME
# ============================================================
#
#       p_0 = T_0_7 p_7
#
# ============================================================

end_effector_position = (
    T07 @ end_effector_vector
)


# ============================================================
# 9. EXTRACT CARTESIAN COORDINATES
# ============================================================
#
#       x = p_0[0]
#       y = p_0[1]
#       z = p_0[2]
#
# ============================================================

x = end_effector_position[0]

y = end_effector_position[1]

z = end_effector_position[2]


# ============================================================
# 10. FINAL OUTPUT
# ============================================================
#
# Only the final end-effector coordinates are printed.
#
# ============================================================

print(f"x = {x:.6f} m")
print(f"y = {y:.6f} m")
print(f"z = {z:.6f} m")