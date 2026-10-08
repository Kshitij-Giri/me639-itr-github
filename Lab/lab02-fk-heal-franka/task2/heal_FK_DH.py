import numpy as np


# ============================================================
# HEAL FORWARD KINEMATICS USING STANDARD DH PARAMETERS
# ============================================================
#
# This code does NOT use MuJoCo.
#
# The DH geometry below was derived from the actual HEAL MJCF
# model supplied for this lab.
#
# The XML defines the six revolute joints and the fixed
# translations/rotations between their bodies. In particular:
#
#   link_1:
#       pos = (0, 0, 0.171)
#       joint_1 axis = (0, 0, 1)
#
#   link_2:
#       pos = (0, 0.0875, 0.1498)
#       quat = (0.707105, 0.707108, 0, 0)
#       joint_2 axis = (0, 0, -1)
#
#   link_3:
#       pos = (0, 0.3, 0)
#       euler = (0, 0, -1.57)
#       joint_3 axis = (0, 0, 1)
#
#   link_4:
#       pos = (0, 0.1593, 0.0875)
#       euler = (-1.57, 0, 0)
#       joint_4 axis = (0, 0, 1)
#
#   link_5:
#       pos = (0, 0.03185, 0.16105)
#       euler = (0.50951, 0, 1.57)
#       joint_5 axis = (0, 0, 1)
#
#   end_effector:
#       pos = (0, -0.1227, 0.0654)
#       quat = (0.707105, 0.707108, 0, 0)
#       joint_6 axis = (0, 0, -1)
#
# The XML therefore gives us the complete fixed kinematic
# geometry and the direction of every revolute joint axis.
#
# We convert that fixed geometry into an equivalent STANDARD
# Denavit-Hartenberg representation.
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
# Therefore:
#
#       T_0_6 = A_1 A_2 A_3 A_4 A_5 A_6
#
# and the end-effector point is
#
#       p_6 = T_0_6 [0 0 0 1]^T
#
#
# IMPORTANT:
#
# theta_i contains the supplied joint variable q_i plus the
# constant angular offset required by the particular DH-frame
# assignment.
#
# The DH parameters below reproduce the kinematic transform
# represented by the supplied HEAL XML.
#
# They are an equivalent DH representation; DH frames do not
# have to be identical to the internal MJCF body frames.
# ============================================================


# ============================================================
# 1. JOINT VARIABLES
# ============================================================
#
# These are the same HEAL joint values used in our previous
# MuJoCo experiment.
#
# Units supplied by the user:
#       degrees
#
# They are converted to radians before being used in the
# transformation matrices.

Q_DEG = np.array([
    30.0,    # q1
   -20.0,    # q2
    25.0,    # q3
    20.0,    # q4
   -15.0,    # q5
    20.0     # q6
])

Q = np.deg2rad(Q_DEG)


# ============================================================
# 2. DH PARAMETERS
# ============================================================
#
# Standard DH parameters:
#
#       a_i      = link length
#       alpha_i  = link twist
#       d_i      = link offset
#       theta_i  = joint variable + constant offset
#
#
# The values below are in metres and radians.
#
# The constant theta offsets are part of the chosen DH-frame
# convention. They account for the fact that the convenient
# DH frames are not identical to the raw MJCF body frames.
#
# ------------------------------------------------------------
#       i        a_i          alpha_i
# ------------------------------------------------------------
#       1        0            +pi/2
#       2       -0.3          -pi
#       3        0            -1.57
#       4        0             0.50951
#       5        0            -pi/2
#       6        0             +pi
#
#
# ------------------------------------------------------------
#       i        d_i
# ------------------------------------------------------------
#       1        0.3208
#       2        1.1527768078
#       3        1.1526499530
#       4        0.3773557993
#       5        0.0001000926
#       6       -0.1227
#
#
# Constant theta offsets:
#
#       theta_offset =
#       [-pi, -pi/2, 0.0007963268, 0, 1.57, 0]
#
# Hence:
#
#       theta_i = q_i + theta_offset_i
#
#
# These parameters were obtained by converting the fixed
# transforms and joint-axis directions contained in the XML
# into an equivalent standard-DH chain.
# ============================================================

a = np.array([
    0.0,
   -0.3,
    0.0,
    0.0,
    0.0,
    0.0
])


alpha = np.array([
    np.pi / 2.0,
   -np.pi,
   -1.57,
    0.50951,
   -np.pi / 2.0,
    np.pi
])


d = np.array([
    0.3208,
    1.1527768078,
    1.1526499530,
    0.3773557993,
    0.0001000926,
   -0.1227
])


theta_offset = np.array([
   -np.pi,
   -np.pi / 2.0,
    0.0007963268,
    0.0,
    1.57,
    0.0
])


# ============================================================
# 3. STANDARD DH TRANSFORMATION MATRIX
# ============================================================
#
# For:
#
#       A_i = Rot_z(theta_i)
#             Trans_z(d_i)
#             Trans_x(a_i)
#             Rot_x(alpha_i)
#
# the standard homogeneous transformation is:
#
#
#       | cos(theta)  -sin(theta)cos(alpha)
#       |                         sin(theta)sin(alpha)
#       |
#       | sin(theta)   cos(theta)cos(alpha)
#       |                        -cos(theta)sin(alpha)
#       |
#       | 0            sin(alpha)
#       |
#       | 0            0
#
#
#       | a cos(theta) |
#       | a sin(theta) |
#       | d            |
#       | 1            |
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
# 4. CALCULATE A1 ... A6
# ============================================================
#
# theta_i = q_i + constant DH offset
#
# Then:
#
#       A1 = DH(a1, alpha1, d1, theta1)
#       A2 = DH(a2, alpha2, d2, theta2)
#       ...
#       A6 = DH(a6, alpha6, d6, theta6)
#
# ============================================================

theta = Q + theta_offset


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


# ============================================================
# 5. COMPLETE FORWARD TRANSFORMATION
# ============================================================
#
#       T_0_6 = A1 A2 A3 A4 A5 A6
#
# This gives the position and orientation of frame 6
# relative to the base frame.

T01 = A1

T02 = A1 @ A2

T03 = A1 @ A2 @ A3

T04 = A1 @ A2 @ A3 @ A4

T05 = A1 @ A2 @ A3 @ A4 @ A5

T06 = A1 @ A2 @ A3 @ A4 @ A5 @ A6


# ============================================================
# 6. END-EFFECTOR HOMOGENEOUS VECTOR
# ============================================================
#
# According to the definition used in the lab:
#
# the end effector is at the origin of frame 6.
#
# Therefore its coordinates in frame 6 are:
#
#       [0]
#       [0]
#       [0]
#       [1]
#
# Transforming this point to the base frame gives:
#
#       p_0 = T_0_6 p_6
#
# ============================================================

end_effector_vector = np.array([
    0.0,
    0.0,
    0.0,
    1.0
])


end_effector_position = (
    T06 @ end_effector_vector
)


# ============================================================
# 7. EXTRACT X, Y, Z
# ============================================================
#
# The first three entries of the homogeneous vector are:
#
#       x
#       y
#       z
#
# ============================================================

x = end_effector_position[0]
y = end_effector_position[1]
z = end_effector_position[2]


# ============================================================
# 8. FINAL OUTPUT
# ============================================================
#
# Only the final Cartesian coordinates are printed.
# ============================================================

print(f"x = {x:.6f} m")
print(f"y = {y:.6f} m")
print(f"z = {z:.6f} m")