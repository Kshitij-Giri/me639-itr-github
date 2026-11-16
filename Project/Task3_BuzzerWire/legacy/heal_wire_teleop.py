#!/usr/bin/env python3
"""
heal_wire_teleop.py  —  Keyboard Cartesian teleop + path recorder for the HEAL buzz-wire task
(ROS 2 Humble)

What it does
------------
* Builds a KDL chain base_link -> end-effector from /robot_description (same as starter code).
* Adds a TOOL CENTRE POINT (TCP) offset so the controlled / recorded point is the RING CENTRE,
  not the flange.  Rotations then pivot about the ring, so turning the ring at a corner does
  not drag it into the wire.
* Keyboard teleop in a WIRE FRAME  W = {u: along the wire, v: vertical, n: wire-plane normal}.
* CLOSED-LOOP pose hold: keys move a *target* pose; DLS-IK tracks it.  Axes you are not
  pressing are actively held, so there is no slow drift of the base / wrist (the failure other
  groups saw with open-loop velocity jogging or free-drive).
* JOINT LOCKS: any joint can be locked (its Jacobian column is removed and it is commanded 0).
* TASK-AXIS SELECTION: choose which of the 6 wire-frame axes are controlled. Presets:
      full6   : all 6 joints, all 6 axes held (board placed as in the photo, wire plane facing robot)
      planar3 : joints 2,3,5 only; axes u, v, rn (board turned so the wire plane is radial,
                i.e. contains the robot's base z-axis) -> base, j4 and j6 physically cannot move
      custom  : use parameters lock_joints / task_axes
* RECORDING: TCP pose + flange pose + joint angles to CSV (+ JSON metadata) at record_rate_hz.
  Keyframe / contact markers can be dropped while recording.
* SAFETY: dead-man key timeout, joint-velocity clamp, slew (acceleration) limit, joint-limit
  margin, workspace box, target-lead clamp, /joint_states watchdog, zero velocity on exit.

Run (in a sourced terminal, while ./heal_server and velocity_control.launch.py are running):
    python3 heal_wire_teleop.py --ros-args -p preset:=full6 -p tcp_offset:="[0.0, 0.0, 0.12]"

Tip: shorten keyboard auto-repeat delay so holding a key gives smooth motion:
    xset r rate 180 40
"""

from __future__ import annotations

import csv
import json
import math
import os
import select
import sys
import termios
import threading
import time
import tty
from datetime import datetime

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy,
                       qos_profile_sensor_data)

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import PoseStamped

import PyKDL as kdl
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromUrdfModel

# Optional: starter-code gripper helper (utils/gripper_commands/heal_dynamixel_gripper.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.dirname(SCRIPT_DIR), SCRIPT_DIR):
    if os.path.isdir(os.path.join(_cand, "utils")) and _cand not in sys.path:
        sys.path.insert(0, _cand)
try:
    from utils.gripper_commands.heal_dynamixel_gripper import GripperController  # type: ignore
except Exception:  # pragma: no cover
    GripperController = None


# =====================================================================================
# SO(3) helpers (pure numpy)
# =====================================================================================

def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def so3_exp(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3) + skew(w)
    k = skew(w / th)
    return np.eye(3) + math.sin(th) * k + (1.0 - math.cos(th)) * (k @ k)


def so3_log(R):
    """Rotation matrix -> rotation vector (angle*axis)."""
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    th = math.acos(c)
    if th < 1e-9:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
    if math.pi - th < 1e-6:
        # near 180 deg: axis from the largest diagonal element
        i = int(np.argmax(np.diag(R)))
        a = np.zeros(3)
        a[i] = math.sqrt(max((R[i, i] + 1.0) * 0.5, 0.0))
        for j in range(3):
            if j != i:
                a[j] = R[i, j] / (2.0 * a[i])
        return th * a / np.linalg.norm(a)
    return th / (2.0 * math.sin(th)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def rot_to_quat(R):
    """-> [x, y, z, w]"""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def wire_frame_rotation(yaw):
    """Columns = [u, v, n] expressed in base frame.  u horizontal along wire, v = +z, n = u x v."""
    u = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    v = np.array([0.0, 0.0, 1.0])
    n = np.cross(u, v)
    return np.column_stack([u, v, n])


def tcp_jacobian(J6, R_flange, tcp_offset):
    """Shift a base-frame geometric Jacobian (ref point = flange) to the TCP.
    v_tcp = v_f + w x r  ->  Jv_tcp = Jv - [r]x Jw,  r = R_flange @ tcp_offset."""
    r = R_flange @ np.asarray(tcp_offset, float)
    J = J6.copy()
    J[:3, :] = J6[:3, :] - skew(r) @ J6[3:, :]
    return J


def dls_solve(J, e, lam):
    m = J.shape[0]
    return J.T @ np.linalg.solve(J @ J.T + (lam ** 2) * np.eye(m), e)


AXIS_NAMES = ["u", "v", "n", "ru", "rv", "rn"]

PRESETS = {
    # board as in the photo: wire plane faces the robot -> base MUST rotate for the lateral sweep,
    # so we do not lock it; we *control* it and hold the full pose closed-loop.
    "full6": dict(lock_joints=[], task_axes=["u", "v", "n", "ru", "rv", "rn"]),
    # board rotated 90 deg so the wire plane is radial (contains the base z-axis):
    # a planar 3R problem -> j2, j3, j5 do everything, j1/j4/j6 locked.
    "planar3": dict(lock_joints=[1, 4, 6], task_axes=["u", "v", "rn"]),
}

HELP = r"""
=================== HEAL BUZZ-WIRE TELEOP ===================
 CARTESIAN mode (wire frame: u=along wire, v=up, n=wire normal)
   a / d : -u / +u   (left / right along wire)
   w / s : +v / -v   (up / down)
   q / e : -n / +n   (depth: in / out of wire plane)
   z / x : -rn / +rn (rotate ring IN the wire plane, about TCP)
   i / k : -ru / +ru     j / l : -rv / +rv   (tilts; normally just held)
   1..6  : toggle LOCK of joint i
 JOINT mode
   1..6  : select joint      = / -  : jog selected joint + / -
 COMMON
   m     : toggle CARTESIAN / JOINT mode
   space : STOP + freeze target at current pose
   [ / ] : slower / faster   (speed level 0..4)
   r     : start / stop RECORDING
   p     : mark KEYFRAME (corner)        b : mark CONTACT (buzzer went off)
   L / R : mark wire LEFT / RIGHT end -> sets wire_yaw from the two TCP points
   g / o : gripper close / open           h : this help
   Ctrl-C: quit (sends zero velocity)
=============================================================
"""


# =====================================================================================
# Keyboard (cbreak mode keeps Ctrl-C working)
# =====================================================================================

class Keyboard:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def get(self, timeout=0.02):
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if r:
            return sys.stdin.read(1)
        return None

    def restore(self):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


# =====================================================================================
# Teleop node
# =====================================================================================

class HealWireTeleop(Node):
    LIN_SPEEDS = [0.002, 0.005, 0.010, 0.020, 0.035]   # m/s   (teaching should be slow)
    ANG_SPEEDS = [0.03, 0.06, 0.10, 0.20, 0.30]        # rad/s
    JNT_SPEEDS = [0.02, 0.04, 0.08, 0.15, 0.25]        # rad/s (joint mode)

    def __init__(self):
        super().__init__("heal_wire_teleop")

        p = self.declare_parameter
        p("base_link", "base_link")
        p("tip_link", "end-effector")
        p("joint_names", ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"])
        p("joint_state_topic", "/joint_states")
        p("command_topic", "/velocity_controller/commands")
        p("robot_description_topic", "/robot_description")
        p("preset", "full6")                          # full6 | planar3 | custom
        p("lock_joints", [0])                         # custom only, 1-based; [0] = none
        p("task_axes", ["u", "v", "n", "ru", "rv", "rn"])  # custom only
        p("tcp_offset", [0.0, 0.0, 0.0])              # ring centre in tip_link frame [m]
        p("wire_yaw_deg", 90.0)                       # direction of the wire (u) in base XY
        p("rate_hz", 100.0)
        p("record_rate_hz", 50.0)
        p("record_dir", os.path.join(os.getcwd(), "recordings"))
        p("damping", 0.05)
        p("kp_lin", 2.0)                              # 1/s closed-loop gain on position error
        p("kp_rot", 2.0)                              # 1/s closed-loop gain on rotation error
        p("max_joint_vel", 0.30)                      # rad/s hard clamp
        p("max_joint_acc", 1.0)                       # rad/s^2 slew limit
        p("joint_limit_margin_deg", 5.0)
        p("key_timeout", 0.30)                        # s; motion stops this long after last key
        p("max_target_lead", 0.01)                    # m; target may not run ahead > this
        p("max_target_lead_rot", 0.10)                # rad
        p("workspace_min", [-1.0, -1.0, 0.02])        # base frame box for the TCP target [m]
        p("workspace_max", [1.0, 1.0, 1.2])
        p("js_timeout", 0.2)
        p("gripper_ws", "/home/iitgn-robotics/Debojit_WS/addverb_ws")

        gp = lambda n: self.get_parameter(n).value
        self.base_link, self.tip_link = gp("base_link"), gp("tip_link")
        self.cmd_names = list(gp("joint_names"))
        self.tcp_offset = np.array(gp("tcp_offset"), float)
        self.wire_yaw = math.radians(gp("wire_yaw_deg"))
        self.dt = 1.0 / gp("rate_hz")
        self.lam = gp("damping")
        self.kp_lin, self.kp_rot = gp("kp_lin"), gp("kp_rot")
        self.vmax_j, self.amax_j = gp("max_joint_vel"), gp("max_joint_acc")
        self.jl_margin = math.radians(gp("joint_limit_margin_deg"))
        self.key_timeout = gp("key_timeout")
        self.max_lead, self.max_lead_rot = gp("max_target_lead"), gp("max_target_lead_rot")
        self.ws_min, self.ws_max = np.array(gp("workspace_min")), np.array(gp("workspace_max"))
        self.js_timeout = gp("js_timeout")
        self.record_dir = gp("record_dir")

        preset = gp("preset")
        if preset in PRESETS:
            locks, axes = PRESETS[preset]["lock_joints"], PRESETS[preset]["task_axes"]
        else:
            locks = [j for j in gp("lock_joints") if j > 0]
            axes = list(gp("task_axes"))
        self.preset = preset
        self.locked = set(int(j) for j in locks)       # 1-based joint indices (cmd order)
        self.task_axes = [a for a in AXIS_NAMES if a in axes]

        # ---------------- state ----------------
        self.lock = threading.Lock()
        self.kdl_ready = False
        self.chain_names: list[str] = []
        self.q_lower = self.q_upper = None
        self.js_msg: JointState | None = None
        self.js_time = 0.0
        self.mode = "CART"
        self.speed_idx = 2
        self.cmd_twist_W = np.zeros(6)                  # requested twist in wire frame
        self.last_key_time = 0.0
        self.sel_joint = 1
        self.jog_dir = 0.0
        self.target_p = None
        self.target_R = None
        self.last_cmd = np.zeros(len(self.cmd_names))
        self.tcp_p = self.tcp_R = self.fl_p = self.fl_R = None
        self.q_cmd_order = None
        self.manip = float("nan")
        self.anchor_L = self.anchor_R = None
        self.status_msg = ""

        # recording
        self.recording = False
        self.rec_file = None
        self.rec_writer = None
        self.rec_count = 0
        self.rec_t0 = 0.0
        self.pending_flag = 0                            # 1 keyframe, 2 contact

        # ---------------- ROS I/O ----------------
        self.pub = self.create_publisher(Float64MultiArray, gp("command_topic"), 1)
        self.pose_pub = self.create_publisher(PoseStamped, "/wire_teleop/tcp_pose", 10)
        self.target_pub = self.create_publisher(PoseStamped, "/wire_teleop/target_pose", 10)
        self.create_subscription(JointState, gp("joint_state_topic"), self._js_cb,
                                 qos_profile_sensor_data)
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                         durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self._desc_sub = self.create_subscription(String, gp("robot_description_topic"),
                                                  self._desc_cb, qos)
        self.create_timer(self.dt, self._control_tick)
        self.create_timer(1.0 / gp("record_rate_hz"), self._record_tick)
        self.create_timer(0.25, self._status_tick)

        self.gripper = None
        if GripperController is not None:
            try:
                self.gripper = GripperController(workspace_path=gp("gripper_ws"), verbose=False)
            except Exception as e:
                self.get_logger().warn(f"Gripper helper unavailable: {e}")

        self.get_logger().info(
            f"preset={preset} locked={sorted(self.locked)} task_axes={self.task_axes} "
            f"tcp_offset={self.tcp_offset.tolist()} wire_yaw={math.degrees(self.wire_yaw):.1f}deg")

    # ------------------------------------------------------------------ callbacks
    def _js_cb(self, msg: JointState):
        self.js_msg = msg
        self.js_time = time.monotonic()

    def _desc_cb(self, msg: String):
        if self.kdl_ready:
            return
        try:
            robot = URDF.from_xml_string(msg.data)
            ok, tree = treeFromUrdfModel(robot)
            if not ok:
                raise RuntimeError("treeFromUrdfModel failed")
            chain = tree.getChain(self.base_link, self.tip_link)
            n = chain.getNrOfJoints()
            if n == 0:
                raise RuntimeError(f"0 joints between {self.base_link} and {self.tip_link}")
            names = []
            for i in range(chain.getNrOfSegments()):
                j = chain.getSegment(i).getJoint()
                if j.getTypeName() not in ("None", "Fixed"):
                    names.append(j.getName())
            lo, hi = [], []
            for nm in names:
                lim = robot.joint_map[nm].limit if nm in robot.joint_map else None
                lo.append(lim.lower if lim is not None and lim.lower is not None else -2 * math.pi)
                hi.append(lim.upper if lim is not None and lim.upper is not None else 2 * math.pi)
            self.chain, self.n = chain, n
            self.chain_names = names
            self.q_lower, self.q_upper = np.array(lo), np.array(hi)
            self.fk = kdl.ChainFkSolverPos_recursive(chain)
            self.jac = kdl.ChainJntToJacSolver(chain)
            self.kdl_ready = True
            self.destroy_subscription(self._desc_sub)
            self.get_logger().info(f"KDL ready: {names}\n limits lo={np.round(lo, 2)} hi={np.round(hi, 2)}")
        except Exception as e:
            self.get_logger().error(f"robot_description parse failed: {e}")

    # ------------------------------------------------------------------ kinematics
    def _q_chain(self):
        msg = self.js_msg
        idx = {n: i for i, n in enumerate(msg.name)}
        q = np.zeros(self.n)
        for i, nm in enumerate(self.chain_names):
            if nm not in idx:
                return None
            q[i] = msg.position[idx[nm]]
        return q

    def _fk_jac(self, q):
        qk = kdl.JntArray(self.n)
        for i in range(self.n):
            qk[i] = float(q[i])
        F = kdl.Frame()
        self.fk.JntToCart(qk, F)
        Jk = kdl.Jacobian(self.n)
        self.jac.JntToJac(qk, Jk)
        J = np.array([[Jk[r, c] for c in range(self.n)] for r in range(6)])
        R = np.array([[F.M[r, c] for c in range(3)] for r in range(3)])
        p = np.array([F.p[0], F.p[1], F.p[2]])
        return p, R, J

    def _chain_to_cmd(self, v_chain):
        m = {n: i for i, n in enumerate(self.chain_names)}
        return np.array([v_chain[m[n]] if n in m else 0.0 for n in self.cmd_names])

    def _cmd_to_chain_idx(self, cmd_idx_1based):
        name = self.cmd_names[cmd_idx_1based - 1]
        return self.chain_names.index(name) if name in self.chain_names else None

    def _print_axes_diagnostic(self, J):
        """Print each joint axis vs. wire normal: tells you which joints are 'in-plane' joints."""
        n_hat = wire_frame_rotation(self.wire_yaw)[:, 2]
        lines = []
        for i, nm in enumerate(self.chain_names):
            w = J[3:, i]
            c = abs(float(np.dot(w, n_hat))) / max(np.linalg.norm(w), 1e-9)
            lines.append(f"  {nm}: axis={np.round(w, 2)}  |cos(axis, wire_normal)|={c:.2f}"
                         + ("  <- in-plane joint" if c > 0.95 else ""))
        self.get_logger().info("Joint axes vs wire normal (1.00 = joint rotates IN the wire plane):\n"
                               + "\n".join(lines))
        if self.preset == "planar3":
            bad = [j for j in range(1, 7) if j not in self.locked and
                   abs(np.dot(J[3:, self._cmd_to_chain_idx(j)], n_hat)) < 0.95]
            if bad:
                self.get_logger().warn(
                    f"planar3: active joints {bad} are NOT parallel to the wire normal. "
                    "Rotate j4/j6 (JOINT mode) until j2,j3,j5 axes read ~1.00, or set wire_yaw_deg.")

    # ------------------------------------------------------------------ target handling
    def _reset_target(self):
        if self.tcp_p is not None:
            self.target_p = self.tcp_p.copy()
            self.target_R = self.tcp_R.copy()

    # ------------------------------------------------------------------ main loop
    def _publish(self, dq):
        msg = Float64MultiArray()
        msg.data = [float(x) for x in dq]
        self.pub.publish(msg)

    def _slew(self, dq_des):
        step = self.amax_j * self.dt
        dq = self.last_cmd + np.clip(dq_des - self.last_cmd, -step, step)
        self.last_cmd = dq
        return dq

    def _control_tick(self):
        now = time.monotonic()
        if not self.kdl_ready or self.js_msg is None:
            return
        if now - self.js_time > self.js_timeout:
            self.last_cmd[:] = 0.0
            self._publish(self.last_cmd)
            self.status_msg = "JOINT_STATES STALE -> zero velocity"
            self.target_p = None
            return
        q = self._q_chain()
        if q is None:
            self.status_msg = "joint names missing in /joint_states"
            return

        p_f, R_f, J_f = self._fk_jac(q)
        p_tcp = p_f + R_f @ self.tcp_offset
        J = tcp_jacobian(J_f, R_f, self.tcp_offset)
        with self.lock:
            self.fl_p, self.fl_R, self.tcp_p, self.tcp_R = p_f, R_f, p_tcp, R_f.copy()
            self.q_cmd_order = self._chain_to_cmd(q)
            first = self.target_p is None
            if first:
                self._reset_target()
                self._print_axes_diagnostic(J)
            key_alive = (now - self.last_key_time) < self.key_timeout
            mode = self.mode
            twist_W = self.cmd_twist_W.copy() if key_alive else np.zeros(6)
            jog = self.jog_dir if key_alive else 0.0
            sel = self.sel_joint

        dq_chain = np.zeros(self.n)
        active = [self._cmd_to_chain_idx(j) for j in range(1, len(self.cmd_names) + 1)
                  if j not in self.locked]
        active = [a for a in active if a is not None]

        if mode == "JOINT":
            ci = self._cmd_to_chain_idx(sel)
            if ci is not None and sel not in self.locked:
                dq_chain[ci] = jog * self.JNT_SPEEDS[self.speed_idx]
            with self.lock:
                self._reset_target()               # keep target glued to the arm in joint mode
        else:
            W = wire_frame_rotation(self.wire_yaw)
            # ---- integrate target from key twist (wire frame -> base frame) ----
            v_ff = W @ twist_W[:3]
            w_ff = W @ twist_W[3:]
            with self.lock:
                self.target_p = np.clip(self.target_p + v_ff * self.dt, self.ws_min, self.ws_max)
                self.target_R = so3_exp(w_ff * self.dt) @ self.target_R
                # lead clamp: target may not run away from the real TCP
                d = self.target_p - p_tcp
                if np.linalg.norm(d) > self.max_lead:
                    self.target_p = p_tcp + d * (self.max_lead / np.linalg.norm(d))
                e_rot = so3_log(self.target_R @ R_f.T)
                if np.linalg.norm(e_rot) > self.max_lead_rot:
                    e_rot *= self.max_lead_rot / np.linalg.norm(e_rot)
                    self.target_R = so3_exp(e_rot) @ R_f
                e_pos = self.target_p - p_tcp

            # ---- desired twist = feed-forward + P on pose error (base frame) ----
            v = v_ff + self.kp_lin * e_pos
            w = w_ff + self.kp_rot * e_rot
            # express in wire frame and pick controlled axes
            xi_W = np.concatenate([W.T @ v, W.T @ w])
            J_W = np.vstack([W.T @ J[:3, :], W.T @ J[3:, :]])
            rows = [AXIS_NAMES.index(a) for a in self.task_axes]
            if len(rows) > len(active):
                self.status_msg = (f"INFEASIBLE: {len(rows)} task axes > {len(active)} free joints"
                                   " -> unlock joints or drop axes")
            elif rows and active:
                Js = J_W[np.ix_(rows, active)]
                es = xi_W[rows]
                # tiny deadband so the hold does not chatter
                if np.linalg.norm(e_pos) < 3e-4 and np.linalg.norm(e_rot) < 2e-3 and not twist_W.any():
                    es = np.zeros_like(es)
                sv = np.linalg.svd(Js, compute_uv=False)
                self.manip = float(sv[-1]) if sv.size else float("nan")
                dq_chain[active] = dls_solve(Js, es, self.lam)

        # ---- joint-limit guard ----
        near_lo = (q - self.q_lower) < self.jl_margin
        near_hi = (self.q_upper - q) < self.jl_margin
        dq_chain[(near_lo & (dq_chain < 0)) | (near_hi & (dq_chain > 0))] = 0.0
        # ---- uniform scaling to max joint velocity (keeps direction) ----
        mx = float(np.max(np.abs(dq_chain))) if dq_chain.size else 0.0
        if mx > self.vmax_j:
            dq_chain *= self.vmax_j / mx
        dq_cmd = self._chain_to_cmd(dq_chain)
        for j in self.locked:                    # locked joints: hard zero
            dq_cmd[j - 1] = 0.0
        dq_cmd = self._slew(dq_cmd)
        for j in self.locked:
            dq_cmd[j - 1] = 0.0
            self.last_cmd[j - 1] = 0.0
        self._publish(dq_cmd)

        # visualisation topics (RViz)
        self._pub_pose(self.pose_pub, p_tcp, R_f)
        if self.target_p is not None:
            self._pub_pose(self.target_pub, self.target_p, self.target_R)

    def _pub_pose(self, pub, p, R):
        m = PoseStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.base_link
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, p)
        qx, qy, qz, qw = rot_to_quat(R)
        m.pose.orientation.x, m.pose.orientation.y = float(qx), float(qy)
        m.pose.orientation.z, m.pose.orientation.w = float(qz), float(qw)
        pub.publish(m)

    # ------------------------------------------------------------------ recording
    def _start_recording(self):
        os.makedirs(self.record_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.record_dir, f"wire_path_{stamp}.csv")
        self.rec_file = open(path, "w", newline="")
        self.rec_writer = csv.writer(self.rec_file)
        hdr = ["t", "flag"] + [f"q_{n}" for n in self.cmd_names] + \
              ["tcp_x", "tcp_y", "tcp_z", "tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw",
               "fl_x", "fl_y", "fl_z", "fl_qx", "fl_qy", "fl_qz", "fl_qw"]
        self.rec_writer.writerow(hdr)
        meta = dict(created=stamp, base_link=self.base_link, tip_link=self.tip_link,
                    joint_names=self.cmd_names, tcp_offset_in_tip=self.tcp_offset.tolist(),
                    wire_yaw_deg=math.degrees(self.wire_yaw), preset=self.preset,
                    locked_joints=sorted(self.locked), task_axes=self.task_axes,
                    flags={"0": "sample", "1": "keyframe", "2": "contact"},
                    units="m, rad; quaternions x,y,z,w; poses in base_link")
        with open(path.replace(".csv", "_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        self.rec_count = 0
        self.rec_t0 = time.monotonic()
        self.recording = True
        self.rec_path = path
        self.get_logger().info(f"RECORDING -> {path}")

    def _stop_recording(self):
        self.recording = False
        if self.rec_file:
            self.rec_file.flush()
            self.rec_file.close()
            self.get_logger().info(f"Saved {self.rec_count} samples -> {self.rec_path}")
        self.rec_file = self.rec_writer = None

    def _record_tick(self):
        if not self.recording or self.tcp_p is None or self.rec_writer is None:
            return
        with self.lock:
            q = self.q_cmd_order.copy()
            p, R, pf, Rf = self.tcp_p.copy(), self.tcp_R.copy(), self.fl_p.copy(), self.fl_R.copy()
            flag, self.pending_flag = self.pending_flag, 0
        row = [f"{time.monotonic() - self.rec_t0:.4f}", flag] + [f"{x:.6f}" for x in q] + \
              [f"{x:.6f}" for x in p] + [f"{x:.7f}" for x in rot_to_quat(R)] + \
              [f"{x:.6f}" for x in pf] + [f"{x:.7f}" for x in rot_to_quat(Rf)]
        self.rec_writer.writerow(row)
        self.rec_count += 1
        if self.rec_count % 100 == 0:
            self.rec_file.flush()

    # ------------------------------------------------------------------ status line
    def _status_tick(self):
        if self.tcp_p is None:
            sys.stdout.write("\r waiting for /robot_description and /joint_states ...      ")
            sys.stdout.flush()
            return
        rec = f"REC {self.rec_count:5d}" if self.recording else "rec off  "
        lk = "".join(str(j) for j in sorted(self.locked)) or "-"
        sel = f" sel=j{self.sel_joint}" if self.mode == "JOINT" else ""
        line = (f"\r[{self.mode}{sel}] spd{self.speed_idx} {rec} locks:{lk} "
                f"TCP=({self.tcp_p[0]:+.3f},{self.tcp_p[1]:+.3f},{self.tcp_p[2]:+.3f}) "
                f"σmin={self.manip:.3f} {self.status_msg[:50]:50s}")
        sys.stdout.write(line)
        sys.stdout.flush()

    # ------------------------------------------------------------------ keyboard handling
    def handle_key(self, k):
        now = time.monotonic()
        lin, ang = self.LIN_SPEEDS[self.speed_idx], self.ANG_SPEEDS[self.speed_idx]
        cart_map = {
            "a": (0, -lin), "d": (0, +lin), "w": (1, +lin), "s": (1, -lin),
            "q": (2, -lin), "e": (2, +lin),
            "i": (3, -ang), "k": (3, +ang), "j": (4, -ang), "l": (4, +ang),
            "z": (5, -ang), "x": (5, +ang),
        }
        with self.lock:
            if self.mode == "CART" and k in cart_map:
                ax, val = cart_map[k]
                if AXIS_NAMES[ax] not in self.task_axes:
                    self.status_msg = f"axis {AXIS_NAMES[ax]} not controlled in this preset"
                    return
                self.cmd_twist_W[:] = 0.0
                self.cmd_twist_W[ax] = val
                self.last_key_time = now
                self.status_msg = ""
            elif self.mode == "JOINT" and k in ("=", "+", "-"):
                self.jog_dir = 1.0 if k in ("=", "+") else -1.0
                self.last_key_time = now
            elif k in "123456":
                j = int(k)
                if self.mode == "JOINT":
                    self.sel_joint = j
                else:
                    self.locked.symmetric_difference_update({j})
                    self.preset = "custom"
                    self._reset_target()
                    self.status_msg = f"locks now {sorted(self.locked)}"
            elif k == " ":
                self.cmd_twist_W[:] = 0.0
                self.jog_dir = 0.0
                self.last_key_time = 0.0
                self._reset_target()
                self.status_msg = "STOP"
            elif k == "m":
                self.mode = "JOINT" if self.mode == "CART" else "CART"
                self.cmd_twist_W[:] = 0.0
                self.jog_dir = 0.0
                self._reset_target()
            elif k == "[":
                self.speed_idx = max(0, self.speed_idx - 1)
            elif k == "]":
                self.speed_idx = min(len(self.LIN_SPEEDS) - 1, self.speed_idx + 1)
            elif k == "p":
                self.pending_flag = 1
                self.status_msg = "keyframe marked"
            elif k == "b":
                self.pending_flag = 2
                self.status_msg = "CONTACT marked"
            elif k in ("L", "R"):
                if k == "L":
                    self.anchor_L = self.tcp_p.copy()
                else:
                    self.anchor_R = self.tcp_p.copy()
                if self.anchor_L is not None and self.anchor_R is not None:
                    d = self.anchor_R - self.anchor_L
                    self.wire_yaw = math.atan2(d[1], d[0])
                    self._reset_target()
                    self.status_msg = f"wire_yaw set to {math.degrees(self.wire_yaw):.1f} deg"
                else:
                    self.status_msg = f"anchor {k} stored"
            elif k == "h":
                sys.stdout.write("\n" + HELP + "\n")
        # outside the lock (these may block / do I/O)
        if k == "r":
            if self.recording:
                self._stop_recording()
            else:
                self._start_recording()
        elif k in ("g", "o"):
            if self.gripper is None:
                self.status_msg = "gripper helper not available"
            else:
                fn = self.gripper.close_gripper if k == "g" else self.gripper.open_gripper
                threading.Thread(target=fn, daemon=True).start()

    def shutdown(self):
        for _ in range(5):
            try:
                self._publish(np.zeros(len(self.cmd_names)))
            except Exception:
                pass
            time.sleep(0.01)
        if self.recording:
            self._stop_recording()


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = HealWireTeleop()
    print(HELP)
    kb = Keyboard()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        while rclpy.ok():
            k = kb.get(0.02)
            if k is not None:
                node.handle_key(k)
    except KeyboardInterrupt:
        pass
    finally:
        kb.restore()
        print("\nStopping: sending zero velocity.")
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
