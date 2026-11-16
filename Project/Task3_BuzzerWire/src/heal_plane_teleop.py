#!/usr/bin/env python3
"""
heal_plane_teleop.py — HEAL "plane teleop" + path recorder (ROS 2 Humble)   v5.9 (2026-11-14, G = exact +X start pose)

Controls (PLANE mode)
---------------------
  Y        : ALIGN TO PRESET (default: gripper points +Y, fingers along X - the verified pose), then lock
             plane control with the plane facing the way the gripper points. Tool tip stays put.
             Preset set by params align_point / align_edge / align_to.
  T        : choose WHICH gripper axis is the plane normal (cycles tool x -> y -> z) and re-aligns.
             The plane is always the vertical plane through the tool tip, facing the way that axis points.
  K        : LOCK the current gripper orientation exactly as it is (no snapping) and start plane
             control in the current plane. Use this after orienting the gripper by hand in JOINT mode.
  H (or h) : square the gripper up to the vertical plane in front of it.
             The move is SIMULATED first; it is refused if any joint would get near a limit
             or swing more than max_align_travel_deg. The safest nearby orientation is chosen.
  w / s    : up / down            (inside the plane)
  z / x    : left / right         (inside the plane)
  a / d    : plane offset         (whole plane toward / away; stays parallel)
  q / e    : final-joint yaw  (joint 6 is LOCKED and never moves unless q/e is held)
  g / o    : gripper close / open
  space    : stop
  [ / ]    : slower / faster (8 levels: 0.2, 0.5, 1, 2, 3.5, 5, 7.5, 10 cm/s)
  { / }    : jump to slowest / fastest
  r        : start / stop recording     p : keyframe     b : contact (buzzer)
  L / R    : mark two points along the wire (left end, right end) -> sets the plane direction.
             Then press K or H. (Works in any mode; move the tip with JOINT jog.)
  v        : print current gripper orientation relative to the plane
  m        : JOINT mode (1-6 select joint, = / - jog);  m again -> PLANE, then H
  G        : GO to the start pose (slow joint-space move), then EXACT alignment (point +X, fingers along Y)
             and plane control - ready to trace
  S        : SAVE the current joint angles as the start pose (kept in ~/.heal_start_pose_v2.json)
  0        : RECOVER  - slow joint-space move of any joint near/over its limit back into the safe band
  9        : HOME     - slow joint-space move to the home pose (default all zeros). Path NOT collision checked!
  ?        : help          Ctrl-C : quit (zero velocity)

Safety layer
------------
  * predictive joint limits: if any joint would enter (limit - margin) within lookahead_s, the WHOLE
    arm stops (no distorted motion), the goal is re-based to where the arm is, and a warning shows.
  * hard stops (default): ±170° all joints, joint4 ±165° (robot rejected joint4 = -178°).
    The robot's real position limits are inside heal_server; set them with soft_lower_deg/soft_upper_deg.
    URDF limits are only shown as an advisory note unless use_urdf_limits:=true.
  * joint velocity cap 0.30 rad/s on every joint (driver rejects > 0.5), acceleration cap 0.6 rad/s².
  * following watchdog: if we command motion but the joints do not move for 1.5 s (robot in error),
    commands go to zero and plane mode disengages.
"""
from __future__ import annotations

import csv
import json
import math
import os
import select
import subprocess
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from plane_core import PlaneCore, tcp_jacobian, rot_to_quat, quintic, preset_orientation, simulate_ik  # noqa: E402

try:
    from addverb_cobot_msgs.srv import Gripper as GripperSrv  # type: ignore
except Exception:
    GripperSrv = None

HELP = __doc__.split("Safety layer")[0]


class Keyboard:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def get(self, timeout=0.02):
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        return sys.stdin.read(1) if r else None

    def restore(self):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


class HealPlaneTeleop(Node):
    # 8 speed levels, from very fine to fast. Every joint is still hard-capped at max_joint_vel (0.30 rad/s,
    # driver rejects > 0.5): if a fast level would need more, the WHOLE motion is scaled down (direction kept).
    LIN_SPEEDS = [0.002, 0.005, 0.010, 0.020, 0.035, 0.050, 0.075, 0.100]   # m/s
    YAW_SPEEDS = [0.02, 0.04, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]           # rad/s
    JNT_SPEEDS = [0.02, 0.04, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]           # rad/s

    def __init__(self):
        super().__init__("heal_plane_teleop")
        p = self.declare_parameter
        p("base_link", "base_link")
        p("tip_link", "rq_tool_tip")
        p("joint_names", ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"])
        p("joint_state_topic", "/joint_states")
        p("command_topic", "/velocity_controller/commands")
        p("robot_description_topic", "/robot_description")
        p("tcp_offset", [0.0, 0.0, 0.0])
        p("wire_yaw_deg", 180.0)                  # only if plane_from_tool:=false. 180 -> plane faces robot's +Y (its front at home)
        p("align_point", "+X")                    # Y key / after G: approach direction  (+X -X +Y -Y +Z -Z)
        p("align_edge", "x")                      # gripper edge (x = finger direction, y = flat-face normal)
        p("align_to", "Y")                        # base axis for that edge
        p("plane_from_tool", True)                # H/K take the plane from the gripper itself
        # rq_tool_tip axes: x = finger open/close direction, y = across the gripper's flat face (thickness),
        # z = approach (out through the fingertips). normal = y  ->  gripper's flat face lies IN the plane.
        p("plane_normal_axis", "y")               # tool axis used as plane normal: x | y | z (T cycles). (verified with K)
        p("invert_lr", False)
        p("invert_offset", False)
        p("rate_hz", 100.0)
        p("record_rate_hz", 50.0)
        p("record_dir", os.path.join(os.getcwd(), "recordings"))
        p("damping", 0.05)
        p("kp_lin", 2.0)
        p("kp_rot", 2.0)
        # ---- safety ----
        p("max_joint_vel", 0.30)                  # rad/s hard cap on EVERY joint (driver REJECTS |cmd| > 0.5)
        p("max_joint_acc", 0.6)                   # rad/s^2
        # The enforced position limits live inside heal_server on the robot (safety_type 1) and are NOT
        # in the ROS source. default_joint_limits in robot_config_info.h is unused dead code, and URDF
        # <limit> tags are not enforced. Only hard evidence: joint4 = -178 deg was rejected.
        # So the hard stops below are a wide band; override with the robot's real values when known.
        p("soft_lower_deg", [-170.0, -170.0, -170.0, -165.0, -170.0, -170.0])
        p("soft_upper_deg", [170.0, 170.0, 170.0, 165.0, 170.0, 170.0])
        p("use_urdf_limits", False)
        # Per-joint direction between the robot and the loaded URDF (only change after an RViz check).
        p("joint_sign", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])               # True -> also stop at URDF limits (j2 [-43,117], j3 ±57)
        p("limit_margin_deg", 5.0)                # stop this far before the soft limit
        p("lookahead_s", 0.5)                     # predictive limit horizon
        p("sing_stop", 0.012)                     # refuse PLANE motion that drives sigma_min below this
                                                  # (Heal wrist singularity: joint5 near +/-90 deg)
        p("safe_band_deg", 15.0)                  # RECOVER brings joints this far inside soft limits
        p("max_align_travel_deg", 60.0)           # H refused if any joint would swing more than this
        p("traj_max_vel", 0.12)                   # rad/s peak for RECOVER / HOME
        p("home_deg", [0.0] * 6)
        # G key target: exact solution for "point +X, fingers along Y" near the team's working pose
        # (sigma 0.133, far from the joint5=+/-90 singularity). Overridden by a pose saved with S.
        p("start_pose_deg", [0.01, 37.93, 27.50, -90.08, 0.09, -0.70])
        p("start_pose_file", os.path.expanduser("~/.heal_start_pose_v2.json"))
        p("align_lin_speed", 0.01)
        p("align_ang_speed", 0.08)
        p("key_timeout", 0.30)
        p("max_lead", 0.03)
        p("workspace_min", [-1.0, -1.0, 0.02])
        p("workspace_max", [1.0, 1.0, 1.2])
        p("js_timeout", 0.2)
        p("gripper_service", "/gripper_controller/command")
        p("gripper_open", 1.0)
        p("gripper_close", 0.0)
        p("gripper_force", 100.0)

        g = lambda n: self.get_parameter(n).value
        self.g = g
        self.base_link, self.tip_link = g("base_link"), g("tip_link")
        self.cmd_names = list(g("joint_names"))
        self.tcp_offset = np.array(g("tcp_offset"), float)
        self.sgn_lr = -1.0 if g("invert_lr") else 1.0
        self.sgn_off = -1.0 if g("invert_offset") else 1.0
        self.dt = 1.0 / g("rate_hz")
        self.vmax_j, self.amax_j = g("max_joint_vel"), g("max_joint_acc")
        self.margin = math.radians(g("limit_margin_deg"))
        self.lookahead = g("lookahead_s")
        self.key_timeout = g("key_timeout")
        self.js_timeout = g("js_timeout")
        self.record_dir = g("record_dir")

        self.core = PlaneCore(
            wire_yaw=math.radians(g("wire_yaw_deg")), kp_lin=g("kp_lin"), kp_rot=g("kp_rot"),
            damping=g("damping"), ref_max_lin=1.5 * max(self.LIN_SPEEDS),
            ref_max_ang=1.5 * max(self.YAW_SPEEDS), align_lin=g("align_lin_speed"),
            align_ang=g("align_ang_speed"), max_lead=g("max_lead"),
            ws_min=g("workspace_min"), ws_max=g("workspace_max"))

        self.lock = threading.Lock()
        self.kdl_ready = False
        self.js_msg, self.js_time = None, 0.0
        self.js_pos, self.js_vel = {}, {}
        self.partial_msgs = 0
        self.mode = "PLANE"            # PLANE | JOINT | TRAJ
        self.speed_idx = 3
        self.rates = np.zeros(4)
        self.last_key_time = 0.0
        self.sel_joint, self.jog_dir = 6, 0.0
        self.last_cmd = np.zeros(len(self.cmd_names))
        self.tcp_p = self.tcp_R = self.q_cmd = self.q_chain = self.last_axis = None
        self.request_align = False
        self.request_freeze = False
        self.request_traj = None       # "recover" | "home"
        self.traj = None               # (q0, q1, T, t0)
        self.anchor_L = self.anchor_R = None
        self.plane_from_tool = bool(self.g("plane_from_tool"))
        self.normal_axis = str(self.g("plane_normal_axis")).lower()
        self.status_msg = ""
        self.warn_msg = ""
        self.nofollow_t = 0.0
        self.nofollow_q0 = None
        self.recording, self.rec_writer, self.rec_file = False, None, None
        self.rec_count, self.rec_t0, self.pending_flag = 0, 0.0, 0

        self.pub = self.create_publisher(Float64MultiArray, g("command_topic"), 1)
        self.pose_pub = self.create_publisher(PoseStamped, "/wire_teleop/tcp_pose", 10)
        self.ref_pub = self.create_publisher(PoseStamped, "/wire_teleop/target_pose", 10)
        self.create_subscription(JointState, g("joint_state_topic"), self._js_cb,
                                 qos_profile_sensor_data)
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                         durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self._desc_sub = self.create_subscription(String, g("robot_description_topic"),
                                                  self._desc_cb, qos)
        self.create_timer(self.dt, self._tick)
        self.create_timer(1.0 / g("record_rate_hz"), self._record_tick)
        self.create_timer(0.25, self._status_tick)

        self.grip_srv = g("gripper_service")
        self.grip_open, self.grip_close = g("gripper_open"), g("gripper_close")
        self.grip_force = g("gripper_force")
        self.grip_cli = self.create_client(GripperSrv, self.grip_srv) if GripperSrv else None

    # ================================================================ ROS callbacks
    def _js_cb(self, msg):
        """Accept ONLY complete arm messages (all joint1..joint6). Partial messages from other
        publishers on /joint_states (gripper, other robots on the network) are ignored, never merged."""
        if not all(n in msg.name for n in self.cmd_names) or len(msg.position) < len(msg.name):
            self.partial_msgs += 1
            return
        vel = msg.velocity
        pos, vv = {}, {}
        for i, nm in enumerate(msg.name):
            pos[nm] = float(msg.position[i])
            vv[nm] = float(vel[i]) if i < len(vel) else 0.0
        self.js_pos, self.js_vel = pos, vv
        self.js_msg, self.js_time = True, time.monotonic()

    def _desc_cb(self, msg):
        if self.kdl_ready:
            return
        try:
            robot = URDF.from_xml_string(msg.data)
            ok, tree = treeFromUrdfModel(robot)
            chain = tree.getChain(self.base_link, self.tip_link)
            if not ok or chain.getNrOfJoints() == 0:
                raise RuntimeError(f"bad chain {self.base_link}->{self.tip_link}")
            names = [chain.getSegment(i).getJoint().getName() for i in range(chain.getNrOfSegments())
                     if chain.getSegment(i).getJoint().getTypeName() not in ("None", "Fixed")]
            slo = dict(zip(self.cmd_names, np.radians(self.g("soft_lower_deg"))))
            shi = dict(zip(self.cmd_names, np.radians(self.g("soft_upper_deg"))))
            lo, hi, ulo, uhi = [], [], [], []
            use_urdf = bool(self.g("use_urdf_limits"))
            for nm in names:
                lim = robot.joint_map[nm].limit if nm in robot.joint_map else None
                l = lim.lower if lim is not None and lim.lower is not None else -math.pi
                h = lim.upper if lim is not None and lim.upper is not None else math.pi
                ulo.append(l)
                uhi.append(h)
                lo.append(max(l, slo[nm]) if use_urdf else slo[nm])
                hi.append(min(h, shi[nm]) if use_urdf else shi[nm])
            self.urdf_lo, self.urdf_hi = np.array(ulo), np.array(uhi)
            self.chain, self.n, self.chain_names = chain, chain.getNrOfJoints(), names
            self.q_lo, self.q_hi = np.array(lo), np.array(hi)
            band = math.radians(self.g("safe_band_deg"))
            self.safe_lo, self.safe_hi = self.q_lo + band, self.q_hi - band
            hd = dict(zip(self.cmd_names, np.radians(self.g("home_deg"))))
            self.q_home = np.clip(np.array([hd.get(nm, 0.0) for nm in names]), self.safe_lo, self.safe_hi)
            sp = list(self.g("start_pose_deg"))
            try:
                with open(self.g("start_pose_file")) as f:
                    sp = json.load(f)["joints_deg"]
                    self.get_logger().info(f"start pose loaded from file: {sp}")
            except Exception:
                pass
            spd = dict(zip(self.cmd_names, np.radians(sp)))
            self.q_start = np.array([spd[nm] for nm in names])
            self.fk = kdl.ChainFkSolverPos_recursive(chain)
            self.jac = kdl.ChainJntToJacSolver(chain)
            self.kdl_ready = True
            self.destroy_subscription(self._desc_sub)
            self.get_logger().info(
                "KDL ready. soft limits [deg]: " + ", ".join(
                    f"{nm}[{math.degrees(a):.0f},{math.degrees(b):.0f}]" for nm, a, b in zip(names, lo, hi)))
        except Exception as e:
            self.get_logger().error(f"robot_description: {e}")

    # ================================================================ kinematics
    def _q_from_js(self):
        pos, vel = dict(self.js_pos), dict(self.js_vel)
        missing = [n for n in self.chain_names if n not in pos]
        if missing:
            self.missing_names = missing
            return None, None
        sg = dict(zip(self.cmd_names, self.g("joint_sign")))
        q = np.array([sg[n] * pos[n] for n in self.chain_names])
        qd = np.array([sg[n] * vel.get(n, 0.0) for n in self.chain_names])
        return q, qd

    def _fk_jac_tip(self, q):
        qk = kdl.JntArray(self.n)
        for i in range(self.n):
            qk[i] = float(q[i])
        F = kdl.Frame()
        self.fk.JntToCart(qk, F)
        Jk = kdl.Jacobian(self.n)
        self.jac.JntToJac(qk, Jk)
        J = np.array([[Jk[r, c] for c in range(self.n)] for r in range(6)])
        R = np.array([[F.M[r, c] for c in range(3)] for r in range(3)])
        return np.array([F.p[0], F.p[1], F.p[2]]), R, J

    def _fkj_tcp(self, q):
        p, R, J = self._fk_jac_tip(q)
        return p + R @ self.tcp_offset, R, tcp_jacobian(J, R, self.tcp_offset)

    def _to_cmd(self, v_chain):
        m = {n: i for i, n in enumerate(self.chain_names)}
        sg = dict(zip(self.cmd_names, self.g("joint_sign")))
        return np.array([sg[n] * v_chain[m[n]] if n in m else 0.0 for n in self.cmd_names])

    # ================================================================ safety helpers
    def _limit_check(self, q, dq):
        """Return list of joint indices that would enter the stop zone within lookahead."""
        q_pred = q + dq * self.lookahead
        lo_hit = (q_pred < self.q_lo + self.margin) & (dq < 0)
        hi_hit = (q_pred > self.q_hi - self.margin) & (dq > 0)
        return list(np.where(lo_hit | hi_hit)[0])

    def _outside_safe(self, q):
        return list(np.where((q < self.q_lo) | (q > self.q_hi))[0])

    def _plane_from_tool(self, p, R):
        """Vertical plane through the tool tip whose normal is the horizontal projection of the chosen
        tool axis, pointing away from the robot base. Falls back to another axis if that one is vertical."""
        order = {"x": [0, 1, 2], "y": [1, 0, 2], "z": [2, 0, 1]}[self.normal_axis]
        for i in order:
            a = R[:, i].copy()
            a[2] = 0.0
            if np.linalg.norm(a) > 0.3:                 # axis not (nearly) vertical
                n = a / np.linalg.norm(a)
                ph = np.array([p[0], p[1], 0.0])
                if np.dot(n, ph) < 0:                    # face away from the robot
                    n = -n
                # plane_frame: n = (sin y, -cos y, 0)  ->  y = atan2(n_x, -n_y)
                self.core.wire_yaw = math.atan2(n[0], -n[1])
                used = "xyz"[i]
                return f"plane normal = tool {used} (yaw {math.degrees(self.core.wire_yaw):+.0f}°)"
        return "plane: no usable tool axis"

    def _start_traj(self, q, kind):
        if kind == "recover":
            q1 = np.clip(q, self.safe_lo, self.safe_hi)
            if np.max(np.abs(q1 - q)) < math.radians(0.5):
                return "RECOVER: all joints already inside the safe band"
        elif kind == "start":
            q1 = self.q_start.copy()
            bad = [self.chain_names[i] for i in range(self.n) if q1[i] < self.q_lo[i] or q1[i] > self.q_hi[i]]
            if bad:
                return "GO refused: saved pose outside soft limits on " + ",".join(bad)
            if np.max(np.abs(q1 - q)) < math.radians(0.3):
                self.lock_after_traj = True
                self.traj = (q.copy(), q.copy(), 0.01, time.monotonic())
                self.core.disengage()
                self.mode = "TRAJ"
                return "already at start pose - locking"
        else:
            q1 = self.q_home.copy()
        d = float(np.max(np.abs(q1 - q)))
        T = max(3.0, 1.875 * d / self.g("traj_max_vel"))
        self.traj = (q.copy(), q1, T, time.monotonic())
        self.lock_after_traj = (kind == "start")
        self.core.disengage()
        self.mode = "TRAJ"
        moves = ", ".join(f"{self.chain_names[i]} {math.degrees(q[i]):.0f}->{math.degrees(q1[i]):.0f}°"
                          for i in range(self.n) if abs(q1[i] - q[i]) > math.radians(0.5))
        self.get_logger().info(f"{kind.upper()} trajectory T={T:.1f}s: {moves}")
        return f"{kind.upper()} running ({T:.0f}s) - space to abort"

    # ================================================================ control loop
    def _publish(self, dq):
        msg = Float64MultiArray()
        msg.data = [float(x) for x in dq]
        self.pub.publish(msg)

    def _stop_now(self):
        self.last_cmd[:] = 0.0
        self._publish(self.last_cmd)

    def _tick(self):
        now = time.monotonic()
        if not self.kdl_ready or self.js_msg is None:
            return
        if now - self.js_time > self.js_timeout:
            self._stop_now()
            self.core.disengage()
            self.mode = "PLANE" if self.mode == "TRAJ" else self.mode
            self.traj = None
            self.warn_msg = "JOINT_STATES STALE - stopped"
            return
        q, qd_meas = self._q_from_js()
        if q is None:
            self.warn_msg = f"waiting for joints {getattr(self, 'missing_names', '?')} in /joint_states"
            return

        p, R, J = self._fkj_tcp(q)
        with self.lock:
            self.tcp_p, self.tcp_R, self.q_chain, self.q_cmd = p, R, q, self._to_cmd(q)
            self.last_axis = J[3:, self.n - 1].copy()
            alive = (now - self.last_key_time) < self.key_timeout
            rates = self.rates.copy() if alive else np.zeros(4)
            jog = self.jog_dir if alive else 0.0
            sel = self.sel_joint
            do_align, self.request_align = self.request_align, False
            do_freeze, self.request_freeze = self.request_freeze, False
            traj_req, self.request_traj = self.request_traj, None
            if traj_req:
                self.status_msg = self._start_traj(q, traj_req)
            mode = self.mode

        outside = self._outside_safe(q)
        beyond_urdf = [i for i in range(self.n) if q[i] < self.urdf_lo[i] or q[i] > self.urdf_hi[i]]
        self.urdf_note = ("  (beyond URDF range: " + ",".join(
            f"{self.chain_names[i]}={math.degrees(q[i]):.0f}°" for i in beyond_urdf) + ")") if beyond_urdf else ""
        dq = np.zeros(self.n)

        # ------------------------------------------------ TRAJ (recover / home)
        if mode == "TRAJ" and self.traj is not None:
            q0, q1, T, t0 = self.traj
            t = now - t0
            q_des, qd_des = quintic(q0, q1, T, t)
            dq = qd_des + 1.0 * (q_des - q)
            if t > T and np.max(np.abs(q1 - q)) < math.radians(0.5):
                self.traj = None
                with self.lock:
                    self.mode = "PLANE"
                dq[:] = 0.0
                if getattr(self, "lock_after_traj", False):
                    self.lock_after_traj = False
                    with self.lock:
                        self.request_align = "preset"     # exact snap (point/edge params), then plane control
                    self.status_msg = "at start pose - exact alignment + lock..."
                else:
                    self.status_msg = "trajectory done - press K/H (or m for JOINT mode)"
            elif t > T + 5.0:
                self.traj = None
                with self.lock:
                    self.mode = "PLANE"
                self.status_msg = "trajectory timeout - robot not following?"
                dq[:] = 0.0

        # ------------------------------------------------ JOINT jog
        elif mode == "JOINT":
            name = self.cmd_names[sel - 1]
            if name in self.chain_names:
                dq[self.chain_names.index(name)] = jog * self.JNT_SPEEDS[self.speed_idx]

        # ------------------------------------------------ PLANE
        else:
            if outside:
                self.core.disengage()
                self.warn_msg = "OUTSIDE SOFT LIMIT: " + ", ".join(
                    f"{self.chain_names[i]}={math.degrees(q[i]):.0f}°" for i in outside) + "  -> press 0"
            else:
                pmsg = ""
                if do_align == "preset":
                    self._stop_now()
                    pt = str(self.g("align_point"))
                    Rg, ang = preset_orientation(R, pt, str(self.g("align_edge")), str(self.g("align_to")))
                    q_end, conv = simulate_ik(q, p, Rg, self._fkj_tcp)
                    bad = [i for i in range(self.n) if q_end[i] < self.q_lo[i] + math.radians(5)
                           or q_end[i] > self.q_hi[i] - math.radians(5)]
                    if not conv:
                        self.status_msg = "Y refused: not reachable with the tip here - move tip out/up (JOINT mode) and retry"
                    elif bad:
                        self.status_msg = "Y refused: would hit limit on " + ",".join(self.chain_names[i] for i in bad)
                    else:
                        if self.plane_from_tool:
                            self._plane_from_tool(p, Rg)       # same plane rule as K (verified controls)
                        elif pt[-1].upper() in "XY":          # plane faces the pointing direction
                            n = np.zeros(3)
                            n["XY".index(pt[-1].upper())] = -1.0 if pt[0] == "-" else 1.0
                            self.core.wire_yaw = math.atan2(n[0], -n[1])
                        self.core.align(p, R, self.last_axis, R_snap=Rg)
                        self.status_msg = (f"Y: rotating {math.degrees(ang):.1f}° to preset "
                                           f"(swing {math.degrees(np.max(np.abs(q_end - q))):.0f}°), then plane control")
                    self.get_logger().info(self.status_msg)
                    do_align = False
                if do_align and self.plane_from_tool:
                    pmsg = self._plane_from_tool(p, R) + " | "
                if do_align == "as_is":
                    self._stop_now()
                    self.core.align(p, R, self.last_axis, as_is=True)
                    self.status_msg = pmsg + "K: locked as-is. " + self.core.describe(R)
                    self.get_logger().info(self.status_msg)
                elif do_align:
                    self._stop_now()                      # hold still while we simulate
                    if self.plane_from_tool:
                        ok, msg = self.core.plan_align_axis(
                            q, p, R, "xyz".index(self.normal_axis), self.last_axis, self._fkj_tcp,
                            self.q_lo, self.q_hi, margin=math.radians(10.0),
                            max_travel=math.radians(self.g("max_align_travel_deg")))
                    else:
                        ok, msg = self.core.plan_align(
                            q, p, R, self.last_axis, self._fkj_tcp, self.q_lo, self.q_hi,
                            margin=math.radians(10.0),
                            max_travel=math.radians(self.g("max_align_travel_deg")))
                    if ok:
                        msg += " | " + self.core.describe(self.core.R_snap)
                    msg = pmsg + msg
                    self.status_msg = msg
                    self.get_logger().info(msg)
                if do_freeze:
                    self.core.freeze(p)
                dq, info = self.core.step(p, R, J, rates, self.dt)
                if info:
                    self.status_msg = info

        # ------------------------------------------------ predictive singularity stop (PLANE mode only)
        sing_blocked = False
        if mode == "PLANE" and self.core.aligned and np.any(dq):
            s_now = float(np.linalg.svd(J, compute_uv=False)[-1])
            _, _, Jp = self._fkj_tcp(q + dq * self.lookahead)
            s_pred = float(np.linalg.svd(Jp, compute_uv=False)[-1])
            if s_pred < self.g("sing_stop") and s_pred < s_now:
                sing_blocked = True
                dq[:] = 0.0
                self.core.sync_to_current(p, R)
                j5 = math.degrees(q[4]) if self.n >= 5 else float("nan")
                self.warn_msg = (f"SINGULARITY AHEAD σ={s_pred:.3f} (joint5={j5:.0f}°, keep it away from ±90°) "
                                 "-> move another way, or m + jog joint5")

        # ------------------------------------------------ predictive joint-limit stop
        if mode != "TRAJ":
            hits = self._limit_check(q, dq)
            if hits:
                dq[:] = 0.0                                   # stop the WHOLE arm
                if mode == "PLANE":
                    self.core.sync_to_current(p, R)           # forget the unreachable goal
                self.warn_msg = "LIMIT AHEAD: " + ", ".join(
                    f"{self.chain_names[i]}={math.degrees(q[i]):.0f}°" for i in hits) + \
                    "  -> move the other way, or press 0"
            elif not outside and not sing_blocked and (np.any(dq) or not self.warn_msg.startswith("SINGULARITY")):
                self.warn_msg = ""                        # a singularity warning stays until a move succeeds

        # ------------------------------------------------ velocity cap (uniform) + slew
        mx = float(np.max(np.abs(dq)))
        if mx > self.vmax_j:
            dq *= self.vmax_j / mx
        dq_cmd = self._to_cmd(dq)
        step = self.amax_j * self.dt
        dq_cmd = self.last_cmd + np.clip(dq_cmd - self.last_cmd, -step, step)
        self.last_cmd = dq_cmd

        # ------------------------------------------------ following watchdog (uses position change)
        if np.max(np.abs(dq_cmd)) > 0.03:
            if self.nofollow_t == 0.0:
                self.nofollow_q0 = q.copy()
            self.nofollow_t += self.dt
        else:
            self.nofollow_t = 0.0
        if self.nofollow_t > 1.5 and np.max(np.abs(q - self.nofollow_q0)) < math.radians(0.3):
            self.core.disengage()
            self.traj = None
            with self.lock:
                self.mode = "PLANE"
            self.last_cmd[:] = 0.0
            dq_cmd = self.last_cmd.copy()
            self.warn_msg = "ROBOT NOT FOLLOWING (error state?) - stopped. Check heal_server / launch."
            self.nofollow_t = 0.0
        elif self.nofollow_t > 1.5:
            self.nofollow_t = 0.0

        self._publish(dq_cmd)
        self._pub_pose(self.pose_pub, p, R)
        if self.core.aligned:
            self._pub_pose(self.ref_pub, self.core.p_ref, self.core.R_ref)

    def _pub_pose(self, pub, p, R):
        m = PoseStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.base_link
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, p)
        x, y, z, w = rot_to_quat(R)
        m.pose.orientation.x, m.pose.orientation.y = float(x), float(y)
        m.pose.orientation.z, m.pose.orientation.w = float(z), float(w)
        pub.publish(m)

    # ================================================================ recording
    def _toggle_recording(self):
        if self.recording:
            self.recording = False
            self.rec_file.close()
            self.get_logger().info(f"Saved {self.rec_count} samples -> {self.rec_path}")
            return
        os.makedirs(self.record_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.rec_path = os.path.join(self.record_dir, f"wire_path_{stamp}.csv")
        self.rec_file = open(self.rec_path, "w", newline="")
        self.rec_writer = csv.writer(self.rec_file)
        self.rec_writer.writerow(
            ["t", "flag"] + [f"q_{n}" for n in self.cmd_names] +
            ["tcp_x", "tcp_y", "tcp_z", "tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw",
             "plane_u", "plane_v", "plane_d", "yaw_psi"])
        meta = dict(created=stamp, base_link=self.base_link, tip_link=self.tip_link,
                    joint_names=self.cmd_names, tcp_offset_in_tip=self.tcp_offset.tolist(),
                    wire_yaw_deg=math.degrees(self.core.wire_yaw),
                    flags={"0": "sample", "1": "keyframe", "2": "contact"},
                    units="m, rad; quaternion x,y,z,w; poses in base_link")
        with open(self.rec_path.replace(".csv", "_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        self.rec_count, self.rec_t0, self.recording = 0, time.monotonic(), True
        self.get_logger().info(f"RECORDING -> {self.rec_path}")

    def _record_tick(self):
        if not self.recording or self.tcp_p is None:
            return
        with self.lock:
            q, p, R = self.q_cmd.copy(), self.tcp_p.copy(), self.tcp_R.copy()
            flag, self.pending_flag = self.pending_flag, 0
        c = self.core.plane_coords(p)
        self.rec_writer.writerow(
            [f"{time.monotonic() - self.rec_t0:.4f}", flag] + [f"{x:.6f}" for x in q] +
            [f"{x:.6f}" for x in p] + [f"{x:.7f}" for x in rot_to_quat(R)] +
            [f"{x:.6f}" for x in c] + [f"{self.core.psi:.6f}"])
        self.rec_count += 1
        if self.rec_count % 100 == 0:
            self.rec_file.flush()

    # ================================================================ status line
    def _status_tick(self):
        if self.tcp_p is None:
            sys.stdout.write("\r waiting for /robot_description and /joint_states ...   ")
            sys.stdout.flush()
            return
        c = self.core.plane_coords(self.tcp_p)
        rec = f"REC {self.rec_count:5d}" if self.recording else "rec off"
        st = "ALIGNED" if self.core.aligned else "free"
        mode = f"JOINT j{self.sel_joint}" if self.mode == "JOINT" else self.mode
        qd = " ".join(f"{math.degrees(x):+4.0f}" for x in self.q_chain) if self.q_chain is not None else ""
        msg = (self.warn_msg or self.status_msg) + getattr(self, "urdf_note", "")
        if self.core.aligned and self.core.s_min == self.core.s_min and self.core.s_min < 0.02 \
                and not msg.startswith("SINGULARITY AHEAD"):
            j5 = math.degrees(self.q_chain[4]) if self.q_chain is not None else float("nan")
            msg = f"NEAR SINGULARITY σ={self.core.s_min:.3f} (j5={j5:.0f}°; keep j5 away from ±90°, ideally <60° or >140°) | " + msg
        src = ("t" + self.normal_axis) if self.plane_from_tool else "LR"
        spd = (f"{100 * self.LIN_SPEEDS[self.speed_idx]:g}cm/s" if self.mode != "JOINT"
               else f"{self.JNT_SPEEDS[self.speed_idx]:g}rad/s")
        line = (f"\r[{mode}|{st}] pl{math.degrees(self.core.wire_yaw):+.0f}°({src}) spd{self.speed_idx}={spd} {rec} u={c[0]:+.3f} v={c[1]:+.3f} d={c[2]:+.3f} "
                f"yaw={math.degrees(self.core.psi):+5.1f} q°[{qd}] {msg[:70]:70s}")
        sys.stdout.write(line)
        sys.stdout.flush()

    # ================================================================ keys
    def handle_key(self, k):
        now = time.monotonic()
        lin, yaw = self.LIN_SPEEDS[self.speed_idx], self.YAW_SPEEDS[self.speed_idx]
        plane_keys = {
            "z": (0, +lin * self.sgn_lr), "x": (0, -lin * self.sgn_lr),
            "w": (1, +lin), "s": (1, -lin),
            "a": (2, +lin * self.sgn_off), "d": (2, -lin * self.sgn_off),
            "q": (3, -yaw), "e": (3, +yaw),
        }
        with self.lock:
            if self.mode == "TRAJ" and k != " ":
                self.status_msg = "trajectory running - space to abort"
                return
            if self.mode == "PLANE" and k in plane_keys:
                if not self.core.aligned:
                    self.status_msg = "press H first"
                    return
                ax, val = plane_keys[k]
                self.rates[:] = 0.0
                self.rates[ax] = val
                self.last_key_time = now
            elif k in ("H", "h"):
                self.mode = "PLANE"
                self.request_align = "snap"
                self.status_msg = "planning H..."
            elif k == "Y":
                self.mode = "PLANE"
                self.request_align = "preset"
                self.status_msg = "aligning to preset..."
            elif k in ("T", "t"):
                self.normal_axis = {"x": "y", "y": "z", "z": "x"}[self.normal_axis]
                self.plane_from_tool = True
                self.mode = "PLANE"
                self.request_align = "snap"
                self.status_msg = f"plane normal -> tool {self.normal_axis}, re-aligning"
            elif k == "K":
                self.mode = "PLANE"
                self.request_align = "as_is"
                self.status_msg = "locking current orientation..."
            elif k == "v" and self.tcp_R is not None:
                sys.stdout.write("\n  plane yaw %.1f°  %s\n" % (math.degrees(self.core.wire_yaw),
                                                              self.core.describe(self.tcp_R)))
            elif k == "0":
                self.request_traj = "recover"
            elif k == "G":
                self.request_traj = "start"
            elif k == "S" and self.q_cmd is not None:
                sp = [round(float(math.degrees(x)), 2) for x in self.q_cmd]
                try:
                    with open(self.g("start_pose_file"), "w") as f:
                        json.dump({"joints_deg": sp, "joint_names": self.cmd_names}, f)
                    spd = dict(zip(self.cmd_names, np.radians(sp)))
                    self.q_start = np.array([spd[nm] for nm in self.chain_names])
                    self.status_msg = f"start pose SAVED: {sp}"
                except Exception as e:
                    self.status_msg = f"save failed: {e}"
                self.get_logger().info(self.status_msg)
            elif k == "9":
                self.request_traj = "home"
            elif self.mode == "JOINT" and k in "123456":
                self.sel_joint = int(k)
            elif self.mode == "JOINT" and k in ("=", "+", "-"):
                self.jog_dir = -1.0 if k == "-" else 1.0
                self.last_key_time = now
            elif k == " ":
                self.rates[:] = 0.0
                self.jog_dir = 0.0
                self.last_key_time = 0.0
                if self.mode == "TRAJ":
                    self.traj = None
                    self.mode = "PLANE"
                    self.core.disengage()
                self.request_freeze = True
                self.status_msg = "STOP"
            elif k == "m":
                self.mode = "JOINT" if self.mode == "PLANE" else "PLANE"
                self.rates[:] = 0.0
                self.jog_dir = 0.0
                self.core.disengage()
                self.status_msg = "JOINT mode" if self.mode == "JOINT" else "press H to align"
            elif k == "[":
                self.speed_idx = max(0, self.speed_idx - 1)
            elif k == "]":
                self.speed_idx = min(len(self.LIN_SPEEDS) - 1, self.speed_idx + 1)
            elif k == "{":                                   # Shift+[ : slowest at once
                self.speed_idx = 0
            elif k == "}":                                   # Shift+] : fastest at once
                self.speed_idx = len(self.LIN_SPEEDS) - 1
            elif k == "p":
                self.pending_flag = 1
                self.status_msg = "keyframe"
            elif k == "b":
                self.pending_flag = 2
                self.status_msg = "CONTACT marked"
            elif k in ("L", "R") and self.tcp_p is not None:
                if k == "L":
                    self.anchor_L = self.tcp_p.copy()
                else:
                    self.anchor_R = self.tcp_p.copy()
                if self.anchor_L is not None and self.anchor_R is not None:
                    dv = self.anchor_R - self.anchor_L
                    self.core.wire_yaw = math.atan2(dv[1], dv[0])
                    self.plane_from_tool = False            # L/R plane wins until restart or T
                    self.core.disengage()
                    self.anchor_L = self.anchor_R = None
                    self.status_msg = (f"plane set: yaw {math.degrees(self.core.wire_yaw):.1f}° "
                                       f"(use -p wire_yaw_deg:={math.degrees(self.core.wire_yaw):.1f} next time). Press K or H")
                else:
                    self.status_msg = f"anchor {k} stored"
            elif k == "?":
                sys.stdout.write("\n" + HELP + "\n")
        if k == "r":
            self._toggle_recording()
        elif k in ("g", "o"):
            self._gripper(self.grip_close if k == "g" else self.grip_open)

    def _gripper(self, pos):
        label = "CLOSE" if pos == self.grip_close else "OPEN"
        if self.grip_cli is not None:
            if not self.grip_cli.service_is_ready():
                self.status_msg = f"gripper service {self.grip_srv} not available"
                return
            req = GripperSrv.Request()
            req.position = float(pos)
            req.grasp_force = float(self.grip_force)
            fut = self.grip_cli.call_async(req)
            fut.add_done_callback(lambda f: self.get_logger().info(f"gripper {label}: {f.result()}"))
            self.status_msg = f"gripper {label} sent"
            return
        cmd = (f'ros2 service call {self.grip_srv} addverb_cobot_msgs/srv/Gripper '
               f'"{{position: {float(pos)}, grasp_force: {float(self.grip_force)}}}"')
        threading.Thread(target=subprocess.run, args=(cmd,),
                         kwargs=dict(shell=True, executable="/bin/bash", capture_output=True),
                         daemon=True).start()
        self.status_msg = f"gripper {label} sent (cli)"

    def shutdown(self):
        for _ in range(5):
            try:
                self._publish(np.zeros(len(self.cmd_names)))
            except Exception:
                pass
            time.sleep(0.01)
        if self.recording:
            self._toggle_recording()


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = HealPlaneTeleop()
    print(HELP)
    kb = Keyboard()
    from rclpy.executors import SingleThreadedExecutor
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
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
        print("\nStopping: zero velocity.")
        node.shutdown()
        executor.shutdown(timeout_sec=1.0)      # stop spinning cleanly before tearing down
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
