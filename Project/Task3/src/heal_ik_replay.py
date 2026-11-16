#!/usr/bin/env python3
"""
heal_ik_replay.py — autonomous buzz-wire replay on the HEAL with DLS-IK + velocity control (ROS 2 Humble)

Prepare the path first (once per recording):
    python3 stitch_recordings.py <part1.csv> <part2.csv> ... -o recordings/wire_path_stitched.csv   (if needed)
    python3 prepare_path.py recordings/wire_path_stitched.csv -o recordings/wire_path_prepared.csv --plot p.png
Run (teleop must NOT be running):
    python3 heal_ik_replay.py --ros-args -p path:=recordings/wire_path_prepared.csv -p clearance_mm:=5.0

Keys
  ENTER : start the replay (from the path end nearest to where the arm is, i.e. REVERSE after teaching)
  space : pause / resume (arm holds on the path)
  x     : abort now (zero velocity)
  + / - : speed scale up / down by 0.1  (default 0.5 = first runs; 1.0 = the prepared profile)
  c     : CONTINUE from the nearest point on the path (after an abort / restart mid-wire), same direction
  E     : go to the recorded END pose   (slow joint-space move, confirm with y) - for re-gripping
  S     : go to the recorded START pose (slow joint-space move, confirm with y)
  f     : flip direction (forward <-> reverse) while idle
  g / o : gripper close / open
  ?     : help        Ctrl-C : quit (zero velocity)

Safety (all always on)
  * only complete /joint_states messages are used; stale joint states -> zero velocity
  * refuses to start / aborts if more than one node publishes /velocity_controller/commands
  * every joint capped at max_joint_vel (0.30 rad/s, driver rejects > 0.5) with uniform scaling,
    acceleration-limited (0.6 rad/s^2)
  * predictive joint-limit stop (soft limits +-170 deg, joint4 +-165 deg)
  * singularity guard (holds if sigma_min would drop below sing_stop)
  * clearance-based tracking governor: slows at clearance/3, holds at clearance/2, aborts at 0.8*clearance
  * following watchdog: commanding motion but joints not moving for 1.5 s -> abort
Every trial is appended to <path folder>/trials.csv (time, direction, scale, duration, max error, result).
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
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import PoseStamped

import PyKDL as kdl
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromUrdfModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from plane_core import rot_to_quat, quintic, so3_log, dls  # noqa: E402
from replay_core import PathFollower, load_prepared  # noqa: E402

try:
    from addverb_cobot_msgs.srv import Gripper as GripperSrv  # type: ignore
except Exception:
    GripperSrv = None

HELP = __doc__.split("Safety (all always on)")[0]
CMD_TOPIC = "/velocity_controller/commands"


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


class HealIkReplay(Node):
    def __init__(self):
        super().__init__("heal_ik_replay")
        p = self.declare_parameter
        p("path", "recordings/wire_path_prepared.csv")
        p("base_link", "base_link")
        p("tip_link", "rq_tool_tip")
        p("joint_names", ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"])
        p("speed_scale", 0.5)
        p("clearance_mm", 5.0)              # (ring inner diameter - wire diameter) / 2  -> MEASURE IT
        p("direction", "auto")              # auto (nearest end) | reverse | forward
        p("start_tol_mm", 10.0)             # arm must be this close to the path start to begin
        p("start_tol_deg", 15.0)
        p("max_joint_vel", 0.30)
        p("max_joint_acc", 0.6)
        p("soft_lower_deg", [-170.0, -170.0, -170.0, -165.0, -170.0, -170.0])
        p("soft_upper_deg", [170.0, 170.0, 170.0, 165.0, 170.0, 170.0])
        p("limit_margin_deg", 5.0)
        p("lookahead_s", 0.5)
        p("sing_stop", 0.012)
        p("goto_max_vel", 0.12)
        p("rate_hz", 100.0)
        p("js_timeout", 0.2)
        p("gripper_service", "/gripper_controller/command")
        p("gripper_open", 1.0)
        p("gripper_close", 0.0)
        p("gripper_force", 40.0)
        g = lambda n: self.get_parameter(n).value
        self.g = g
        self.names = list(g("joint_names"))
        self.dt = 1.0 / g("rate_hz")

        # ---- path
        self.path_csv = os.path.abspath(g("path"))
        self.s, self.P, self.Q, self.v = load_prepared(self.path_csv)
        meta_p = self.path_csv.replace(".csv", "_meta.json")
        self.meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
        self.q_start_rec = np.array(self.meta["start"]["q_rad"]) if "start" in self.meta else None
        self.q_end_rec = np.array(self.meta["end"]["q_rad"]) if "end" in self.meta else None
        self.direction = str(g("direction"))
        self.speed_scale = float(g("speed_scale"))
        self.reverse = True
        self.n_pub = 1
        self.pending_goto = None
        self.get_logger().info(
            f"path {os.path.basename(self.path_csv)}: {100*self.s[-1]:.1f} cm, {len(self.s)} points, "
            f"est. {self.meta.get('speed', {}).get('est_duration_s', '?')} s at scale 1.0")

        # ---- state
        self.state = "WAIT"          # WAIT | IDLE | CONFIRM_GOTO | GOTO | RUN | PAUSE | DONE | ABORT
        self.follower = None
        self.goto = None             # (q0, q1, T, t0, name)
        self.goto_req = None
        self.msg = ""
        self.kdl_ready = False
        self.js_pos, self.js_t = None, 0.0
        self.last_cmd = np.zeros(6)
        self.t_run0 = 0.0
        self.nofollow_t, self.nofollow_q0 = 0.0, None
        self.pub_check_t = 0.0
        self.lock = threading.Lock()
        self.tcp = None

        self.pub = self.create_publisher(Float64MultiArray, CMD_TOPIC, 1)
        self.ref_pub = self.create_publisher(PoseStamped, "/wire_replay/reference", 10)
        self.create_subscription(JointState, "/joint_states", self._js_cb, qos_profile_sensor_data)
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                         durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self._desc_sub = self.create_subscription(String, "/robot_description", self._desc_cb, qos)
        self.create_timer(self.dt, self._tick)
        self.create_timer(0.25, self._status)
        self.grip_cli = self.create_client(GripperSrv, g("gripper_service")) if GripperSrv else None

    # ================================================================ ROS inputs
    def _js_cb(self, m):
        if not all(n in m.name for n in self.names):
            return                              # partial / foreign message
        idx = {n: i for i, n in enumerate(m.name)}
        self.js_pos = np.array([m.position[idx[n]] for n in self.names])
        self.js_t = time.monotonic()

    def _desc_cb(self, m):
        if self.kdl_ready:
            return
        ok, tree = treeFromUrdfModel(URDF.from_xml_string(m.data))
        chain = tree.getChain(self.g("base_link"), self.g("tip_link"))
        # KEEP references: KDL solvers only hold a pointer to the chain; if the Python objects are
        # garbage-collected the solver silently returns zeros (this was the 711 mm bug).
        self.kdl_tree, self.kdl_chain = tree, chain
        names = [chain.getSegment(i).getJoint().getName() for i in range(chain.getNrOfSegments())
                 if chain.getSegment(i).getJoint().getTypeName() not in ("None", "Fixed")]
        if names != self.names:
            self.get_logger().error(f"chain joints {names} != {self.names}")
            return
        self.fk = kdl.ChainFkSolverPos_recursive(self.kdl_chain)
        self.jac = kdl.ChainJntToJacSolver(self.kdl_chain)
        self.q_lo = np.radians(self.g("soft_lower_deg"))
        self.q_hi = np.radians(self.g("soft_upper_deg"))
        self.kdl_ready = True
        self.destroy_subscription(self._desc_sub)
        if self.q_end_rec is not None and "end" in self.meta:
            p_chk, _, _ = self.fkj(self.q_end_rec)
            err = 1000 * float(np.linalg.norm(p_chk - np.array(self.meta["end"]["tcp"])))
            if err > 2.0:
                self.kdl_ready = False
                self.get_logger().error(
                    f"FK SELF-CHECK FAILED: recorded END joints give tip {np.round(p_chk, 4).tolist()} but the "
                    f"recording says {np.round(self.meta['end']['tcp'], 4).tolist()} ({err:.1f} mm). "
                    f"Robot model differs from recording time - NOT running.")
            else:
                self.get_logger().info(f"FK self-check OK: recorded END joints reproduce the recorded tip ({err:.2f} mm)")

    def fkj(self, q):
        qk = kdl.JntArray(6)
        for i in range(6):
            qk[i] = float(q[i])
        F = kdl.Frame()
        rc = self.fk.JntToCart(qk, F)
        if rc is not None and rc < 0:
            raise RuntimeError(f"KDL FK failed (code {rc})")
        Jk = kdl.Jacobian(6)
        self.jac.JntToJac(qk, Jk)
        J = np.array([[Jk[r, c] for c in range(6)] for r in range(6)])
        R = np.array([[F.M[r, c] for c in range(3)] for r in range(3)])
        return np.array([F.p[0], F.p[1], F.p[2]]), R, J

    # ================================================================ helpers
    def _publish(self, dq):
        m = Float64MultiArray()
        m.data = [float(x) for x in dq]
        self.pub.publish(m)

    def _zero(self):
        self.last_cmd[:] = 0.0
        self._publish(self.last_cmd)

    def _make_follower(self, p, R):
        """Choose direction and build the follower; returns (follower, distance_mm, angle_deg)."""
        best = None
        dirs = {"reverse": [True], "forward": [False], "auto": [True, False]}[self.direction]
        for rev in dirs:
            f = PathFollower(self.s, self.P, self.Q, self.v, reverse=rev,
                             speed_scale=self.speed_scale, clearance_mm=float(self.g("clearance_mm")))
            ps, Rs = f.start_pose()
            d = float(np.linalg.norm(ps - p))
            a = math.degrees(np.linalg.norm(so3_log(Rs @ R.T)))
            if best is None or d < best[1]:
                best = (f, d, a, rev)
        return best

    def _abort(self, why):
        self._zero()
        if self.follower is not None and self.state in ("RUN", "PAUSE"):
            self._log_trial("ABORT: " + why)
        self.state = "ABORT"
        self.msg = why
        self.get_logger().error(why)

    def _log_trial(self, result):
        st = self.follower.stats() if self.follower else {}
        path = os.path.join(os.path.dirname(self.path_csv), "trials.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["date", "path", "direction", "speed_scale", "duration_s", "result", "max_err_mm",
                            "mean_err_mm", "max_rot_err_deg", "hold_time_s", "progress_cm", "length_cm"])
            w.writerow([datetime.now().isoformat(timespec="seconds"), os.path.basename(self.path_csv),
                        "reverse" if self.reverse else "forward", self.follower.scale if self.follower else "",
                        round(time.monotonic() - self.t_run0, 1), result, st.get("max_err_mm"),
                        st.get("mean_err_mm"), st.get("max_rot_err_deg"), st.get("hold_time_s"),
                        float(st.get("progress_cm", 0) or 0), float(st.get("length_cm", 0) or 0)])
        self.get_logger().info(f"trial logged -> {path}: {result} {st}")

    # ================================================================ control loop
    def _tick(self):
        now = time.monotonic()
        if not self.kdl_ready or self.js_pos is None:
            return
        if now - self.js_t > self.g("js_timeout"):
            self._zero()
            if self.state in ("RUN", "GOTO"):
                self._abort("joint_states stale")
            return
        q = self.js_pos.copy()
        p, R, J = self.fkj(q)
        self.tcp = (p, R)
        if self.state == "WAIT":
            self.state = "IDLE"

        # single-publisher check (another teleop / script / PC on the same domain = danger)
        if now - self.pub_check_t > 1.0:
            self.pub_check_t = now
            n_pub = self.count_publishers(CMD_TOPIC)
            if n_pub > 1 and self.state in ("RUN", "GOTO", "PAUSE"):
                self._abort(f"{n_pub} publishers on {CMD_TOPIC} - stop the other node (teleop?)")
            self.n_pub = n_pub

        dq = np.zeros(6)
        with self.lock:
            state = self.state
            if self.goto_req is not None:
                self._start_goto(q, *self.goto_req)
                self.goto_req = None
                state = self.state
        if state == "RUN":
            dq, info = self.follower.step(p, R, J, self.dt)
            if info.startswith("ABORT"):
                self._abort(info)
                return
            self.msg = info
            if self.follower.done:
                self._zero()
                self.state = "DONE"
                dur = time.monotonic() - self.t_run0
                self.msg = f"DONE in {dur:.1f} s"
                self._log_trial("DONE")
                return
            # singularity guard: refuse to drive into the joint5 ~ +-90 deg wrist singularity
            if np.any(dq):
                s_now = float(np.linalg.svd(J, compute_uv=False)[-1])
                _, _, Jp = self.fkj(q + dq * self.g("lookahead_s"))
                s_pred = float(np.linalg.svd(Jp, compute_uv=False)[-1])
                if s_pred < self.g("sing_stop") and s_pred < s_now:
                    self._abort(f"singularity ahead (sigma {s_pred:.3f}, joint5={math.degrees(q[4]):.0f}°)")
                    return
        elif state == "PAUSE":
            # hold on the current reference point
            pr, Rr, _ = self.follower.ref(self.follower.sp)
            xi = np.concatenate([3.0 * (pr - p), 2.0 * so3_log(Rr @ R.T)])
            dq, _ = dls(J, xi, 0.05)
        elif state == "GOTO":
            q0, q1, T, t0, name = self.goto
            t = now - t0
            qd, qdd = quintic(q0, q1, T, t)
            dq = qdd + 1.0 * (qd - q)
            if t > T and np.max(np.abs(q1 - q)) < math.radians(0.3):
                self._zero()
                self.state = "IDLE"
                self.msg = f"at recorded {name} pose - grip if needed, then ENTER"
                return

        # ---- predictive joint limits (all moving states)
        if np.any(dq):
            q_pred = q + dq * self.g("lookahead_s")
            m = math.radians(self.g("limit_margin_deg"))
            hit = ((q_pred < self.q_lo + m) & (dq < 0)) | ((q_pred > self.q_hi - m) & (dq > 0))
            if hit.any():
                self._abort("joint limit ahead: " + ", ".join(
                    f"{self.names[i]}={math.degrees(q[i]):.0f}°" for i in np.where(hit)[0]))
                return
        # ---- velocity cap (uniform) + acceleration limit
        mx = float(np.max(np.abs(dq)))
        if mx > self.g("max_joint_vel"):
            dq *= self.g("max_joint_vel") / mx
        step = self.g("max_joint_acc") * self.dt
        dq = self.last_cmd + np.clip(dq - self.last_cmd, -step, step)
        self.last_cmd = dq
        # ---- following watchdog
        if np.max(np.abs(dq)) > 0.03:
            if self.nofollow_t == 0.0:
                self.nofollow_q0 = q.copy()
            self.nofollow_t += self.dt
            if self.nofollow_t > 1.5 and np.max(np.abs(q - self.nofollow_q0)) < math.radians(0.3):
                self._abort("robot not following commands (error state?) - check heal_server")
                return
        else:
            self.nofollow_t = 0.0
        self._publish(dq)
        if self.follower is not None:
            pr, Rr, _ = self.follower.ref(self.follower.sp)
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.g("base_link")
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, pr)
            x, y, z, w = rot_to_quat(Rr)
            msg.pose.orientation.x, msg.pose.orientation.y = float(x), float(y)
            msg.pose.orientation.z, msg.pose.orientation.w = float(z), float(w)
            self.ref_pub.publish(msg)

    def _start_goto(self, q, name, q1):
        bad = [self.names[i] for i in range(6) if q1[i] < self.q_lo[i] or q1[i] > self.q_hi[i]]
        if bad:
            self.msg = f"{name} pose outside soft limits on {bad}"
            self.state = "IDLE"
            return
        d = float(np.max(np.abs(q1 - q)))
        T = max(3.0, 1.875 * d / self.g("goto_max_vel"))
        self.goto = (q.copy(), q1.copy(), T, time.monotonic(), name)
        self.state = "GOTO"
        self.msg = f"going to recorded {name} pose ({T:.0f} s) - space/x stops"

    # ================================================================ status
    def _status(self):
        if self.tcp is None:
            sys.stdout.write("\r waiting for /robot_description and /joint_states ...   ")
            sys.stdout.flush()
            return
        f = self.follower
        if f is not None and self.state in ("RUN", "PAUSE", "DONE", "ABORT"):
            pr, Rr, _ = f.ref(f.sp)
            e = 1000 * np.linalg.norm(pr - self.tcp[0])
            er = math.degrees(np.linalg.norm(so3_log(Rr @ self.tcp[1].T)))
            t = time.monotonic() - self.t_run0 if self.state in ("RUN", "PAUSE") else 0
            line = (f"\r[{self.state}] {'REV' if self.reverse else 'FWD'} x{f.scale:.1f} "
                    f"{100*f.sp/f.L:5.1f}% ({100*f.sp:5.1f}/{100*f.L:.1f} cm) err {e:4.1f}mm {er:4.1f}° "
                    f"gov {f.governor:.2f} σ {f.s_min:.3f} t {t:5.1f}s {self.msg[:45]:45s}")
        else:
            line = f"\r[{self.state}] {self.msg[:110]:110s}"
        sys.stdout.write(line)
        sys.stdout.flush()

    # ================================================================ keys
    def handle_key(self, k):
        with self.lock:
            st = self.state
            if st == "CONFIRM_GOTO":
                if k == "y":
                    self.goto_req = self.pending_goto
                    self.msg = "starting go-to..."
                else:
                    self.state = "IDLE"
                    self.msg = "go-to cancelled"
                return
            if k in ("\n", "\r"):
                if st in ("IDLE", "DONE", "ABORT"):
                    self._try_start()
            elif k == " ":
                if st == "RUN":
                    self.state = "PAUSE"
                    self.msg = "PAUSED - space to resume"
                elif st == "PAUSE":
                    self.state = "RUN"
                    self.msg = ""
                elif st == "GOTO":
                    self._zero()
                    self.state = "IDLE"
                    self.msg = "go-to stopped"
            elif k == "x":
                if st in ("RUN", "PAUSE", "GOTO"):
                    self._abort("aborted by user")
            elif k in ("+", "="):
                self._scale(+0.1)
            elif k == "-":
                self._scale(-0.1)
            elif k in ("E", "S") and st in ("IDLE", "DONE", "ABORT"):
                qt = self.q_end_rec if k == "E" else self.q_start_rec
                if qt is None:
                    self.msg = "no recorded start/end joints in the path meta file"
                else:
                    self.pending_goto = ("END" if k == "E" else "START", qt)
                    self.state = "CONFIRM_GOTO"
                    self.msg = (f"go to recorded {self.pending_goto[0]} pose? path NOT collision checked - "
                                f"ring must be OFF the wire or at that end. y = go, any other key = cancel")
            elif k == "c" and st in ("IDLE", "ABORT", "DONE"):
                self._try_continue()
            elif k == "f" and st in ("IDLE", "DONE", "ABORT"):
                self.direction = {"auto": "forward", "forward": "reverse", "reverse": "forward"}[self.direction]
                self.msg = f"direction: {self.direction}"
            elif k == "?":
                sys.stdout.write("\n" + HELP + "\n")
        if k in ("g", "o"):
            self._gripper(self.g("gripper_close") if k == "g" else self.g("gripper_open"))

    def _scale(self, d):
        self.speed_scale = float(np.clip(round(self.speed_scale + d, 1), 0.1, 1.5))
        if self.follower:
            self.follower.scale = self.speed_scale
        self.msg = f"speed scale {self.speed_scale:.1f}"

    def _try_start(self):
        self.n_pub = self.count_publishers(CMD_TOPIC)
        if self.n_pub > 1:
            self.msg = f"refusing: {self.n_pub} publishers on {CMD_TOPIC} (teleop still running?)"
            return
        p, R = self.tcp
        f, d, a, rev = self._make_follower(p, R)
        if d > self.g("start_tol_mm") / 1000 or a > self.g("start_tol_deg"):
            qn = np.degrees(self.js_pos)
            dom = os.environ.get("ROS_DOMAIN_ID", "<unset>")
            lines = [f"START REFUSED - diagnostics (ROS_DOMAIN_ID={dom})",
                     f"  current joints [deg]      : {np.round(qn, 1).tolist()}",
                     f"  current tool tip [m]      : {np.round(p, 4).tolist()}"]
            for nm in ("end", "start"):
                if nm in self.meta:
                    lines.append(f"  recorded {nm.upper():5s} tip in file [m]: {np.round(self.meta[nm]['tcp'], 4).tolist()}")
            lines.append(f"  KDL chain: {self.g('base_link')} -> {self.g('tip_link')}, "
                         f"robot_description publishers: {self.count_publishers('/robot_description')}")
            for nm, qr in (("END", self.q_end_rec), ("START", self.q_start_rec)):
                if qr is not None:
                    pr_, _, _ = self.fkj(qr)
                    lines.append(f"  recorded {nm:5s} joints [deg]: {np.round(np.degrees(qr), 1).tolist()}"
                                 f"  -> tip {np.round(pr_, 4).tolist()}  (max joint diff "
                                 f"{np.max(np.abs(np.degrees(qr) - qn)):.1f}°)")
            self.get_logger().error("\n".join(lines))
            self.msg = (f"refusing: arm is {1000*d:.1f} mm / {a:.1f}° from the nearest path end"
                        f" (tol {self.g('start_tol_mm')} mm / {self.g('start_tol_deg')}°). Mid-wire? press c. "
                        f"Ring off the wire? E or S")
            return
        self.follower, self.reverse = f, rev
        self.t_run0 = time.monotonic()
        self.last_cmd[:] = 0.0
        self.state = "RUN"
        self.msg = f"start ({'REVERSE' if rev else 'FORWARD'}), {1000*d:.1f} mm / {a:.1f}° from path start"
        self.get_logger().info(self.msg)

    def _try_continue(self):
        self.n_pub = self.count_publishers(CMD_TOPIC)
        if self.n_pub > 1:
            self.msg = f"refusing: {self.n_pub} publishers on {CMD_TOPIC}"
            return
        p, R = self.tcp
        if self.follower is None:                       # e.g. after a restart: pick direction by param
            rev = self.direction != "forward"
            self.follower = PathFollower(self.s, self.P, self.Q, self.v, reverse=rev, speed_scale=self.speed_scale,
                                         clearance_mm=float(self.g("clearance_mm")))
            self.reverse = rev
        sp, d, a = self.follower.nearest(p, R)
        if d > self.g("start_tol_mm") / 1000 or math.degrees(a) > self.g("start_tol_deg"):
            self.msg = (f"refusing: nearest path point is {1000*d:.1f} mm / {math.degrees(a):.1f}° away "
                        f"(tol {self.g('start_tol_mm')} mm / {self.g('start_tol_deg')}°)")
            return
        self.follower.scale = self.speed_scale
        self.follower.resume_from(sp)
        self.t_run0 = time.monotonic()
        self.last_cmd[:] = 0.0
        self.state = "RUN"
        self.msg = f"continuing {'REVERSE' if self.reverse else 'FORWARD'} from {100*sp:.1f} cm ({1000*d:.1f} mm away)"
        self.get_logger().info(self.msg)

    def _gripper(self, pos):
        if self.grip_cli is not None and self.grip_cli.service_is_ready():
            req = GripperSrv.Request()
            req.position = float(pos)
            req.grasp_force = float(self.g("gripper_force"))
            self.grip_cli.call_async(req)
            self.msg = "gripper command sent"
            return
        cmd = (f'ros2 service call {self.g("gripper_service")} addverb_cobot_msgs/srv/Gripper '
               f'"{{position: {float(pos)}, grasp_force: {float(self.g("gripper_force"))}}}"')
        threading.Thread(target=subprocess.run, args=(cmd,),
                         kwargs=dict(shell=True, executable="/bin/bash", capture_output=True), daemon=True).start()
        self.msg = "gripper command sent (cli)"

    def shutdown(self):
        for _ in range(5):
            try:
                self._publish(np.zeros(6))
            except Exception:
                pass
            time.sleep(0.01)
        if self.state in ("RUN", "PAUSE") and self.follower is not None:
            self._log_trial("ABORT: quit")


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = HealIkReplay()
    print(HELP)
    kb = Keyboard()
    from rclpy.executors import SingleThreadedExecutor
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    th = threading.Thread(target=ex.spin, daemon=True)
    th.start()
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
        ex.shutdown(timeout_sec=1.0)
        th.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
