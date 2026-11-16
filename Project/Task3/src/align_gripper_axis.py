#!/usr/bin/env python3
"""
align_gripper_axis.py — make the gripper point exactly down (-Z) and line one gripper axis up
with a base axis.  Tool tip stays where it is.  ROS 2 Humble, Python 3.10.

Gripper (rq_tool_tip) axes:
    x = finger open/close direction  (the gripper's long edge)
    y = across the flat face          (the gripper's short edge)
    z = approach, out of the fingertips
Base (base_link) axes:  Z up, +Y = robot's front at home, X = sideways.

Usage (velocity_control.launch.py running, teleop NOT running):
    python3 align_gripper_axis.py --point -Z --edge x --to Y   # point down, fingers along Y
    python3 align_gripper_axis.py --point +Y --edge x --to X   # point forward (+Y), fingers along X
    python3 align_gripper_axis.py --point +X --edge x --to Z   # point along +X, fingers vertical
    add --dry-run to only print the plan.
--point = where the gripper's approach axis (tool z) should point: +X -X +Y -Y +Z -Z
--to    = base axis for the gripper edge; must be perpendicular to --point. Sign is chosen automatically.
The move is simulated first (joint limits checked); you confirm with Enter before anything moves.
Options:
    --max-vel 0.10  joint speed cap [rad/s]
The script picks the smaller of the two possible turns (+axis or -axis), since the gripper is symmetric.
Ctrl-C or Space stops immediately (zero velocity).
"""
import argparse
import math
import os
import select
import sys
import termios
import threading
import time
import tty

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
import PyKDL as kdl
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromUrdfModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plane_core import so3_exp, so3_log, dls, simulate_ik  # noqa: E402

JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
LO = np.radians([-170.0, -170.0, -170.0, -165.0, -170.0, -170.0])   # same soft limits as the teleop
HI = -LO
MARGIN = math.radians(5.0)


class Aligner(Node):
    def __init__(self, args):
        super().__init__("align_gripper_axis")
        self.args = args
        self.q = None
        self.t_js = 0.0
        self.kdl_ok = False
        self.R_goal = None
        self.p_goal = None
        self.done = False
        self.stop = False
        self.awaiting = False
        self.R_ref = None
        self.last = np.zeros(6)
        self.pub = self.create_publisher(Float64MultiArray, "/velocity_controller/commands", 1)
        self.create_subscription(JointState, "/joint_states", self.js_cb, qos_profile_sensor_data)
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                         durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.desc_sub = self.create_subscription(String, "/robot_description", self.desc_cb, qos)
        self.create_timer(0.01, self.tick)

    def js_cb(self, m):
        if not all(n in m.name for n in JOINTS):
            return                                   # ignore partial / foreign messages
        idx = {n: i for i, n in enumerate(m.name)}
        self.q = np.array([m.position[idx[n]] for n in JOINTS])
        self.t_js = time.monotonic()

    def desc_cb(self, m):
        if self.kdl_ok:
            return
        ok, tree = treeFromUrdfModel(URDF.from_xml_string(m.data))
        self.chain = tree.getChain("base_link", self.args.tip)
        names = [self.chain.getSegment(i).getJoint().getName() for i in range(self.chain.getNrOfSegments())
                 if self.chain.getSegment(i).getJoint().getTypeName() not in ("None", "Fixed")]
        assert names == JOINTS, f"unexpected chain joints {names}"
        self.fk = kdl.ChainFkSolverPos_recursive(self.chain)
        self.jac = kdl.ChainJntToJacSolver(self.chain)
        self.kdl_ok = True

    def fkj(self, q):
        qk = kdl.JntArray(6)
        for i in range(6):
            qk[i] = float(q[i])
        F = kdl.Frame()
        self.fk.JntToCart(qk, F)
        Jk = kdl.Jacobian(6)
        self.jac.JntToJac(qk, Jk)
        J = np.array([[Jk[r, c] for c in range(6)] for r in range(6)])
        R = np.array([[F.M[r, c] for c in range(3)] for r in range(3)])
        return np.array([F.p[0], F.p[1], F.p[2]]), R, J

    def plan(self, p, R):
        """Goal orientation: tool z along --point, tool --edge along +/- --to (smaller turn)."""
        axes = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}
        pt = self.args.point.upper()
        zg = (-1.0 if pt[0] == "-" else 1.0) * axes[pt[-1]]
        to = axes[self.args.to.upper()]
        if abs(zg @ to) > 0.5:
            raise SystemExit(f"--to {self.args.to} is not perpendicular to --point {self.args.point}")
        ei = {"x": 0, "y": 1}[self.args.edge]
        best = None
        for sgn in (1.0, -1.0):
            e = sgn * to
            if ei == 0:                      # x = e, y = z cross x
                xg, yg = e, np.cross(zg, e)
            else:                            # y = e, x = y cross z
                yg, xg = e, np.cross(e, zg)
            Rg = np.column_stack([xg, yg, zg])
            ang = float(np.linalg.norm(so3_log(Rg @ R.T)))
            if best is None or ang < best[0]:
                best = (ang, Rg)
        return best[1], best[0]

    def send(self, v):
        m = Float64MultiArray()
        m.data = [float(x) for x in v]
        self.pub.publish(m)

    def tick(self):
        if not self.kdl_ok or self.q is None:
            return
        if self.stop or self.done:
            self.send(np.zeros(6))
            return
        if time.monotonic() - self.t_js > 0.2:
            self.send(np.zeros(6))
            self.get_logger().warn("joint_states stale - stopping")
            self.stop = True
            return
        p, R, J = self.fkj(self.q)
        if self.R_goal is None:
            self.R_goal, ang = self.plan(p, R)
            self.p_goal = p.copy()
            q_end, conv = simulate_ik(self.q, p, self.R_goal, self.fkj)
            print(f"\n plan: rotate {math.degrees(ang):.1f}° about the tool tip -> approach {self.args.point}, "
                  f"tool {self.args.edge} along base {self.args.to}")
            print(f" now     [deg]: {np.round(np.degrees(self.q), 1)}")
            if not conv:
                print(" pre-check: not reachable with the tip HERE (wrist would have to sit behind the tip).\n"
                  " Move the tip further out / higher with the teleop (w/s, a/d), quit it, and retry.")
                self.done = True
                return
            print(f" predict [deg]: {np.round(np.degrees(q_end), 1)}")
            bad = [i for i in range(6) if q_end[i] < LO[i] or q_end[i] > HI[i]]
            if bad:
                print(" refusing: would exceed soft limits on " + ", ".join(JOINTS[i] for i in bad))
                self.done = True
                return
            if self.args.dry_run:
                print(" --dry-run: nothing moved.")
                self.done = True
                return
            print(" press ENTER to move (Space/Ctrl-C stops at any time), anything else cancels")
            self.awaiting = True
            self.R_ref = R.copy()
            return
        if self.awaiting:
            self.send(np.zeros(6))
            return
        # move the reference toward the goal at a limited angular rate (smooth, no big error jumps)
        dr = so3_log(self.R_goal @ self.R_ref.T)
        n = float(np.linalg.norm(dr))
        stepr = dr if n <= self.args.ang_rate * 0.01 else dr * (self.args.ang_rate * 0.01 / n)
        self.R_ref = so3_exp(stepr) @ self.R_ref
        # predictive joint-limit stop
        e = np.concatenate([self.p_goal - p, so3_log(self.R_ref @ R.T)])
        if np.linalg.norm(e[:3]) < 1e-3 and np.linalg.norm(so3_log(self.R_goal @ R.T)) < math.radians(0.2):
            self.done = True
            self.send(np.zeros(6))
            print(f"\n DONE. tool x = {np.round(R[:, 0], 4)}, y = {np.round(R[:, 1], 4)}, z = {np.round(R[:, 2], 4)}")
            print(f" joints [deg] = {np.round(np.degrees(self.q), 2)}")
            return
        xi = np.concatenate([1.5 * e[:3], stepr / 0.01 + 1.5 * e[3:]])
        dq, _ = dls(J, xi, 0.05)
        qp = self.q + dq * 0.5
        hit = ((qp < LO + MARGIN) & (dq < 0)) | ((qp > HI - MARGIN) & (dq > 0))
        if hit.any():
            print("\n STOP: joint limit ahead on " + ", ".join(JOINTS[i] for i in np.where(hit)[0]))
            self.stop = True
            self.send(np.zeros(6))
            return
        m = float(np.max(np.abs(dq)))
        if m > self.args.max_vel:
            dq *= self.args.max_vel / m
        step = 0.3 * 0.01                                   # 0.3 rad/s^2 slew
        dq = self.last + np.clip(dq - self.last, -step, step)
        self.last = dq
        self.send(dq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", default="-Z", help="approach direction: +X -X +Y -Y +Z -Z (default -Z = down)")
    ap.add_argument("--edge", choices=["x", "y"], default="x", help="gripper axis to align (x = finger direction)")
    ap.add_argument("--to", choices=["X", "Y", "Z"], default="Y", help="base axis for that edge")
    ap.add_argument("--ang-rate", type=float, default=0.08, help="re-orientation speed [rad/s]")
    ap.add_argument("--tip", default="rq_tool_tip")
    ap.add_argument("--max-vel", type=float, default=0.10)
    ap.add_argument("--dry-run", action="store_true")
    args, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    node = Aligner(args)
    from rclpy.executors import SingleThreadedExecutor
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    th = threading.Thread(target=ex.spin, daemon=True)
    th.start()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd) if sys.stdin.isatty() else None
    if old:
        tty.setcbreak(fd)
    try:
        while rclpy.ok() and not node.done:
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not r:
                continue
            ch = sys.stdin.read(1)
            if node.awaiting:
                if ch in ("\n", "\r"):
                    node.awaiting = False
                    print(" moving...")
                else:
                    print(" cancelled.")
                    node.stop = True
                    break
            elif ch == " ":
                node.stop = True
                print("\n STOPPED by Space.")
                break
        time.sleep(0.2)
    except KeyboardInterrupt:
        node.stop = True
    finally:
        if old:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        for _ in range(5):
            node.send(np.zeros(6))
            time.sleep(0.01)
        ex.shutdown(timeout_sec=1.0)
        th.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
