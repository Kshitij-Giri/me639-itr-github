#!/usr/bin/env python3
"""
plane_core.py — pure-numpy core of the HEAL "plane teleop" (no ROS, unit-testable).

Plane frame W (base frame columns):
    u : horizontal, in the plane   (a/d)
    v : +z, vertical, in the plane (w/s)
    n : horizontal plane normal, pointing away from the robot (z/x = plane offset)

State after 'H' (align):
    su, sv : TCP coordinates inside the plane
    d      : plane offset along n  (TCP depth)
    psi    : final-joint yaw angle applied on top of the squared-up orientation
Goal pose:
    p = su*u + sv*v + d*n
    R = Rot(a_yaw, psi) @ R_snap
where R_snap = current tool orientation snapped to the nearest orientation whose axes are all
along ±u/±v/±n (gripper "squared up" to the plane), and a_yaw = final-joint axis after the snap.
"""
from __future__ import annotations

import itertools
import math

import numpy as np


# ----------------------------------------------------------------------------- SO(3)
def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def so3_exp(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3) + skew(w)
    k = skew(w / th)
    return np.eye(3) + math.sin(th) * k + (1.0 - math.cos(th)) * (k @ k)


def so3_log(R):
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    th = math.acos(c)
    if th < 1e-9:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if math.pi - th < 1e-6:
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


def plane_frame(wire_yaw):
    """columns [u, v, n].  wire_yaw = direction of u in base XY.  n = u x v."""
    u = np.array([math.cos(wire_yaw), math.sin(wire_yaw), 0.0])
    v = np.array([0.0, 0.0, 1.0])
    return np.column_stack([u, v, np.cross(u, v)])


_SIGNED_PERMS = []
for perm in itertools.permutations(range(3)):
    for signs in itertools.product((1.0, -1.0), repeat=3):
        P = np.zeros((3, 3))
        for col, (row, s) in enumerate(zip(perm, signs)):
            P[row, col] = s
        if np.linalg.det(P) > 0:
            _SIGNED_PERMS.append(P)          # 24 proper rotations


def snap_to_frame(R, W):
    """Nearest orientation to R whose axes are all along ±columns of W."""
    M = W.T @ R
    best = max(_SIGNED_PERMS, key=lambda P: float(np.sum(P * M)))
    return W @ best


def tcp_jacobian(J6, R_tip, tcp_offset):
    r = R_tip @ np.asarray(tcp_offset, float)
    J = J6.copy()
    J[:3, :] = J6[:3, :] - skew(r) @ J6[3:, :]
    return J


def dls(J, e, lam0, eps=0.015, lam_max=0.06):
    # eps/lam_max tuned to the Heal: normal working poses have sigma ~0.03-0.06; only the
    # wrist singularity band (joint5 within ~15 deg of +/-90) falls below 0.015.
    """Damped least squares with adaptive damping near singularities."""
    s_min = float(np.linalg.svd(J, compute_uv=False)[-1])
    lam2 = lam0 ** 2
    if s_min < eps:
        lam2 += (1.0 - (s_min / eps) ** 2) * lam_max ** 2
    return J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(J.shape[0]), e), s_min


# ----------------------------------------------------------------------------- core
class PlaneCore:
    def __init__(self, wire_yaw=math.pi / 2, kp_lin=2.0, kp_rot=2.0, damping=0.05,
                 ref_max_lin=0.06, ref_max_ang=0.40, align_lin=0.02, align_ang=0.15,
                 max_lead=0.01, ws_min=(-1, -1, 0.02), ws_max=(1, 1, 1.2)):
        self.wire_yaw = wire_yaw
        self.kp_lin, self.kp_rot, self.lam = kp_lin, kp_rot, damping
        self.ref_max_lin, self.ref_max_ang = ref_max_lin, ref_max_ang
        self.align_lin, self.align_ang = align_lin, align_ang
        self.max_lead = max_lead
        self.ws_min, self.ws_max = np.array(ws_min, float), np.array(ws_max, float)
        self.aligned = False
        self.aligning = False
        self.free_spin = True          # final joint locked & spin free unless q/e is pressed
        self.su = self.sv = self.d = self.psi = 0.0
        self.R_snap = None
        self.a_yaw = None
        self.p_ref = self.R_ref = None
        self.s_min = float("nan")

    # --- helpers
    @property
    def W(self):
        return plane_frame(self.wire_yaw)

    def goal(self, su=None, sv=None, d=None, psi=None):
        W = self.W
        su = self.su if su is None else su
        sv = self.sv if sv is None else sv
        d = self.d if d is None else d
        psi = self.psi if psi is None else psi
        p = su * W[:, 0] + sv * W[:, 1] + d * W[:, 2]
        R = so3_exp(self.a_yaw * psi) @ self.R_snap
        return p, R

    # --- events
    def align(self, p_tcp, R_tcp, last_joint_axis_base, R_snap=None, as_is=False):
        """'H': square the gripper up to the plane through the current TCP.
        as_is=True ('K'): keep the current orientation exactly (no snapping)."""
        W = self.W
        if as_is:
            self.R_snap = R_tcp.copy()
        else:
            self.R_snap = snap_to_frame(R_tcp, W) if R_snap is None else R_snap
        a_tool = R_tcp.T @ (last_joint_axis_base / np.linalg.norm(last_joint_axis_base))
        a = self.R_snap @ a_tool
        if as_is:
            self.a_yaw = a / np.linalg.norm(a)
        else:
            # snap the yaw axis to the nearest plane axis too (exact when the joint axis is a tool axis)
            k = int(np.argmax(np.abs(W.T @ a)))
            self.a_yaw = W[:, k] * np.sign(W[:, k] @ a)
        self.su, self.sv, self.d = (W.T @ p_tcp).tolist()
        self.psi = 0.0
        self.p_ref, self.R_ref = p_tcp.copy(), R_tcp.copy()
        self.aligned, self.aligning = True, True

    def plan_align(self, q, p_tcp, R_tcp, last_joint_axis_base, fkj, q_lo, q_hi,
                   margin=math.radians(10), max_travel=math.radians(60), max_snap=math.radians(95)):
        """Like align(), but simulate every nearby squared-up orientation and pick the one that
        stays inside joint limits with the smallest joint travel.
        Returns (ok, message)."""
        W = self.W
        best = None
        reasons = []
        # Only the NEAREST squared-up orientation is considered, so H always does what you expect.
        for ang, S in candidate_snaps(R_tcp, W, max_snap)[:1]:
            q_end, conv = simulate_ik(q, p_tcp, S, fkj)
            if not conv:
                reasons.append(f"{math.degrees(ang):.0f}°: IK no converge")
                continue
            lo_bad = q_end < (q_lo + margin)
            hi_bad = q_end > (q_hi - margin)
            if (lo_bad | hi_bad).any():
                j = int(np.argmax(lo_bad | hi_bad))
                reasons.append(f"{math.degrees(ang):.0f}°: j{j+1} -> {math.degrees(q_end[j]):.0f}° near limit")
                continue
            travel = float(np.max(np.abs(q_end - q)))
            if travel > max_travel:
                reasons.append(f"{math.degrees(ang):.0f}°: needs {math.degrees(travel):.0f}° joint swing")
                continue
            if best is None or travel < best[0]:
                best = (travel, S, ang)
        if best is None:
            return False, "H refused (reposition first): " + "; ".join(reasons[:3])
        self.align(p_tcp, R_tcp, last_joint_axis_base, R_snap=best[1])
        return True, f"H ok: rotate {math.degrees(best[2]):.0f}°, max joint swing {math.degrees(best[0]):.0f}°"

    def describe(self, R):
        """Which way each tool axis points, in plane words."""
        names = ["along", "up", "away"]
        W = self.W
        out = []
        for i, ax in enumerate("xyz"):
            c = W.T @ R[:, i]
            k = int(np.argmax(np.abs(c)))
            word = names[k] if c[k] > 0 else {"along": "-along", "up": "down", "away": "toward-robot"}[names[k]]
            out.append(f"tool{ax}->{word}({math.degrees(math.acos(min(1, abs(c[k])))):.0f}°off)")
        return " ".join(out)

    def plan_align_axis(self, q, p_tcp, R_tcp, axis_idx, last_joint_axis_base, fkj, q_lo, q_hi,
                        margin=math.radians(10), max_travel=math.radians(60)):
        """H: minimal rotation that makes tool axis `axis_idx` equal to the plane normal n.
        The other two tool axes then lie in the plane; roll about n is kept as the user set it."""
        n = self.W[:, 2]
        a = R_tcp[:, axis_idx]
        if np.dot(a, n) < 0:
            n_t = -n                      # keep the axis' own sense, avoid a 180° flip
        else:
            n_t = n
        c = float(np.clip(np.dot(a, n_t), -1.0, 1.0))
        axis = np.cross(a, n_t)
        ang = math.acos(c)
        R_goal = R_tcp.copy() if np.linalg.norm(axis) < 1e-9 else \
            so3_exp(axis / np.linalg.norm(axis) * ang) @ R_tcp
        q_end, conv = simulate_ik(q, p_tcp, R_goal, fkj)
        note = ""
        if not conv:
            # pre-check inconclusive: still align (slowly); runtime limit-stop + watchdog protect.
            note = " (pre-check inconclusive - aligning slowly, watch it)"
            q_end = q
        bad = (q_end < q_lo + margin) | (q_end > q_hi - margin)
        if bad.any():
            j = int(np.argmax(bad))
            return False, f"H refused: j{j+1} would go to {math.degrees(q_end[j]):.0f}° (near limit)"
        travel = float(np.max(np.abs(q_end - q)))
        if travel > max_travel:
            return False, f"H refused: needs {math.degrees(travel):.0f}° joint swing - reposition closer first"
        self.align(p_tcp, R_tcp, last_joint_axis_base, R_snap=R_goal)
        # yaw = final joint axis, exactly (no snapping to plane axes)
        a_tool = R_tcp.T @ (last_joint_axis_base / np.linalg.norm(last_joint_axis_base))
        ay = R_goal @ a_tool
        self.a_yaw = ay / np.linalg.norm(ay)
        return True, f"H ok: tilt correction {math.degrees(ang):.0f}°" + (note or f", max joint swing {math.degrees(travel):.0f}°")

    def sync_to_current(self, p_tcp, R_tcp):
        """Re-base the goal on where the arm actually is (used after a limit stop)."""
        if not self.aligned:
            return
        self.su, self.sv, self.d = (self.W.T @ p_tcp).tolist()
        r = so3_log(R_tcp @ self.R_snap.T)
        self.psi = float(np.dot(r, self.a_yaw))
        self.p_ref, self.R_ref = p_tcp.copy(), R_tcp.copy()

    def freeze(self, p_tcp):
        """Space: stop where we are (keep plane, orientation and yaw)."""
        if not self.aligned:
            return
        su, sv, d = self.W.T @ p_tcp
        self.su, self.sv = su, sv
        self.p_ref = self.goal()[0]

    def disengage(self):
        self.aligned = self.aligning = False

    # --- control step
    def step(self, p_tcp, R_tcp, J_tcp, rates, dt):
        """rates = (du, dv, dd, dpsi) in m/s and rad/s from keys.  Returns (dq, info)."""
        n = J_tcp.shape[1]
        if not self.aligned:
            return np.zeros(n), "press H to align"

        info = ""
        # 1) integrate plane parameters from keys (blocked while still aligning)
        if not self.aligning and np.any(rates):
            du, dv, dd, dpsi = rates
            cand = (self.su + du * dt, self.sv + dv * dt, self.d + dd * dt, self.psi + dpsi * dt)
            p_new, _ = self.goal(*cand)
            dist_new = np.linalg.norm(p_new - p_tcp)
            dist_old = np.linalg.norm(self.goal()[0] - p_tcp)
            if np.any(p_new < self.ws_min) or np.any(p_new > self.ws_max):
                info = "workspace limit"
            elif dist_new > self.max_lead and dist_new > dist_old:
                info = "arm lagging"
            else:
                self.su, self.sv, self.d, self.psi = cand

        # 2) move the reference toward the goal with rate limits
        p_goal, R_goal = self.goal()
        lin_rate = self.align_lin if self.aligning else self.ref_max_lin
        ang_rate = self.align_ang if self.aligning else self.ref_max_ang
        dp = p_goal - self.p_ref
        dp_n = np.linalg.norm(dp)
        step_p = dp if dp_n <= lin_rate * dt else dp * (lin_rate * dt / dp_n)
        dr = so3_log(R_goal @ self.R_ref.T)
        dr_n = np.linalg.norm(dr)
        step_r = dr if dr_n <= ang_rate * dt else dr * (ang_rate * dt / dr_n)
        self.p_ref = self.p_ref + step_p
        self.R_ref = so3_exp(step_r) @ self.R_ref
        if self.aligning:
            info = "aligning..."
            if dp_n < 1e-4 and dr_n < 1e-3:
                self.aligning = False

        # 3) feed-forward + P on tracking error
        e_p = self.p_ref - p_tcp
        e_r = so3_log(self.R_ref @ R_tcp.T)
        v = step_p / dt + self.kp_lin * e_p
        w = step_r / dt + self.kp_rot * e_r
        yawing = abs(rates[3]) > 0 if rates is not None else False
        if self.free_spin and not yawing and not self.aligning:
            # Spin about the final-joint axis is NOT controlled and the final joint is LOCKED (dq=0).
            # Task = 3 position + 2 rotations perpendicular to the final-joint axis, solved by j1..j(n-1).
            a = J_tcp[3:, n - 1] / max(np.linalg.norm(J_tcp[3:, n - 1]), 1e-9)
            t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
            b1 = np.cross(a, t); b1 /= np.linalg.norm(b1)
            b2 = np.cross(a, b1)
            B = np.column_stack([b1, b2])
            Jt = np.vstack([J_tcp[:3, :n - 1], B.T @ J_tcp[3:, :n - 1]])
            xt = np.concatenate([v, B.T @ w])
            e_rp = B.T @ e_r
            if np.linalg.norm(e_p) < 3e-4 and np.linalg.norm(e_rp) < 2e-3 and not np.any(step_p) \
                    and not np.any(B.T @ step_r):
                xt[:] = 0.0
            dq5, self.s_min = dls(Jt, xt, self.lam)
            dq = np.zeros(n)
            dq[:n - 1] = dq5
            # follow the real spin so the goal never fights it (no jump when q/e is pressed later)
            self.psi -= float(np.dot(e_r, self.a_yaw))
            return dq, info
        xi = np.concatenate([v, w])
        if np.linalg.norm(e_p) < 3e-4 and np.linalg.norm(e_r) < 2e-3 and not np.any(step_p) \
                and not np.any(step_r):
            xi[:] = 0.0                       # deadband: no chatter when holding
        dq, self.s_min = dls(J_tcp, xi, self.lam)
        return dq, info

    # --- for display / recording
    def plane_coords(self, p_tcp):
        return self.W.T @ p_tcp


# ----------------------------------------------------------------------------- planning helpers
def candidate_snaps(R, W, max_angle):
    """All axis-aligned orientations within max_angle [rad] of R, nearest first."""
    out = []
    for P in _SIGNED_PERMS:
        S = W @ P
        ang = float(np.linalg.norm(so3_log(S @ R.T)))
        if ang <= max_angle:
            out.append((ang, S))
    out.sort(key=lambda x: x[0])
    return out


def simulate_ik(q0, p_goal, R_goal, fkj, iters=1500, lam=0.05, max_step=0.05):
    """Numerically predict the joint configuration the controller will end in.
    fkj(q) -> (p_tcp, R_tcp, J_tcp).  Returns (q_final, converged)."""
    q = np.array(q0, float)
    for _ in range(iters):
        p, R, J = fkj(q)
        e = np.concatenate([p_goal - p, so3_log(R_goal @ R.T)])
        if np.linalg.norm(e[:3]) < 1e-3 and np.linalg.norm(e[3:]) < math.radians(1.0):
            return q, True
        dq, _ = dls(J, e, lam)
        m = float(np.max(np.abs(dq)))
        if m > max_step:
            dq *= max_step / m
        q = q + dq
    return q, False


def quintic(q0, q1, T, t):
    """Rest-to-rest quintic. Returns (q, qd). Peak |qd| = 1.875*|q1-q0|/T."""
    s = min(max(t / T, 0.0), 1.0)
    d = np.asarray(q1) - np.asarray(q0)
    pos = 10 * s**3 - 15 * s**4 + 6 * s**5
    vel = (30 * s**2 - 60 * s**3 + 30 * s**4) / T
    return np.asarray(q0) + d * pos, d * vel


def preset_orientation(R, point="+Y", edge="x", to="X"):
    """Orientation with tool z along `point` (+X -X +Y -Y +Z -Z) and tool `edge` (x|y) along +/- `to`
    (X|Y|Z), choosing the sign that is the smaller turn from R. Same maths as align_gripper_axis.py."""
    axes = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}
    pt = point.upper()
    zg = (-1.0 if pt[0] == "-" else 1.0) * axes[pt[-1]]
    t = axes[to.upper()]
    if abs(zg @ t) > 0.5:
        raise ValueError(f"align_to {to} is not perpendicular to align_point {point}")
    best = None
    for sgn in (1.0, -1.0):
        e = sgn * t
        if edge == "x":
            xg, yg = e, np.cross(zg, e)
        else:
            yg, xg = e, np.cross(e, zg)
        Rg = np.column_stack([xg, yg, zg])
        ang = float(np.linalg.norm(so3_log(Rg @ R.T)))
        if best is None or ang < best[0]:
            best = (ang, Rg)
    return best[1], best[0]
