#!/usr/bin/env python3
"""
replay_core.py — pure-numpy core of the HEAL buzz-wire IK replay (no ROS, unit-testable).

Path follower with
  * path parameter s advanced at  s_dot = speed_scale * v_ref(s) * governor(error)
  * reference pose p(s), R(s) from the prepared path (linear position, normalised-lerp quaternion)
  * full 6-D DLS-IK on velocity: twist = feed-forward(dp/ds, dR/ds * s_dot) + P on pose error
  * error governor (clearance based):
        e < e_slow            -> full speed
        e_slow .. e_hold      -> speed scaled down linearly to 0
        e > e_hold            -> path progress HOLDS (arm keeps converging onto the reference)
        e > e_abort           -> ABORT (zero velocity)
    and the same for orientation (deg)
"""
import math

import numpy as np

from plane_core import so3_exp, so3_log, dls


def quat_to_R(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_prepared(path_csv):
    d = np.genfromtxt(path_csv, delimiter=",", names=True)
    P = np.column_stack([d["x"], d["y"], d["z"]])
    Q = np.column_stack([d["qx"], d["qy"], d["qz"], d["qw"]])
    for i in range(1, len(Q)):                      # keep quaternion signs continuous
        if Q[i] @ Q[i - 1] < 0:
            Q[i] = -Q[i]
    return d["s"].astype(float), P, Q, d["v_ref"].astype(float)


class PathFollower:
    def __init__(self, s, P, Q, v_ref, reverse=False, speed_scale=0.5, clearance_mm=5.0,
                 kp_lin=3.0, kp_rot=2.0, damping=0.05, rot_slow_deg=3.0, rot_hold_deg=6.0, rot_abort_deg=12.0):
        if reverse:
            L = s[-1]
            s = L - s[::-1]
            P, Q, v_ref = P[::-1].copy(), Q[::-1].copy(), v_ref[::-1].copy()
        self.s, self.P, self.Q, self.v = s, P, Q, v_ref
        self.L = float(s[-1])
        self.scale = speed_scale
        c = clearance_mm / 1000.0
        self.e_slow, self.e_hold, self.e_abort = c / 3.0, c / 2.0, 0.8 * c
        self.r_slow, self.r_hold, self.r_abort = (math.radians(rot_slow_deg), math.radians(rot_hold_deg),
                                                  math.radians(rot_abort_deg))
        self.kp_lin, self.kp_rot, self.lam = kp_lin, kp_rot, damping
        self.sp = 0.0                 # current path parameter
        self.done = False
        self.aborted = ""
        self.hold_time = 0.0
        self.max_e = 0.0
        self.max_er = 0.0
        self.sum_e = 0.0
        self.n = 0
        self.governor = 1.0
        self.s_min = float("nan")
        self.approaching = True       # first settle onto the path's first pose, then advance
        self.approach_lin = 0.005     # m/s max during approach
        self.approach_ang = 0.10      # rad/s max during approach

    # ---------------------------------------------------------------- reference along the path
    def ref(self, sp):
        sp = min(max(sp, 0.0), self.L)
        i = int(np.searchsorted(self.s, sp, side="right") - 1)
        i = min(max(i, 0), len(self.s) - 2)
        f = (sp - self.s[i]) / max(self.s[i + 1] - self.s[i], 1e-12)
        p = (1 - f) * self.P[i] + f * self.P[i + 1]
        q = (1 - f) * self.Q[i] + f * self.Q[i + 1]
        v = (1 - f) * self.v[i] + f * self.v[i + 1]
        return p, quat_to_R(q / np.linalg.norm(q)), v

    def nearest(self, p, R):
        """Nearest path point to a pose: returns (s, dist_m, angle_rad)."""
        d = np.linalg.norm(self.P - p, axis=1)
        i = int(np.argmin(d))
        ang = float(np.linalg.norm(so3_log(quat_to_R(self.Q[i]) @ R.T)))
        return float(self.s[i]), float(d[i]), ang

    def resume_from(self, sp):
        self.sp = float(min(max(sp, 0.0), self.L))
        self.done = False
        self.aborted = ""
        self.approaching = True

    def start_pose(self):
        p, R, _ = self.ref(0.0)
        return p, R

    def end_pose(self):
        p, R, _ = self.ref(self.L)
        return p, R

    # ---------------------------------------------------------------- one control step
    def step(self, p, R, J, dt):
        """p, R: measured TCP pose; J: 6xn base-frame TCP Jacobian. Returns (dq, info)."""
        n = J.shape[1]
        if self.done or self.aborted:
            return np.zeros(n), self.aborted or "done"
        p_r, R_r, v_r = self.ref(self.sp)
        e_p = p_r - p
        e_r = so3_log(R_r @ R.T)
        e, er = float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))
        self.max_e, self.max_er = max(self.max_e, e), max(self.max_er, er)
        self.sum_e += e
        self.n += 1
        # ---- approach phase: converge onto the first pose before the path starts to advance
        if self.approaching:
            if e < 0.5 * self.e_slow and er < 0.5 * self.r_slow:
                self.approaching = False
            else:
                v = self.kp_lin * e_p
                w = self.kp_rot * e_r
                nv, nw = np.linalg.norm(v), np.linalg.norm(w)
                if nv > self.approach_lin:
                    v *= self.approach_lin / nv
                if nw > self.approach_ang:
                    w *= self.approach_ang / nw
                dq, self.s_min = dls(J, np.concatenate([v, w]), self.lam)
                return dq, f"approaching path start ({1000*e:.1f} mm, {math.degrees(er):.1f}°)"
        # ---- governor
        if e > self.e_abort or er > self.r_abort:
            self.aborted = (f"ABORT: tracking error {1000*e:.1f} mm / {math.degrees(er):.1f}° "
                            f"at {100*self.sp:.1f} cm (limits {1000*self.e_abort:.1f} mm / "
                            f"{math.degrees(self.r_abort):.0f}°)")
            return np.zeros(n), self.aborted
        g_lin = np.clip((self.e_hold - e) / (self.e_hold - self.e_slow), 0.0, 1.0) if e > self.e_slow else 1.0
        g_rot = np.clip((self.r_hold - er) / (self.r_hold - self.r_slow), 0.0, 1.0) if er > self.r_slow else 1.0
        self.governor = float(min(g_lin, g_rot))
        if self.governor <= 0.0:
            self.hold_time += dt
        sdot = self.scale * max(v_r, 0.002) * self.governor
        sp_next = min(self.sp + sdot * dt, self.L)
        # ---- feed-forward from the reference motion over this step
        p_n, R_n, _ = self.ref(sp_next)
        v_ff = (p_n - p_r) / dt
        w_ff = so3_log(R_n @ R_r.T) / dt
        self.sp = sp_next
        xi = np.concatenate([v_ff + self.kp_lin * e_p, w_ff + self.kp_rot * e_r])
        dq, self.s_min = dls(J, xi, self.lam)
        if self.sp >= self.L and e < 0.5e-3 and er < math.radians(0.5):
            self.done = True
        return dq, ""

    def stats(self):
        return dict(max_err_mm=round(1000 * self.max_e, 2), mean_err_mm=round(1000 * self.sum_e / max(self.n, 1), 2),
                    max_rot_err_deg=round(math.degrees(self.max_er), 2), hold_time_s=round(self.hold_time, 2),
                    progress_cm=round(100 * self.sp, 1), length_cm=round(100 * self.L, 1))
