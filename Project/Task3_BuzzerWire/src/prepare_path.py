#!/usr/bin/env python3
"""
prepare_path.py — turn a (stitched) teleop recording into ONE smooth, replay-ready wire path.

    python3 prepare_path.py recordings/wire_path_stitched.csv -o recordings/wire_path_prepared.csv
    python3 prepare_path.py ... --smooth-mm 4 --plot recordings/prepared.png

Pure numpy (+ matplotlib only for --plot). Pipeline:
  1. invalid rows      : all-zero joints / impossible jumps are dropped (robot-fault samples)
  2. spin angle psi    : the gripper pointing axis is (near) constant; psi = ring rotation about it
  3. pauses            : keep a sample only if the tip moved >= --dwell-mm or psi changed >= --dwell-deg
  4. back-and-forth    : if the path later returns within --loop-mm (and --loop-deg) of an earlier point,
                         the detour in between is cut; reversals that come back are removed this way
  5. one smooth path   : resample by arc length (position + psi), Gaussian smoothing over --smooth-mm,
                         which also turns the teleop STAIRCASE (one-axis-at-a-time moves) into the
                         curve it approximates. Max deviation from the cleaned path is reported.
  6. speed profile     : fast on straight stretches, slow at bends and at ring rotations, acceleration
                         limited (smooth), see --v-straight / --v-corner / --r-corner-mm / --w-max
  7. output CSV        : s, x,y,z, qx,qy,qz,qw, psi, v_ref, radius ; plus _meta.json with the start/end
                         joint angles (for the go-to-start / go-to-end commands) and statistics
"""
import argparse
import json
import math
import os

import numpy as np


# ------------------------------------------------------------------ small maths
def quat_to_R(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
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


def rodrigues(axis, ang):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def gauss_smooth(y, sigma_pts):
    """Gaussian smoothing along axis 0 with reflected ends (keeps the end points in place)."""
    if sigma_pts < 0.5:
        return y.copy()
    r = int(math.ceil(4 * sigma_pts))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_pts) ** 2)
    k /= k.sum()
    pad_lo = 2 * y[0] - y[r:0:-1]          # point reflection: preserves end position and slope
    pad_hi = 2 * y[-1] - y[-2:-r - 2:-1]
    yp = np.concatenate([pad_lo, y, pad_hi])
    out = np.empty_like(y)
    for c in range(y.shape[1]):
        out[:, c] = np.convolve(yp[:, c], k, mode="valid")
    return out


def resample(S, Y, ds):
    s_new = np.arange(0.0, S[-1], ds)
    if S[-1] - s_new[-1] > 1e-9:
        s_new = np.append(s_new, S[-1])
    return s_new, np.column_stack([np.interp(s_new, S, Y[:, c]) for c in range(Y.shape[1])])


# ------------------------------------------------------------------ pipeline
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--dwell-mm", type=float, default=0.25)
    ap.add_argument("--dwell-deg", type=float, default=0.5)
    ap.add_argument("--loop-mm", type=float, default=1.5)
    ap.add_argument("--loop-deg", type=float, default=5.0)
    ap.add_argument("--loop-min-detour-mm", type=float, default=3.0)
    ap.add_argument("--tube-mm", type=float, default=3.0,
                    help="smooth path stays within this distance of the cleaned recording (corner cut bound)")
    ap.add_argument("--band-iters", type=int, default=600, help="elastic-band smoothing iterations")
    ap.add_argument("--psi-smooth-mm", type=float, default=10.0, help="Gaussian sigma for ring rotation")
    ap.add_argument("--round-mm", type=float, default=5.0, help="final light rounding of kinks (Gaussian sigma)")
    ap.add_argument("--psi-mm-per-rad", type=float, default=5.0, help="weight of ring rotation in arc length")
    ap.add_argument("--ds-mm", type=float, default=1.0, help="output spacing")
    ap.add_argument("--v-straight", type=float, default=0.03, help="m/s on straight stretches")
    ap.add_argument("--v-corner", type=float, default=0.01, help="m/s at bends")
    ap.add_argument("--r-corner-mm", type=float, default=30.0, help="radius at/below which v = v_corner")
    ap.add_argument("--r-straight-mm", type=float, default=80.0, help="radius at/above which v = v_straight")
    ap.add_argument("--w-max", type=float, default=0.25, help="max ring rotation rate rad/s")
    ap.add_argument("--a-max", type=float, default=0.02, help="path acceleration m/s^2")
    ap.add_argument("--plot", default="")
    a = ap.parse_args()

    d = np.genfromtxt(a.csv, delimiter=",", names=True)
    q = np.column_stack([d[f"q_joint{i}"] for i in range(1, 7)])
    P = np.column_stack([d["tcp_x"], d["tcp_y"], d["tcp_z"]])
    Q = np.column_stack([d["tcp_qx"], d["tcp_qy"], d["tcp_qz"], d["tcp_qw"]])
    flag = d["flag"].astype(int)

    # 1. invalid rows
    ok = ~np.all(np.abs(q) < 1e-9, axis=1)
    step = np.r_[0.0, np.linalg.norm(np.diff(P, axis=0), axis=1)]
    bad_jump = np.where(step > 0.005)[0]
    if len(bad_jump):
        ok[bad_jump[0]:] = False
    q, P, Q, flag = q[ok], P[ok], Q[ok], flag[ok]
    n_raw = int(ok.sum())

    # 2. pointing axis + spin angle psi
    Rs = [quat_to_R(x) for x in Q]
    Z = np.array([R[:, 2] for R in Rs])
    a_pt = Z.mean(0)
    a_pt /= np.linalg.norm(a_pt)
    point_spread = max(math.degrees(math.acos(min(1.0, float(z @ a_pt)))) for z in Z[::50])
    R0 = Rs[0]
    psi = np.array([math.atan2(float((R0[:, 0] @ R[:, 1]) - (R0[:, 1] @ R[:, 0])),
                               float((R0[:, 0] @ R[:, 0]) + (R0[:, 1] @ R[:, 1]))) for R in Rs])
    psi = np.unwrap(psi)
    # sign convention: positive psi = rotation about +a_pt. Check with a small numeric test:
    test = rodrigues(a_pt, 0.3) @ R0
    t_psi = math.atan2(float((R0[:, 0] @ test[:, 1]) - (R0[:, 1] @ test[:, 0])),
                       float((R0[:, 0] @ test[:, 0]) + (R0[:, 1] @ test[:, 1])))
    if t_psi < 0:
        psi = -psi

    # 3. pauses
    keep = [0]
    for i in range(1, len(P)):
        j = keep[-1]
        if (np.linalg.norm(P[i] - P[j]) >= a.dwell_mm / 1000 or abs(psi[i] - psi[j]) >= math.radians(a.dwell_deg)
                or flag[i] in (1, 3)):
            keep.append(i)
    if keep[-1] != len(P) - 1:
        keep.append(len(P) - 1)
    keep = np.array(keep)
    Pk, psik, flk = P[keep], psi[keep], flag[keep]
    n_after_dwell = len(keep)

    # 4. back-and-forth (detours that come back)
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(Pk, axis=0), axis=1))]
    out_idx, removed = [], []
    i, N = 0, len(Pk)
    while i < N:
        out_idx.append(i)
        later = np.arange(i + 2, N)
        if len(later):
            dist = np.linalg.norm(Pk[later] - Pk[i], axis=1)
            cand = later[(dist < a.loop_mm / 1000) &
                         (np.abs(psik[later] - psik[i]) < math.radians(a.loop_deg)) &
                         (seg[later] - seg[i] > a.loop_min_detour_mm / 1000)]
            # never jump over a keyframe/join
            if len(cand):
                j = int(cand.max())
                if not np.any(np.isin(flk[i + 1:j], (1, 3))):
                    removed.append(dict(at_mm=round(1000 * seg[i], 1), detour_mm=round(1000 * (seg[j] - seg[i]), 1),
                                        pos=np.round(Pk[i], 4).tolist()))
                    i = j
                    continue
        i += 1
    Pc, psic = Pk[out_idx], psik[out_idx]
    n_after_loops = len(Pc)

    # 5. one smooth path: tube-constrained elastic band
    #    - reference = cleaned path, resampled every 1 mm (position arc length)
    #    - iterate: move each point toward the midpoint of its neighbours (normal component only, so
    #      spacing is kept) and then project it back so it never leaves a tube of --tube-mm around the
    #      reference. Result: as smooth as possible, never more than tube_mm from where the ring went.
    Sp = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(Pc, axis=0), axis=1))]
    s_ref, Ref = resample(Sp, np.column_stack([Pc, psic]), 0.001)
    Pref, psi_ref = Ref[:, :3], Ref[:, 3]
    B = Pref.copy()
    tube = a.tube_mm / 1000.0
    W = 60                                         # search window (points = mm) for nearest ref point
    idx = np.arange(len(B))
    offs = np.arange(-W, W + 1)
    for it in range(a.band_iters):
        mid = 0.5 * (B[:-2] + B[2:])
        t = B[2:] - B[:-2]
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        L = mid - B[1:-1]
        L -= np.sum(L * t, axis=1, keepdims=True) * t          # normal component only
        B[1:-1] += 0.5 * L
        # tube projection (nearest reference point within +-W of the point's own index)
        cand = np.clip(idx[:, None] + offs[None, :], 0, len(Pref) - 1)
        dv = B[:, None, :] - Pref[cand]
        dist = np.linalg.norm(dv, axis=2)
        k = np.argmin(dist, axis=1)
        dmin = dist[idx, k]
        near = Pref[cand[idx, k]]
        over = dmin > tube
        B[over] = near[over] + (B[over] - near[over]) * (tube / dmin[over])[:, None]
    # light final rounding of the kinks the band leaves between straight stretches, then re-check
    B = gauss_smooth(B, a.round_mm)
    cand = np.clip(idx[:, None] + offs[None, :], 0, len(Pref) - 1)
    dev = np.min(np.linalg.norm(B[:, None, :] - Pref[cand], axis=2), axis=1)
    # ring rotation: smooth the reference psi (it follows the reference index of each band point)
    psi_b = gauss_smooth(psi_ref[:, None], a.psi_smooth_mm)[:, 0]
    # final resample of the band by its own arc length
    Sb = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(B, axis=0), axis=1))]
    s_out, Y = resample(Sb, np.column_stack([B, psi_b]), a.ds_mm / 1000)
    Pp, psip = Y[:, :3], Y[:, 3]
    L_pos = float(s_out[-1])

    # 6. speed profile  (s = position arc length; v_ref = tip speed along the path)
    k = max(1, int(round(10.0 / a.ds_mm)))
    rad = np.full(len(Pp), np.inf)
    for i in range(k, len(Pp) - k):
        A_, B_, C_ = Pp[i - k], Pp[i], Pp[i + k]
        ab, bc, ca = np.linalg.norm(B_ - A_), np.linalg.norm(C_ - B_), np.linalg.norm(A_ - C_)
        cr = np.linalg.norm(np.cross(B_ - A_, C_ - A_))
        if cr > 1e-12:
            rad[i] = ab * bc * ca / (2 * cr)
    rc, rs = a.r_corner_mm / 1000, a.r_straight_mm / 1000
    frac = np.clip((rad - rc) / (rs - rc), 0, 1)
    v = a.v_corner + frac * (a.v_straight - a.v_corner)
    #   ring rotation: |dpsi/dt| = v * |dpsi/ds| <= w_max
    dpsi_ds = np.abs(np.gradient(psip, s_out))
    with np.errstate(divide="ignore"):
        v = np.minimum(v, np.where(dpsi_ds > 1e-9, a.w_max / dpsi_ds, np.inf))
    v = np.maximum(v, 0.002)
    #   look-ahead: already slow 1 cm before a bend
    win = max(1, int(round(10.0 / a.ds_mm)))
    v = np.array([v[max(0, i - win):i + win + 1].min() for i in range(len(v))])
    #   acceleration limit (forward + backward pass), start and end at rest
    ds_all = np.r_[np.diff(s_out), 0.0]
    v[0] = v[-1] = 0.0
    for i in range(1, len(v)):
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * a.a_max * ds_all[i - 1]))
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * a.a_max * ds_all[i]))
    vm = 0.5 * (v[1:] + v[:-1])
    T_est = float(np.sum(ds_all[:-1] / np.maximum(vm, 1e-4)))

    # orientation from psi
    Qp = np.array([R_to_quat(rodrigues(a_pt, p - psi[0]) @ R0) for p in psip])

    # 7. write
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        f.write("s,x,y,z,qx,qy,qz,qw,psi,v_ref,radius\n")
        for i in range(len(Pp)):
            f.write(f"{s_out[i]:.5f},{Pp[i,0]:.6f},{Pp[i,1]:.6f},{Pp[i,2]:.6f},"
                    f"{Qp[i,0]:.7f},{Qp[i,1]:.7f},{Qp[i,2]:.7f},{Qp[i,3]:.7f},{psip[i]:.6f},"
                    f"{v[i]:.5f},{min(rad[i], 9.99):.4f}\n")
    meta = dict(
        source=os.path.abspath(a.csv),
        start=dict(q_rad=q[0].tolist(), tcp=P[0].tolist(), quat=Q[0].tolist()),
        end=dict(q_rad=q[-1].tolist(), tcp=P[-1].tolist(), quat=Q[-1].tolist()),
        pointing_axis=a_pt.tolist(), pointing_spread_deg=round(point_spread, 2),
        samples=dict(raw_valid=n_raw, after_pauses=n_after_dwell, after_back_and_forth=n_after_loops,
                     output=len(Pp)),
        back_and_forth_removed=removed,
        smoothing=dict(method="tube-constrained elastic band", tube_mm=a.tube_mm,
                       max_dev_mm=round(1000 * float(dev.max()), 2),
                       p95_dev_mm=round(1000 * float(np.percentile(dev, 95)), 2)),
        path_length_cm=round(100 * L_pos, 1), ring_rotation_total_deg=round(math.degrees(np.ptp(psip)), 1),
        speed=dict(v_straight=a.v_straight, v_corner=a.v_corner, a_max=a.a_max, w_max=a.w_max,
                   est_duration_s=round(T_est, 1)),
    )
    with open(a.out.replace(".csv", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"valid samples {n_raw} -> after pauses {n_after_dwell} -> after back-and-forth {n_after_loops} "
          f"({len(removed)} detours cut) -> output {len(Pp)} points every {a.ds_mm} mm")
    print(f"path length {100*L_pos:.1f} cm, ring rotation range {math.degrees(np.ptp(psip)):.0f}°, "
          f"pointing axis spread {point_spread:.2f}°")
    print(f"smoothing (tube {a.tube_mm} mm): deviation from cleaned path max {1000*dev.max():.2f} mm, "
          f"95% {1000*np.percentile(dev,95):.2f} mm")
    print(f"speed {100*v.min():.1f}..{100*v.max():.1f} cm/s, estimated run time {T_est:.0f} s "
          f"(before any speed scaling)")
    print(f"wrote {a.out}")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # 2-D view in the plane spanned by the two largest path directions
        c = Pp.mean(0)
        _, _, Vt = np.linalg.svd(Pp - c, full_matrices=False)
        e1, e2 = Vt[0], Vt[1]
        if e2[2] < 0:
            e2 = -e2
        if e1 @ np.array([0, 1, 0]) < 0:
            e1 = -e1
        pr = lambda X: np.column_stack([(X - c) @ e1, (X - c) @ e2]) * 100
        fig = plt.figure(figsize=(17, 13))
        ax = fig.add_subplot(3, 1, 1)
        R_ = pr(P)
        ax.plot(R_[:, 0], R_[:, 1], color="0.75", lw=3, label="raw teleop recording")
        C_ = pr(Pref)
        ax.plot(C_[:, 0], C_[:, 1], color="tab:orange", lw=1, label="cleaned (pauses + back-and-forth removed)")
        S_ = pr(Pp)
        sc = ax.scatter(S_[:, 0], S_[:, 1], c=100 * v, s=6, cmap="RdYlGn", vmin=0,
                        vmax=100 * a.v_straight, label="smooth replay path (colour = speed)", zorder=3)
        for r in removed:
            rr = pr(np.array([r["pos"]]))[0]
            ax.plot(rr[0], rr[1], "kx", ms=8)
        ax.plot(*S_[0], "bo", ms=10, label="recorded start")
        ax.plot(*S_[-1], "rs", ms=10, label="recorded END = replay starts here (reversed)")
        ax.set_aspect("equal")
        ax.set_xlabel("along wire [cm]")
        ax.set_ylabel("up [cm]")
        ax.set_title(f"Wire path: {100*L_pos:.0f} cm smooth path, {len(removed)} back-and-forth detours cut (x), "
                     f"never more than {1000*dev.max():.1f} mm from the recording")
        ax.legend(loc="lower left", fontsize=8)
        plt.colorbar(sc, ax=ax, label="speed [cm/s]")
        # zooms: largest deviation, tightest corner
        tt = np.diff(Pp, axis=0)
        tt /= np.maximum(np.linalg.norm(tt, axis=1, keepdims=True), 1e-12)
        turn = np.degrees(np.arccos(np.clip(np.sum(tt[1:] * tt[:-1], axis=1), -1, 1)))
        Bp = pr(B)
        zooms = [(Bp[int(np.argmax(dev))], f"largest deviation ({1000*dev.max():.1f} mm)"),
                 (S_[int(np.argmax(turn)) + 1], f"tightest corner (radius ~{57.3/max(turn.max(),1e-6):.1f} mm)")]
        for zi, (cz, title) in enumerate(zooms):
            az = fig.add_subplot(3, 2, 3 + zi)
            az.plot(R_[:, 0], R_[:, 1], color="0.75", lw=5, label="raw recording")
            az.plot(C_[:, 0], C_[:, 1], color="tab:orange", lw=1.5, label="cleaned")
            az.plot(S_[:, 0], S_[:, 1], color="tab:blue", lw=1.5, marker=".", ms=3, label="smooth replay path")
            az.set_xlim(cz[0] - 2.5, cz[0] + 2.5)
            az.set_ylim(cz[1] - 2.5, cz[1] + 2.5)
            az.set_aspect("equal")
            az.grid(alpha=0.3)
            az.set_title("zoom: " + title)
            az.legend(fontsize=7)
        ax2 = fig.add_subplot(3, 2, 5)
        ax2.plot(100 * s_out, 100 * v)
        ax2.set_xlabel("path parameter from recorded start [cm]")
        ax2.set_ylabel("speed [cm/s]")
        ax2.set_title(f"speed profile (est. {T_est:.0f} s at 100%); replay runs it right-to-left")
        ax3 = fig.add_subplot(3, 2, 6)
        ax3.plot(100 * s_out, np.degrees(psip))
        ax3.set_xlabel("path parameter from recorded start [cm]")
        ax3.set_ylabel("ring rotation psi [deg]")
        ax3.set_title("ring rotation along the path")
        plt.tight_layout()
        plt.savefig(a.plot, dpi=110)
        print(f"plot -> {a.plot}")


if __name__ == "__main__":
    main()
