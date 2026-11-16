#!/usr/bin/env python3
"""
stitch_recordings.py — join several teleop recordings (taken in order along the wire) into one CSV.

Usage:
    python3 stitch_recordings.py recordings/wire_path_A.csv recordings/wire_path_B.csv [more...] \
            -o recordings/wire_path_stitched.csv

What it does
  * keeps the files in the order given (= order along the wire)
  * time is made continuous (each part continues 0.02 s after the previous one ends)
  * checks the JOIN between parts: tool-tip jump and orientation jump between the last sample of one part
    and the first sample of the next. Prints them; warns if > --max-gap-mm / --max-gap-deg.
    (After a restart + K the arm should not have moved, so the gap is normally < 1 mm.)
  * marks the first sample of every later part with flag 3 ("join") so the IK cleaner keeps it as a node
  * recomputes plane_u / plane_v / plane_d from the FIRST part's plane (K may define a slightly
    different plane after a restart; the base-frame tcp_* columns are what the IK uses anyway)
  * DROPS INVALID ROWS: samples where every joint is exactly 0 (logged after a robot fault, when
    /joint_states reports zeros) and anything after a tool-tip jump > --max-step-mm (default 5 mm,
    impossible at 50 Hz teleop speeds). Reported per part.
  * writes <out>_meta.json describing the parts
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    hdr, data = rows[0], rows[1:]
    meta_path = path.replace(".csv", "_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    return hdr, data, meta


def quat_angle_deg(q1, q2):
    d = abs(float(np.dot(q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2))))
    return math.degrees(2 * math.acos(min(1.0, d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--max-gap-mm", type=float, default=2.0)
    ap.add_argument("--max-gap-deg", type=float, default=2.0)
    ap.add_argument("--max-step-mm", type=float, default=5.0)
    a = ap.parse_args()

    hdr0, _, meta0 = load(a.parts[0])
    col = {n: i for i, n in enumerate(hdr0)}
    need = ["t", "flag", "tcp_x", "tcp_y", "tcp_z", "tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw"]
    if any(n not in col for n in need):
        sys.exit(f"{a.parts[0]}: missing columns {[n for n in need if n not in col]}")
    yaw = math.radians(meta0.get("wire_yaw_deg", 0.0))
    u = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    v = np.array([0.0, 0.0, 1.0])
    n = np.cross(u, v)

    out_rows, t_off, prev_last, report = [], 0.0, None, []
    for k, path in enumerate(a.parts):
        hdr, data, _ = load(path)
        if hdr != hdr0:
            sys.exit(f"{path}: columns differ from {a.parts[0]} (recorded with a different teleop version?)")
        # ---- drop invalid rows: all-zero joints, and everything after an impossible jump
        qcols = [col[c] for c in hdr0 if c.startswith("q_")]
        n_in = len(data)
        kept, prev_p, cut_reason = [], None, ""
        for r in data:
            qv = [float(r[i]) for i in qcols]
            if all(abs(x) < 1e-9 for x in qv):
                cut_reason = "all-zero joints (robot fault)"
                break
            pp = np.array([float(r[col[c]]) for c in ("tcp_x", "tcp_y", "tcp_z")])
            if prev_p is not None and 1000 * np.linalg.norm(pp - prev_p) > a.max_step_mm:
                cut_reason = f"tool-tip jump > {a.max_step_mm} mm"
                break
            kept.append(r)
            prev_p = pp
        data = kept
        if len(data) < n_in:
            print(f"{os.path.basename(path)}: dropped last {n_in - len(data)} of {n_in} rows ({cut_reason})")
        if not data:
            print(f"skip empty part {path}")
            continue
        t0 = float(data[0][col["t"]])
        first = data[0]
        if prev_last is not None:
            p1 = np.array([float(prev_last[col[c]]) for c in ("tcp_x", "tcp_y", "tcp_z")])
            p2 = np.array([float(first[col[c]]) for c in ("tcp_x", "tcp_y", "tcp_z")])
            q1 = np.array([float(prev_last[col[c]]) for c in ("tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw")])
            q2 = np.array([float(first[col[c]]) for c in ("tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw")])
            gap_mm = 1000 * float(np.linalg.norm(p2 - p1))
            gap_deg = quat_angle_deg(q1, q2)
            ok = gap_mm <= a.max_gap_mm and gap_deg <= a.max_gap_deg
            report.append(dict(join=k, gap_mm=round(gap_mm, 2), gap_deg=round(gap_deg, 2), ok=ok))
            print(f"join {k}: {os.path.basename(a.parts[k-1])} -> {os.path.basename(path)}: "
                  f"tip jump {gap_mm:.2f} mm, orientation jump {gap_deg:.2f}°  {'OK' if ok else 'WARNING: large gap'}")
        for i, r in enumerate(data):
            r = list(r)
            r[col["t"]] = f"{t_off + float(r[col['t']]) - t0:.4f}"
            if k > 0 and i == 0:
                r[col["flag"]] = "3"
            p = np.array([float(r[col[c]]) for c in ("tcp_x", "tcp_y", "tcp_z")])
            for name, axis in (("plane_u", u), ("plane_v", v), ("plane_d", n)):
                if name in col:
                    r[col[name]] = f"{float(p @ axis):.6f}"
            out_rows.append(r)
        t_off = float(out_rows[-1][col["t"]]) + 0.02
        prev_last = data[-1]

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr0)
        w.writerows(out_rows)
    meta = dict(meta0)
    meta.update(stitched_from=a.parts, joins=report,
                flags={"0": "sample", "1": "keyframe", "2": "contact", "3": "join (start of a later part)"})
    with open(a.out.replace(".csv", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    P = np.array([[float(r[col[c]]) for c in ("tcp_x", "tcp_y", "tcp_z")] for r in out_rows])
    L = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))
    print(f"wrote {a.out}: {len(out_rows)} samples from {len(a.parts)} parts, path length {100*L:.1f} cm")
    if any(not j["ok"] for j in report):
        print("WARNING: at least one join has a large gap - check the plot before running IK.")


if __name__ == "__main__":
    main()
