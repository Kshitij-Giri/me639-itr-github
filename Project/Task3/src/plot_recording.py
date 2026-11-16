#!/usr/bin/env python3
"""Quick check of a recorded wire path:  python3 plot_recording.py recordings/wire_path_XXXX.csv
Shows the TCP path in 3D and projected into the wire plane (u, v), with keyframes/contacts marked,
plus per-joint angles over time (so you can see if the base joint wandered)."""
import json
import math
import sys

import numpy as np
import matplotlib.pyplot as plt

path = sys.argv[1]
data = np.genfromtxt(path, delimiter=",", names=True)
meta = json.load(open(path.replace(".csv", "_meta.json")))
yaw = math.radians(meta["wire_yaw_deg"])
u = np.array([math.cos(yaw), math.sin(yaw), 0.0]); v = np.array([0, 0, 1.0]); n = np.cross(u, v)
P = np.column_stack([data["tcp_x"], data["tcp_y"], data["tcp_z"]])
P0 = P - P[0]
pu, pv, pn = P0 @ u, P0 @ v, P0 @ n
flag = data["flag"].astype(int)
qcols = [c for c in data.dtype.names if c.startswith("q_")]

step = np.linalg.norm(np.diff(P, axis=0), axis=1)
print(f"{len(P)} samples, {data['t'][-1]:.1f} s, path length {step.sum()*100:.1f} cm, "
      f"out-of-plane range {np.ptp(pn)*1000:.1f} mm, keyframes {np.sum(flag==1)}, contacts {np.sum(flag==2)}")

fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(221, projection="3d")
ax.plot(P[:, 0], P[:, 1], P[:, 2]); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title("TCP path (base frame)")
ax2 = fig.add_subplot(222)
ax2.plot(pu * 100, pv * 100, ".-", ms=2)
ax2.plot(pu[flag == 1] * 100, pv[flag == 1] * 100, "go", label="keyframe")
ax2.plot(pu[flag == 2] * 100, pv[flag == 2] * 100, "rx", ms=10, label="contact")
ax2.set_aspect("equal"); ax2.set_xlabel("u [cm]"); ax2.set_ylabel("v [cm]"); ax2.legend()
ax2.set_title("Path in wire plane")
ax3 = fig.add_subplot(223)
ax3.plot(data["t"], pn * 1000); ax3.set_xlabel("t [s]"); ax3.set_ylabel("n [mm]")
ax3.set_title("Out-of-plane (depth) deviation")
ax4 = fig.add_subplot(224)
for c in qcols:
    ax4.plot(data["t"], np.degrees(data[c] - data[c][0]), label=c[2:])
ax4.set_xlabel("t [s]"); ax4.set_ylabel("Δq [deg]"); ax4.legend(); ax4.set_title("Joint motion")
plt.tight_layout(); plt.show()
