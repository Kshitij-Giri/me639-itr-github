#!/usr/bin/env python3
"""Sample and plot HEAL TCP positions reachable with the end-effector facing down.

Outputs CSV + 3D and tabletop-plane PNG plots in the lab results/ directory.
This is sampled Mink IK reachability; it does not perform collision checking.
"""
from pathlib import Path
import csv
import math
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import mujoco
import mink
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
LAB_DIR = SCRIPT_DIR.parent
DESC = LAB_DIR / "robot-descriptions"
RESULTS = LAB_DIR / "results"
ARM_XML = DESC / "single_arm_heal_effort_actuation_rs_mj.xml"
GRIPPER_XML = DESC / "robotiq_2f85_v4" / "2f85.xml"
SITE_NAME = "ft_site_eff"

# Workspace sampling volume (metres). Z is sampled more finely so the
# selected tabletop height is based on reachability across multiple heights.
X_VALUES = np.linspace(0.00, 0.90, 13)
Y_VALUES = np.linspace(-0.60, 0.60, 17)
Z_VALUES = np.arange(0.10, 0.901, 0.05)
YAW_DEG = 0.0
# Minimum horizontal clearance between the proposed tabletop rectangle and
# the HEAL base XY point. This is a conservative point-based guard, not a
# substitute for checking the full robot/base collision geometry.
BASE_CLEARANCE_M = 0.12
DT, MAX_IK_STEPS = 0.01, 120
SOLVER, DAMPING = "daqp", 1e-3
POS_TOL, ORI_TOL, GAIN = 0.015, 0.12, 0.2


def vec(text):
    return np.zeros(3) if text is None else np.fromstring(text, sep=" ")


def read_base_position():
    """Read the HEAL base position; table XML dimensions are deliberately ignored."""
    arm_root = ET.parse(ARM_XML).getroot()
    base = arm_root.find(".//worldbody/body[@name='base_link']")
    if base is None:
        raise ValueError("Could not find body 'base_link' in the HEAL XML.")
    return vec(base.get("pos"))


def load_model():
    for p in (ARM_XML, GRIPPER_XML):
        if not p.is_file():
            raise FileNotFoundError(f"Missing required file: {p}")
    arm = mujoco.MjSpec.from_file(str(ARM_XML))
    gripper = mujoco.MjSpec.from_file(str(GRIPPER_XML))
    mount = arm.site("right_center")
    if mount is None:
        raise ValueError("HEAL site 'right_center' was not found.")
    arm.attach(gripper, prefix="gripper/", site=mount)
    model = arm.compile()
    data = mujoco.MjData(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)
    if sid < 0:
        raise ValueError(f"Site '{SITE_NAME}' was not found after attaching gripper.")
    return model, data, sid


def estimate_tcp_offset(model, data, site_id):
    """Estimate TCP from pad geometry centres (provisional; validate physically)."""
    data.qpos[:] = model.qpos0.copy()
    mujoco.mj_forward(model, data)
    pad_names = {"gripper/left_pad", "gripper/right_pad"}
    pad_ids = set()
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if name in pad_names:
            pad_ids.add(bid)
    if len(pad_ids) != 2:
        raise RuntimeError(f"Could not identify both gripper pad bodies: {pad_ids}")
    centres = [data.geom_xpos[g].copy() for g in range(model.ngeom)
               if int(model.geom_bodyid[g]) in pad_ids]
    if not centres:
        raise RuntimeError("No geometries found on the gripper pad bodies.")
    tcp_world = np.mean(centres, axis=0)
    site_pos = data.site_xpos[site_id].copy()
    site_rot = data.site_xmat[site_id].reshape(3, 3).copy()
    offset = site_rot.T @ (tcp_world - site_pos)
    print("Provisional TCP offset in ft_site_eff frame:", np.round(offset, 4))
    print("WARNING: pad-centre midpoint is an estimate, not a validated contact TCP.")
    return offset


def downward_rotation(yaw_deg=0.0):
    """Set tool-local +Z to world -Z, with optional yaw about world Z."""
    yaw = math.radians(yaw_deg)
    rz = np.array([[math.cos(yaw), -math.sin(yaw), 0.0],
                   [math.sin(yaw),  math.cos(yaw), 0.0],
                   [0.0,            0.0,           1.0]])
    rx_pi = np.diag([1.0, -1.0, -1.0])
    return rz @ rx_pi


def solve_target(model, tcp_pos, tcp_rot, tcp_offset):
    cfg = mink.Configuration(model)
    cfg.update(model.qpos0.copy())
    task = mink.FrameTask(frame_name=SITE_NAME, frame_type="site",
                          position_cost=1.0, orientation_cost=1.0, gain=GAIN)
    site_pos = tcp_pos - tcp_rot @ tcp_offset
    target = np.eye(4)
    target[:3, :3], target[:3, 3] = tcp_rot, site_pos
    task.set_target(mink.SE3.from_matrix(target))
    pe, oe = float("inf"), float("inf")
    for _ in range(MAX_IK_STEPS):
        try:
            vel = mink.solve_ik(cfg, [task], DT, SOLVER, damping=DAMPING)
            cfg.integrate_inplace(vel, DT)
        except Exception:
            return False, pe, oe
        err = task.compute_error(cfg)
        pe, oe = float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:]))
        if pe <= POS_TOL and oe <= ORI_TOL:
            return True, pe, oe
    return False, pe, oe


def draw_table_plane(ax, x_min, x_max, y_min, y_max, z, color="saddlebrown", alpha=0.18):
    """Draw a proposed tabletop plane without relying on table.xml."""
    xx, yy = np.meshgrid([x_min, x_max], [y_min, y_max])
    zz = np.full_like(xx, z, dtype=float)
    ax.plot_surface(xx, yy, zz, color=color, alpha=alpha, shade=False)
    corners = [(x_min,y_min,z),(x_max,y_min,z),(x_max,y_max,z),
               (x_min,y_max,z),(x_min,y_min,z)]
    ax.plot([p[0] for p in corners], [p[1] for p in corners],
            [p[2] for p in corners], color=color, lw=2)


def largest_all_green_rectangle(reachable):
    """Largest axis-aligned rectangle of grid cells that are all reachable.

    Returns (x_start, x_end, y_start, y_end) as inclusive array indices,
    or None if no reachable cell exists. Uses the maximal-rectangle histogram
    algorithm; red/unreachable cells are never included inside the rectangle.
    """
    grid = np.asarray(reachable, dtype=bool)
    if grid.ndim != 2 or not grid.any():
        return None
    # Rows are Y values, columns are X values.
    heights = np.zeros(grid.shape[1], dtype=int)
    best_area, best = 0, None
    for yi, row in enumerate(grid):
        heights = np.where(row, heights + 1, 0)
        stack = []
        for xi in range(len(heights) + 1):
            h = heights[xi] if xi < len(heights) else 0
            start = xi
            while stack and stack[-1][1] > h:
                sx, sh = stack.pop()
                area = sh * (xi - sx)
                if area > best_area:
                    best_area = area
                    best = (sx, xi - 1, yi - sh + 1, yi)
                start = sx
            if not stack or stack[-1][1] < h:
                stack.append((start, h))
    return best


def rectangle_for_height(rows_at_z, x_values, y_values, base_pos):
    """Get the largest all-green rectangle, rejecting rectangles too near base."""
    grid = np.zeros((len(y_values), len(x_values)), dtype=bool)
    for r in rows_at_z:
        xi = int(np.argmin(np.abs(x_values - float(r["x_m"]))))
        yi = int(np.argmin(np.abs(y_values - float(r["y_m"]))))
        grid[yi, xi] = bool(int(r["reachable"]))

    # Exclude cells inside a small base-clearance disk before rectangle search.
    xx, yy = np.meshgrid(x_values, y_values)
    near_base = np.hypot(xx - base_pos[0], yy - base_pos[1]) < BASE_CLEARANCE_M
    grid[near_base] = False
    indices = largest_all_green_rectangle(grid)
    if indices is None:
        return None, grid
    x0, x1, y0, y1 = indices
    # Grid-point bounds are expanded by half a sample interval to estimate
    # tabletop edges, but all sampled points inside the rectangle are green.
    dx = float(np.mean(np.diff(x_values))) if len(x_values) > 1 else 0.0
    dy = float(np.mean(np.diff(y_values))) if len(y_values) > 1 else 0.0
    bounds = (float(x_values[x0] - dx/2), float(x_values[x1] + dx/2),
              float(y_values[y0] - dy/2), float(y_values[y1] + dy/2))
    bounds = (max(bounds[0], float(x_values.min())),
              min(bounds[1], float(x_values.max())),
              max(bounds[2], float(y_values.min())),
              min(bounds[3], float(y_values.max())))
    return (bounds, indices, grid), grid

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    base_pos = read_base_position()
    model, data, site_id = load_model()
    tcp_offset = estimate_tcp_offset(model, data, site_id)
    rot = downward_rotation(YAW_DEG)
    total = len(X_VALUES) * len(Y_VALUES) * len(Z_VALUES)
    print(f"HEAL base world position: {base_pos}")
    print(f"Testing {total} TCP targets at {len(Z_VALUES)} heights; table.xml dimensions are ignored.")
    print(f"Tool local +Z points world -Z; yaw={YAW_DEG} deg")
    print("Note: sampled IK estimate only; collisions are not checked.")

    rows, points = [], []
    count = 0
    for z in Z_VALUES:
        for y in Y_VALUES:
            for x in X_VALUES:
                target = np.array([x, y, z], dtype=float)
                ok, pe, oe = solve_target(model, target, rot, tcp_offset)
                rows.append({"x_m":float(x), "y_m":float(y), "z_m":float(z),
                             "yaw_deg":YAW_DEG, "reachable":int(ok),
                             "position_error_m":pe, "orientation_error_rad":oe})
                if ok:
                    points.append(target)
                count += 1
                if count % 100 == 0 or count == total:
                    print(f"Tested {count}/{total}")

    csv_path = RESULTS / "heal_dexterous_workspace.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f"CSV saved: {csv_path}")
    pts = np.asarray(points, dtype=float).reshape((-1, 3))

    # At every tested height, find the largest rectangle containing only
    # green/reachable sampled points, excluding the base-clearance region.
    height_results = []
    for z in Z_VALUES:
        at_z = [r for r in rows if abs(float(r["z_m"]) - z) < 1e-8]
        result, grid = rectangle_for_height(at_z, X_VALUES, Y_VALUES, base_pos)
        if result is None:
            height_results.append({"z":float(z), "area":0.0, "bounds":None, "result":None})
            continue
        bounds, indices, _ = result
        xmin, xmax, ymin, ymax = bounds
        area = max(0.0, xmax-xmin) * max(0.0, ymax-ymin)
        height_results.append({"z":float(z), "area":area, "bounds":bounds,
                              "indices":indices, "result":result, "grid":grid})

    # Primary criterion: largest all-green rectangular area. Tie-breaker: the
    # highest height, so we prefer a higher table when area is equivalent.
    valid_heights = [h for h in height_results if h["area"] > 0 and h["result"] is not None]
    if not valid_heights:
        raise RuntimeError("No all-green rectangle found. Increase workspace sampling or inspect IK convergence.")
    best = max(valid_heights, key=lambda h: (h["area"], -h["z"]))
    ideal_z = best["z"]
    xmin, xmax, ymin, ymax = best["bounds"]
    ideal_width_x, ideal_width_y = xmax-xmin, ymax-ymin
    best_at_z = [r for r in rows if abs(float(r["z_m"]) - ideal_z) < 1e-8]
    reachable_at_z = sum(int(r["reachable"]) for r in best_at_z)
    total_at_z = len(best_at_z)
    print("\n=== OPTIMIZED TABLE CANDIDATE (sampled IK) ===")
    print(f"Ideal tabletop height Z: {ideal_z:.3f} m")
    print(f"X limits: [{xmin:.3f}, {xmax:.3f}] m; X dimension: {ideal_width_x:.3f} m")
    print(f"Y limits: [{ymin:.3f}, {ymax:.3f}] m; Y dimension: {ideal_width_y:.3f} m")
    print(f"All-green rectangle area: {best['area']:.4f} m^2")
    print(f"Reachable targets at selected height: {reachable_at_z}/{total_at_z} ({100*reachable_at_z/max(1,total_at_z):.1f}%)")
    print(f"Base point clearance threshold applied in XY: {BASE_CLEARANCE_M:.2f} m")
    print("WARNING: rectangle guarantees only that sampled grid points inside it were green; it is not a collision-validated continuous workspace.")

    # Figure 1: 3D workspace and selected ideal tabletop plane/rectangle.
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    if len(pts):
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=9, alpha=0.42, label="IK-reachable TCP targets")
    draw_table_plane(ax, xmin, xmax, ymin, ymax, ideal_z)
    ax.scatter([base_pos[0]], [base_pos[1]], [base_pos[2]], marker="*", s=220, label="HEAL base")
    ax.text(*base_pos, "  HEAL base", fontsize=9)
    ax.set_xlabel("World X (m)"); ax.set_ylabel("World Y (m)"); ax.set_zlabel("World Z (m)")
    ax.set_title(f"HEAL Downward-Facing Workspace + Optimized Table Plane (Z={ideal_z:.2f} m)")
    ax.set_xlim(X_VALUES.min(), X_VALUES.max()); ax.set_ylim(Y_VALUES.min(), Y_VALUES.max())
    ax.set_zlim(0, Z_VALUES.max()); ax.set_box_aspect((1,1,0.85)); ax.legend()
    plt.tight_layout()
    p3d = RESULTS / "heal_dexterous_workspace_3d.png"
    fig.savefig(p3d, dpi=220)

    # Figure 2: height profile, using largest all-green rectangle area as score.
    fig_h, ax_h = plt.subplots(figsize=(9, 5.5))
    hz = [h["z"] for h in height_results]
    ha = [h["area"] for h in height_results]
    ax_h.plot(hz, ha, marker="o", lw=1.8, label="Largest all-green rectangle area")
    ax_h.axvline(ideal_z, color="tab:green", linestyle="--",
                 label=f"Selected height = {ideal_z:.2f} m")
    ax_h.set_xlabel("Candidate tabletop height Z (m)")
    ax_h.set_ylabel("Largest all-green rectangle area (m²)")
    ax_h.set_title("Table Height Selection: Reachable Rectangular Area vs Height")
    ax_h.grid(True, alpha=0.3); ax_h.legend(); plt.tight_layout()
    ph = RESULTS / "heal_dexterous_workspace_height_optimization.png"
    fig_h.savefig(ph, dpi=220)

    # Figure 3: 2D plane at the selected height, with only-green optimum rectangle.
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    selected_rows = best_at_z
    for r in selected_rows:
        ax2.scatter(r["x_m"], r["y_m"],
                    color=("tab:green" if r["reachable"] else "tab:red"), s=28)
    rect = plt.Rectangle((xmin, ymin), ideal_width_x, ideal_width_y, fill=False,
                         lw=2.5, edgecolor="tab:blue", label="Largest all-green rectangle")
    ax2.add_patch(rect)
    ax2.scatter(base_pos[0], base_pos[1], marker="*", s=220, label="HEAL base XY")
    from matplotlib.lines import Line2D
    ax2.legend(handles=[Line2D([0],[0],marker="o",color="w",markerfacecolor="tab:green",markersize=8,label="IK reachable"),
                        Line2D([0],[0],marker="o",color="w",markerfacecolor="tab:red",markersize=8,label="IK not reached"),
                        rect, Line2D([0],[0],marker="*",color="w",markerfacecolor="tab:blue",markersize=12,label="HEAL base XY")])
    ax2.set_aspect("equal", adjustable="box"); ax2.set_xlabel("World X (m)"); ax2.set_ylabel("World Y (m)")
    ax2.set_title(f"Reachability at Optimized Height Z={ideal_z:.3f} m\nLargest all-green rectangle: {ideal_width_x:.2f} × {ideal_width_y:.2f} m")
    ax2.grid(True, alpha=0.25); plt.tight_layout()
    p2d = RESULTS / "heal_dexterous_workspace_table_plane.png"
    fig2.savefig(p2d, dpi=220)

    # Save a concise optimization summary for reproducibility.
    summary_path = RESULTS / "heal_dexterous_workspace_table_optimization.txt"
    summary_path.write_text(
        f"Ideal tabletop height Z: {ideal_z:.4f} m\n"
        f"X limits: [{xmin:.4f}, {xmax:.4f}] m\n"
        f"Y limits: [{ymin:.4f}, {ymax:.4f}] m\n"
        f"X dimension: {ideal_width_x:.4f} m\nY dimension: {ideal_width_y:.4f} m\n"
        f"Rectangle area: {best['area']:.6f} m^2\n"
        f"Reachable sampled targets at selected height: {reachable_at_z}/{total_at_z}\n"
        f"Base point clearance threshold: {BASE_CLEARANCE_M:.3f} m\n"
        "Caution: sampled IK only; no robot/object collision checking.\n"
    )
    print(f"3D plot saved: {p3d}")
    print(f"Height-optimization plot saved: {ph}")
    print(f"Selected-height 2D plot saved: {p2d}")
    print(f"Optimization summary saved: {summary_path}")
    print(f"Overall reachable sampled targets: {len(points)}/{len(rows)} ({100*len(points)/max(1,len(rows)):.1f}%)")
    plt.show()


if __name__ == "__main__":
    main()