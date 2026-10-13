
#!/usr/bin/env python3
"""
HEAL Vertical-Grasp Dexterous Workspace
=======================================

Purpose
-------
Estimate the region where the HEAL robot can approach a cube from above
with the Robotiq 2F-85 gripper pointing vertically downward.

For every target TCP position and each tested yaw angle, the script:
1. Starts from the default robot configuration.
2. Solves IK for a safe approach position above the target.
3. Moves the target downward in small sequential steps.
4. Uses the previous IK solution as the starting configuration for
   the next step.
5. Marks the target successful only if the approach and all descent
   steps converge within the specified tolerances.

IMPORTANT
---------
- The TCP is estimated from the gripper pad geometry. Validate it
  before using the results for physical grasping.
- This is a kinematic test. Table, cube, tray, and self-collisions
  are not checked here.
- Successful IK steps do not guarantee a collision-free physical path.
- The sampled grid is task-specific; it is not a proof of the complete
  classical dexterous workspace.
"""

from pathlib import Path
import csv

import mujoco
import numpy as np
import matplotlib.pyplot as plt
import mink


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
LAB_DIR = SCRIPT_DIR.parent
RESULTS_DIR = LAB_DIR / "results"

ARM_XML = (
    LAB_DIR
    / "robot-descriptions"
    / "single_arm_heal_effort_actuation_rs_mj.xml"
)

GRIPPER_XML = (
    LAB_DIR
    / "robot-descriptions"
    / "robotiq_2f85_v4"
    / "2f85.xml"
)

SITE_NAME = "ft_site_eff"

CSV_PATH = RESULTS_DIR / "heal_dexterous_workspace_results.csv"
PLOT_PATH = RESULTS_DIR / "heal_dexterous_workspace.png"


# ============================================================
# WORKSPACE SETTINGS
# ============================================================

# These positions represent the desired TCP grasp positions in metres.
# Keep this initial grid for comparison with the existing results.
X_VALUES = np.linspace(0.10, 0.60, 5)
Y_VALUES = np.linspace(-0.30, 0.30, 5)
Z_VALUES = np.linspace(0.20, 0.70, 4)

# The gripper remains vertical. Yaw rotates it around the world Z axis.
YAW_ANGLES_DEG = [0, 90, 180, 270]

# Approach height above the target TCP grasp position.
APPROACH_HEIGHT = 0.10  # metres

# Number of intermediate target points during vertical descent.
# Five intervals means six descent targets, including the final target.
DESCENT_INTERVALS = 5

# Mink IK settings.
DT = 0.01
MAX_IK_STEPS = 200
SOLVER = "daqp"
DAMPING = 1e-3

# Acceptance tolerances.
POSITION_TOLERANCE = 0.01       # metres
ORIENTATION_TOLERANCE = 0.10    # radians


# ============================================================
# BUILD COMBINED HEAL + GRIPPER MODEL
# ============================================================

def load_combined_model():
    for path in (ARM_XML, GRIPPER_XML):
        if not path.is_file():
            raise FileNotFoundError(f"Missing XML file: {path}")

    arm = mujoco.MjSpec.from_file(str(ARM_XML))
    gripper = mujoco.MjSpec.from_file(str(GRIPPER_XML))

    mount = arm.site("right_center")
    if mount is None:
        raise ValueError("Arm site 'right_center' was not found.")

    arm.attach(gripper, prefix="gripper/", site=mount)

    model = arm.compile()
    data = mujoco.MjData(model)

    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME
    )

    if site_id < 0:
        raise ValueError(f"Site '{SITE_NAME}' was not found.")

    return model, data, site_id


# ============================================================
# ESTIMATE TCP FROM GRIPPER PAD GEOMETRY
# ============================================================

def estimate_tcp_offset(model, data, site_id):
    """
    Estimate a virtual TCP from the centres of the pad geometries.

    The offset is expressed in the site coordinate frame. This is a
    provisional estimate, not a verified physical contact point.
    """

    data.qpos[:] = model.qpos0.copy()
    mujoco.mj_forward(model, data)

    pad_body_names = {
        "gripper/left_pad",
        "gripper/right_pad",
    }

    pad_body_ids = set()

    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, body_id
        )
        if name in pad_body_names:
            pad_body_ids.add(body_id)

    if len(pad_body_ids) != 2:
        raise RuntimeError(
            "Could not identify both gripper pad bodies. "
            f"Found {len(pad_body_ids)}."
        )

    pad_centres = []

    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])

        if body_id in pad_body_ids:
            pad_centres.append(data.geom_xpos[geom_id].copy())

    if not pad_centres:
        raise RuntimeError("No pad geometries found.")

    tcp_world = np.mean(pad_centres, axis=0)
    site_world = data.site_xpos[site_id].copy()
    site_rotation = data.site_xmat[site_id].reshape(3, 3).copy()

    tcp_offset_local = site_rotation.T @ (tcp_world - site_world)

    print("\n--- PROVISIONAL TCP ---")
    print("Pad geometry centres:", len(pad_centres))
    print("TCP estimate (world):", tcp_world)
    print("Site position (world):", site_world)
    print("TCP offset (site frame):", tcp_offset_local)
    print(
        "WARNING: The pad-centre midpoint is an estimate. "
        "Validate it before physical grasping."
    )

    return tcp_offset_local, site_rotation


# ============================================================
# BUILD A TARGET SITE TRANSFORM FROM A TCP TARGET
# ============================================================

def target_site_transform(tcp_position, tcp_rotation, tcp_offset_local):
    """
    Convert a desired TCP pose into the corresponding site pose.

    The TCP orientation is assumed to match the site orientation.
    """

    target_site_position = (
        tcp_position - tcp_rotation @ tcp_offset_local
    )

    transform = np.eye(4)
    transform[:3, :3] = tcp_rotation
    transform[:3, 3] = target_site_position

    return transform


# ============================================================
# CREATE A MINK FRAME TASK
# ============================================================

def make_frame_task(site_name):
    return mink.FrameTask(
        frame_name=site_name,
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        gain=0.2,
    )


# ============================================================
# SOLVE ONE TARGET WHILE PRESERVING CONFIGURATION
# ============================================================

def move_ik_to_target(configuration, task, target_transform):
    """
    Iteratively solve one target from the current configuration.

    The configuration is modified in place. Therefore, a successful
    approach configuration can be used for the next descent step.
    """

    task.set_target(mink.SE3.from_matrix(target_transform))

    position_error = float("inf")
    orientation_error = float("inf")

    for _ in range(MAX_IK_STEPS):
        try:
            velocity = mink.solve_ik(
                configuration,
                [task],
                DT,
                SOLVER,
                damping=DAMPING,
            )
            configuration.integrate_inplace(velocity, DT)

        except Exception as exc:
            return False, position_error, orientation_error, str(exc)

        error = task.compute_error(configuration)

        position_error = float(np.linalg.norm(error[:3]))
        orientation_error = float(np.linalg.norm(error[3:]))

        if (
            position_error <= POSITION_TOLERANCE
            and orientation_error <= ORIENTATION_TOLERANCE
        ):
            return True, position_error, orientation_error, ""

    return False, position_error, orientation_error, ""


# ============================================================
# TEST APPROACH AND CONTINUOUS VERTICAL DESCENT
# ============================================================

def test_vertical_grasp(
    model,
    initial_qpos,
    tcp_position,
    tcp_rotation,
    tcp_offset_local,
):
    """
    Test a vertical approach followed by sequential downward IK steps.

    Returns a result dictionary for one target position and orientation.
    """

    configuration = mink.Configuration(model)
    configuration.update(initial_qpos.copy())

    task = make_frame_task(SITE_NAME)

    # --------------------------------------------------------
    # 1. Approach pose above the target
    # --------------------------------------------------------

    approach_tcp_position = tcp_position + np.array(
        [0.0, 0.0, APPROACH_HEIGHT]
    )

    approach_transform = target_site_transform(
        approach_tcp_position,
        tcp_rotation,
        tcp_offset_local,
    )

    approach_ok, approach_pos_err, approach_ori_err, solver_error = (
        move_ik_to_target(
            configuration,
            task,
            approach_transform,
        )
    )

    if not approach_ok:
        return {
            "approach_success": 0,
            "descent_success": 0,
            "success": 0,
            "approach_position_error_m": approach_pos_err,
            "approach_orientation_error_rad": approach_ori_err,
            "grasp_position_error_m": float("inf"),
            "grasp_orientation_error_rad": float("inf"),
            "max_descent_position_error_m": float("inf"),
            "max_descent_orientation_error_rad": float("inf"),
            "solver_error": solver_error,
        }

    # --------------------------------------------------------
    # 2. Descend vertically in sequential steps
    # --------------------------------------------------------

    max_position_error = 0.0
    max_orientation_error = 0.0

    descent_ok = True
    final_position_error = float("inf")
    final_orientation_error = float("inf")
    descent_solver_error = ""

    for step in range(1, DESCENT_INTERVALS + 1):
        fraction = step / DESCENT_INTERVALS

        step_tcp_position = (
            approach_tcp_position
            + fraction * (tcp_position - approach_tcp_position)
        )

        step_transform = target_site_transform(
            step_tcp_position,
            tcp_rotation,
            tcp_offset_local,
        )

        ok, pos_err, ori_err, error_message = move_ik_to_target(
            configuration,
            task,
            step_transform,
        )

        max_position_error = max(max_position_error, pos_err)
        max_orientation_error = max(max_orientation_error, ori_err)

        final_position_error = pos_err
        final_orientation_error = ori_err

        if not ok:
            descent_ok = False
            descent_solver_error = error_message
            break

    return {
        "approach_success": 1,
        "descent_success": int(descent_ok),
        "success": int(descent_ok),
        "approach_position_error_m": approach_pos_err,
        "approach_orientation_error_rad": approach_ori_err,
        "grasp_position_error_m": final_position_error,
        "grasp_orientation_error_rad": final_orientation_error,
        "max_descent_position_error_m": max_position_error,
        "max_descent_orientation_error_rad": max_orientation_error,
        "solver_error": descent_solver_error,
    }


# ============================================================
# COMPUTE THE VERTICAL-GRASP WORKSPACE
# ============================================================

def compute_workspace(model, data, site_id, tcp_offset_local):
    initial_qpos = model.qpos0.copy()

    data.qpos[:] = initial_qpos
    mujoco.mj_forward(model, data)

    # Reference orientation in the model's default configuration.
    base_rotation = data.site_xmat[site_id].reshape(3, 3).copy()

    rows = []

    total_positions = (
        len(X_VALUES) * len(Y_VALUES) * len(Z_VALUES)
    )
    total_tests = total_positions * len(YAW_ANGLES_DEG)

    print("\nHEAL Vertical-Grasp Dexterous Workspace")
    print("---------------------------------------")
    print("Positions:", total_positions)
    print("Yaw orientations per position:", len(YAW_ANGLES_DEG))
    print("Total vertical-grasp tests:", total_tests)
    print("Approach height:", APPROACH_HEIGHT, "m")
    print("Descent intervals:", DESCENT_INTERVALS)
    print("\nTesting target positions...")

    count = 0

    for x in X_VALUES:
        for y in Y_VALUES:
            for z in Z_VALUES:
                tcp_position = np.array([x, y, z], dtype=float)
                successful_yaws = 0
                successful_approaches = 0

                for yaw_deg in YAW_ANGLES_DEG:
                    yaw = np.deg2rad(yaw_deg)

                    rotation_z = np.array([
                        [np.cos(yaw), -np.sin(yaw), 0.0],
                        [np.sin(yaw),  np.cos(yaw), 0.0],
                        [0.0,           0.0,        1.0],
                    ])

                    # Preserve the downward-facing reference orientation.
                    # Only rotate around the world vertical axis.
                    tcp_rotation = rotation_z @ base_rotation

                    result = test_vertical_grasp(
                        model,
                        initial_qpos,
                        tcp_position,
                        tcp_rotation,
                        tcp_offset_local,
                    )

                    successful_approaches += result["approach_success"]
                    successful_yaws += result["success"]
                    count += 1

                    rows.append({
                        "x_m": float(x),
                        "y_m": float(y),
                        "z_m": float(z),
                        "yaw_deg": yaw_deg,
                        **result,
                    })

                    if count % 20 == 0:
                        print(f"Processed {count}/{total_tests}")

                # One summary row per target position.
                rows.append({
                    "x_m": float(x),
                    "y_m": float(y),
                    "z_m": float(z),
                    "yaw_deg": "ALL_ORIENTATIONS",
                    "approach_success": successful_approaches,
                    "descent_success": "",
                    "success": successful_yaws,
                    "approach_position_error_m": "",
                    "approach_orientation_error_rad": "",
                    "grasp_position_error_m": "",
                    "grasp_orientation_error_rad": "",
                    "max_descent_position_error_m": "",
                    "max_descent_orientation_error_rad": "",
                    "solver_error": "",
                })

    return rows


# ============================================================
# SAVE CSV RESULTS
# ============================================================

def save_results(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    columns = [
        "x_m",
        "y_m",
        "z_m",
        "yaw_deg",
        "approach_success",
        "descent_success",
        "success",
        "approach_position_error_m",
        "approach_orientation_error_rad",
        "grasp_position_error_m",
        "grasp_orientation_error_rad",
        "max_descent_position_error_m",
        "max_descent_orientation_error_rad",
        "solver_error",
    ]

    with CSV_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV saved to:\n{CSV_PATH}")


# ============================================================
# PLOT VERTICAL-GRASP WORKSPACE
# ============================================================

def plot_results(rows):
    pose_rows = [
        row for row in rows
        if row["yaw_deg"] != "ALL_ORIENTATIONS"
    ]

    positions = {}

    for row in pose_rows:
        key = (row["x_m"], row["y_m"], row["z_m"])
        positions.setdefault(key, []).append(int(row["success"]))

    xyz = []
    coverage = []

    for position, outcomes in positions.items():
        xyz.append(position)
        coverage.append(sum(outcomes) / len(outcomes))

    xyz = np.asarray(xyz, dtype=float)
    coverage = np.asarray(coverage, dtype=float)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=coverage,
        vmin=0.0,
        vmax=1.0,
        s=55,
    )

    fig.colorbar(
        scatter,
        ax=ax,
        label="Fraction of yaw tests passing vertical grasp",
        shrink=0.7,
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("TCP grasp height Z (m)")
    ax.set_title("HEAL Vertical-Grasp Dexterous Workspace")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)

    print(f"Plot saved to:\n{PLOT_PATH}")

    plt.show()
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading combined HEAL + Robotiq model...")

    model, data, site_id = load_combined_model()

    print("Model loaded.")
    print("Joints:", model.njnt)
    print("nq:", model.nq)
    print("nv:", model.nv)
    print("Actuators:", model.nu)

    tcp_offset_local, _ = estimate_tcp_offset(
        model, data, site_id
    )

    rows = compute_workspace(
        model, data, site_id, tcp_offset_local
    )

    save_results(rows)
    plot_results(rows)

    pose_rows = [
        row for row in rows
        if row["yaw_deg"] != "ALL_ORIENTATIONS"
    ]

    total = len(pose_rows)
    approach_successes = sum(
        int(row["approach_success"]) for row in pose_rows
    )
    grasp_successes = sum(
        int(row["success"]) for row in pose_rows
    )

    position_summaries = [
        row for row in rows
        if row["yaw_deg"] == "ALL_ORIENTATIONS"
    ]

    all_yaw_positions = sum(
        int(row["success"]) == len(YAW_ANGLES_DEG)
        for row in position_summaries
    )

    print("\n--- SUMMARY ---")
    print("Target poses tested:", total)
    print(
        "Successful approach poses:",
        f"{approach_successes}/{total}",
        f"({100 * approach_successes / total:.2f}%)",
    )
    print(
        "Successful vertical grasp/descent tests:",
        f"{grasp_successes}/{total}",
        f"({100 * grasp_successes / total:.2f}%)",
    )
    print(
        "Positions passing every tested yaw:",
        f"{all_yaw_positions}/{len(position_summaries)}",
    )
    print(
        "\nInterpretation: these are sampled kinematic results. "
        "Collision checking and physical TCP validation are still needed."
    )


if __name__ == "__main__":
    main()
