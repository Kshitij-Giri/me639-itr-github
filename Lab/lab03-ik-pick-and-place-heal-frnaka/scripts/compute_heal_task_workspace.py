#!/usr/bin/env python3
"""
HEAL Robot: Task Workspace Analysis
-----------------------------------
1. Loads the standalone HEAL robot model.
2. Samples valid joint configurations.
3. Computes the selected site's world position using forward kinematics.
4. Saves workspace points and the 3D plot in the results/ directory.

Current limitation:
The selected site is right_center, the robot's mounting site.
The actual gripper TCP will be identified in a later stage.
"""

from pathlib import Path
import csv

import mujoco
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
LAB_DIR = SCRIPT_DIR.parent

ROBOT_XML = (
    LAB_DIR
    / "robot-descriptions"
    / "single_arm_heal_effort_actuation_rs_mj.xml"
)

# All generated outputs go here, outside the scripts directory.
RESULTS_DIR = LAB_DIR / "results"

CSV_PATH = RESULTS_DIR / "heal_task_workspace_points.csv"
PLOT_PATH = RESULTS_DIR / "heal_task_workspace.png"

SITE_NAME = "right_center"

NUM_SAMPLES = 10000
RANDOM_SEED = 42


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """Load the HEAL model and validate the selected site."""

    if not ROBOT_XML.is_file():
        raise FileNotFoundError(
            f"Robot XML not found:\n{ROBOT_XML}"
        )

    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    data = mujoco.MjData(model)

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        SITE_NAME,
    )

    if site_id < 0:
        print("\nAvailable sites:")
        for sid in range(model.nsite):
            print(
                mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_SITE,
                    sid,
                )
            )

        raise ValueError(
            f"Site '{SITE_NAME}' was not found in the model."
        )

    return model, data, site_id


# ============================================================
# JOINT CONFIGURATION SAMPLING
# ============================================================

def print_joint_information(model):
    """Print joint names and limits."""

    print("\n--- JOINT INFORMATION ---")

    for jid in range(model.njnt):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            jid,
        )

        limited = bool(model.jnt_limited[jid])

        if limited:
            limits = model.jnt_range[jid].tolist()
        else:
            limits = "unlimited in XML; kept at default position"

        print(
            f"ID={jid}, name={name}, "
            f"limits={limits}"
        )


def sample_configuration(model, rng):
    """
    Sample one configuration.

    Limited hinge and slide joints are sampled uniformly within
    their XML limits. Other joints remain at their default values.
    """

    qpos = model.qpos0.copy()

    for jid in range(model.njnt):
        joint_type = model.jnt_type[jid]
        qpos_address = int(model.jnt_qposadr[jid])

        if not model.jnt_limited[jid]:
            continue

        lower, upper = model.jnt_range[jid]

        if joint_type in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            qpos[qpos_address] = rng.uniform(lower, upper)

    return qpos


# ============================================================
# COMPUTE TASK WORKSPACE
# ============================================================

def compute_workspace(model, data, site_id):
    """Compute sampled world-space positions using forward kinematics."""

    rng = np.random.default_rng(RANDOM_SEED)
    points = []

    print(f"\nSampling {NUM_SAMPLES} configurations...")

    for i in range(NUM_SAMPLES):
        data.qpos[:] = sample_configuration(model, rng)

        # Update body and site positions for the sampled configuration.
        mujoco.mj_forward(model, data)

        position = data.site_xpos[site_id].copy()

        if np.all(np.isfinite(position)):
            points.append(position)

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{NUM_SAMPLES}")

    if not points:
        raise RuntimeError("No valid workspace points were generated.")

    return np.asarray(points)


# ============================================================
# SAVE RESULTS
# ============================================================

def save_workspace_csv(points):
    """Save XYZ coordinates in metres."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x_m", "y_m", "z_m"])
        writer.writerows(points)

    print(f"\nWorkspace CSV saved to:\n{CSV_PATH}")


def plot_workspace(points):
    """Create and save the 3D workspace plot."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=2,
        alpha=0.35,
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(
        f"HEAL Task Workspace ({len(points)} samples)"
    )

    # Equal scale across axes to avoid visual distortion.
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float(np.max(maximum - minimum)) / 2.0, 0.01)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)

    print(f"Workspace plot saved to:\n{PLOT_PATH}")

    plt.show()
    plt.close(fig)


# ============================================================
# PRINT WORKSPACE BOUNDS
# ============================================================

def print_workspace_bounds(points):
    """Print the sampled coordinate bounds."""

    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    spans = maximum - minimum

    print("\n--- SAMPLED WORKSPACE BOUNDS ---")
    print("Axis       Minimum (m)    Maximum (m)    Span (m)")

    for i, axis in enumerate(("X", "Y", "Z")):
        print(
            f"{axis:<10} {minimum[i]:>10.4f}"
            f"      {maximum[i]:>10.4f}"
            f"      {spans[i]:>10.4f}"
        )

    print(
        "\nThese are sampled bounds for the selected mounting site, "
        "not exact robot limits or the final dexterous workspace."
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("HEAL Task Workspace Analysis")
    print("----------------------------")
    print(f"Robot XML: {ROBOT_XML}")
    print(f"Target site: {SITE_NAME}")
    print(f"Results directory: {RESULTS_DIR}")

    model, data, site_id = load_model()

    print(f"\nJoints: {model.njnt}")
    print(f"Position coordinates (nq): {model.nq}")
    print(f"Velocity coordinates (nv): {model.nv}")
    print(f"Actuators: {model.nu}")

    print_joint_information(model)

    points = compute_workspace(model, data, site_id)

    print(f"\nValid workspace points: {len(points)}")

    save_workspace_csv(points)
    print_workspace_bounds(points)
    plot_workspace(points)

    print("\nWorkspace analysis completed successfully.")


if __name__ == "__main__":
    main()