#!/usr/bin/env python3
"""Franka Panda: sampled task-workspace analysis in MuJoCo.

This script loads the standalone Franka Panda model, samples its seven arm
joints within their MuJoCo joint limits, computes the world position of the
end-effector site using forward kinematics, and saves a CSV and 3D plot.

The result is a sampled position workspace, not an exact boundary and not a
collision-free/dexterous workspace. The fingers are left at their default
configuration; only joints joint1 ... joint7 are sampled.
"""

from pathlib import Path
import csv

import mujoco
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Paths and configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
LAB_DIR = SCRIPT_DIR.parent

# This path matches the uploaded folder structure: robot-descriptions/franka/...
ROBOT_XML = LAB_DIR / "robot-descriptions" / "franka" / "mjx_panda.xml"
RESULTS_DIR = LAB_DIR / "results"
CSV_PATH = RESULTS_DIR / "franka_task_workspace_points.csv"
PLOT_PATH = RESULTS_DIR / "franka_task_workspace.png"

# In mjx_panda.xml, this site is attached to the Panda hand and is the best
# available end-effector reference for the initial general workspace analysis.
SITE_NAME = "gripper"
ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
NUM_SAMPLES = 10_000
RANDOM_SEED = 42
PROGRESS_EVERY = 1_000


# -----------------------------------------------------------------------------
# Load model and validate names
# -----------------------------------------------------------------------------
def load_model():
    """Load the Franka model and find the selected end-effector site."""
    if not ROBOT_XML.is_file():
        raise FileNotFoundError(
            f"Franka robot XML not found:\n{ROBOT_XML}\n\n"
            "Check that this script is saved in the lab's scripts/ folder and "
            "that robot-descriptions/franka/mjx_panda.xml exists."
        )

    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    data = mujoco.MjData(model)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)
    if site_id < 0:
        available = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid)
            for sid in range(model.nsite)
        ]
        raise ValueError(
            f"Site {SITE_NAME!r} was not found. Available sites: {available}"
        )

    joint_ids = {}
    for name in ARM_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Required Franka arm joint {name!r} was not found.")
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"Expected {name!r} to be a hinge joint.")
        if not model.jnt_limited[jid]:
            raise ValueError(
                f"Joint {name!r} has no explicit limits in the compiled model; "
                "refusing to sample it using guessed limits."
            )
        joint_ids[name] = jid

    return model, data, site_id, joint_ids


# -----------------------------------------------------------------------------
# Joint information and configuration sampling
# -----------------------------------------------------------------------------
def print_joint_information(model, joint_ids):
    """Print all model joints and the limits used for the seven arm joints."""
    print("\n--- FRANKA ARM JOINT INFORMATION ---")
    for name in ARM_JOINT_NAMES:
        jid = joint_ids[name]
        lower, upper = model.jnt_range[jid]
        print(f"{name}: [{lower:.5f}, {upper:.5f}] rad")

    print("\n--- OTHER JOINTS ---")
    arm_set = set(ARM_JOINT_NAMES)
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name in arm_set:
            continue
        limited = bool(model.jnt_limited[jid])
        limits = model.jnt_range[jid].tolist() if limited else "unlimited"
        print(f"{name}: type={int(model.jnt_type[jid])}, limits={limits}; kept at qpos0")


def sample_configuration(model, joint_ids, rng):
    """Sample the seven arm hinges; leave fingers and all other joints at qpos0."""
    qpos = model.qpos0.copy()
    for name in ARM_JOINT_NAMES:
        jid = joint_ids[name]
        qadr = int(model.jnt_qposadr[jid])
        lower, upper = model.jnt_range[jid]
        qpos[qadr] = rng.uniform(lower, upper)
    return qpos


# -----------------------------------------------------------------------------
# Compute sampled task workspace
# -----------------------------------------------------------------------------
def compute_workspace(model, data, site_id, joint_ids):
    """Run forward kinematics for sampled configurations and collect XYZ."""
    rng = np.random.default_rng(RANDOM_SEED)
    points = []

    print(f"\nSampling {NUM_SAMPLES:,} Franka arm configurations...")
    for i in range(NUM_SAMPLES):
        data.qpos[:] = sample_configuration(model, joint_ids, rng)
        mujoco.mj_forward(model, data)
        position = data.site_xpos[site_id].copy()
        if np.all(np.isfinite(position)):
            points.append(position)

        if (i + 1) % PROGRESS_EVERY == 0:
            print(f"Processed {i + 1:,}/{NUM_SAMPLES:,}")

    if not points:
        raise RuntimeError("No valid Franka workspace points were generated.")
    return np.asarray(points, dtype=float)


# -----------------------------------------------------------------------------
# Save CSV and plot
# -----------------------------------------------------------------------------
def save_workspace_csv(points):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x_m", "y_m", "z_m"])
        writer.writerows(points)
    print(f"\nWorkspace CSV saved to:\n{CSV_PATH}")


def set_equal_axes(ax, points):
    """Use equal axis ranges so the 3D workspace is not visually distorted."""
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float(np.max(maximum - minimum)) / 2.0, 0.01)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def plot_workspace(points):
    """Create and save the 3D scatter plot."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, alpha=0.30)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Franka Panda Task Workspace ({len(points):,} samples)")
    set_equal_axes(ax, points)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200, bbox_inches="tight")
    print(f"Workspace plot saved to:\n{PLOT_PATH}")
    plt.show()
    plt.close(fig)


def print_workspace_bounds(points):
    """Print sampled coordinate bounds and spans in metres."""
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    spans = maximum - minimum

    print("\n--- SAMPLED FRANKA TASK WORKSPACE BOUNDS ---")
    print("Axis       Minimum (m)    Maximum (m)    Span (m)")
    for i, axis in enumerate(("X", "Y", "Z")):
        print(
            f"{axis:<10} {minimum[i]:>10.4f}      "
            f"{maximum[i]:>10.4f}      {spans[i]:>10.4f}"
        )
    print(
        "\nThese are sampled bounds for the 'gripper' site in the supplied "
        "Franka XML. They are not exact limits, do not enforce collision-free "
        "motion, and do not guarantee a particular end-effector orientation."
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("Franka Panda Task Workspace Analysis")
    print("------------------------------------")
    print(f"Robot XML: {ROBOT_XML}")
    print(f"Target site: {SITE_NAME}")
    print(f"Samples: {NUM_SAMPLES:,}")
    print(f"Results directory: {RESULTS_DIR}")

    model, data, site_id, joint_ids = load_model()
    print(f"\nJoints in full model: {model.njnt}")
    print(f"Position coordinates (nq): {model.nq}")
    print(f"Velocity coordinates (nv): {model.nv}")
    print(f"Actuators: {model.nu}")
    print_joint_information(model, joint_ids)

    points = compute_workspace(model, data, site_id, joint_ids)
    print(f"\nValid workspace points: {len(points):,}")
    save_workspace_csv(points)
    print_workspace_bounds(points)
    plot_workspace(points)
    print("\nFranka task workspace analysis completed.")


if __name__ == "__main__":
    main()
