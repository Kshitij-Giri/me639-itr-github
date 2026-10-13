#!/usr/bin/env python3
"""Build and simulate a randomized HEAL + Robotiq 2F-85 scene.

The table stays fixed. Cube and container start just above the tabletop and
settle onto it through gravity and contact. Their starting Z coordinates are
derived from the compiled table geometry, not hard-coded table heights.
This is a physics/scene test only; no pick-and-place controller is included.
"""
from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
LAB_DIR = SCRIPT_DIR.parent
DESCRIPTIONS_DIR = LAB_DIR / "robot-descriptions"

ARM_XML = DESCRIPTIONS_DIR / "single_arm_heal_effort_actuation_rs_mj.xml"
GRIPPER_XML = DESCRIPTIONS_DIR / "robotiq_2f85_v4" / "2f85.xml"
TABLE_XML = DESCRIPTIONS_DIR / "table.xml"
CUBE_XML = DESCRIPTIONS_DIR / "cube.xml"
CONTAINER_XML = DESCRIPTIONS_DIR / "container.xml"

# Dimensions of the square footprints, matching cube.xml/container.xml.
CUBE_HALF_X = 0.02
CUBE_HALF_Y = 0.02
CUBE_HALF_Z = 0.02
CONTAINER_HALF_X = 0.06
CONTAINER_HALF_Y = 0.06
CONTAINER_BASE_HALF_Z = 0.006

# Robotiq 2F-85 nominal opening; XML does not encode a single exact outside
# width scalar. Required center distance is half this nominal opening.
GRIPPER_OPENING_GAP_M = 0.085
MIN_OBJECT_CENTER_DISTANCE = GRIPPER_OPENING_GAP_M / 2.0

EDGE_MARGIN = 0.005
MAX_SAMPLING_ATTEMPTS = 10000
RANDOM_SEED = None
OBJECT_YAW = 0.0
INITIAL_CLEARANCE_Z = 0.001
VIEWER_SLEEP_SECONDS = 0.005


def require_files():
    paths = [ARM_XML, GRIPPER_XML, TABLE_XML, CUBE_XML, CONTAINER_XML]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required scene component(s) not found:\n  - "
            + "\n  - ".join(missing)
            + "\nCheck that the files are in robot-descriptions/."
        )


def attach_world_component(main_spec, component_path, prefix, frame_name):
    component_spec = mujoco.MjSpec.from_file(str(component_path))
    world_frame = main_spec.worldbody.add_frame(name=frame_name)
    main_spec.attach(component_spec, prefix=prefix, frame=world_frame)


def construct_scene():
    require_files()
    arm_spec = mujoco.MjSpec.from_file(str(ARM_XML))

    gripper_spec = mujoco.MjSpec.from_file(str(GRIPPER_XML))
    mount_site = arm_spec.site("right_center")
    if mount_site is None:
        raise ValueError("HEAL model does not contain site 'right_center'.")
    arm_spec.attach(gripper_spec, prefix="gripper/", site=mount_site)

    attach_world_component(
        arm_spec, TABLE_XML, prefix="table_component/", frame_name="table_frame"
    )
    attach_world_component(
        arm_spec, CUBE_XML, prefix="cube_component/", frame_name="cube_frame"
    )
    attach_world_component(
        arm_spec, CONTAINER_XML, prefix="container_component/",
        frame_name="container_frame"
    )
    return arm_spec.compile()

def compensate_gravity(model, data):
    """
    Applies feedforward control to each actuator to compensate for gravity and Coriolis effects.
    This uses the bias force vector `data.qfrc_bias`, which is populated by `mj_forward()`.
    """
    for aid in range(model.nu):
        joint_dof = model.actuator_trnid[aid][0]
        data.ctrl[aid] = data.qfrc_bias[joint_dof]

def find_id_by_suffix(model, object_type, suffix):
    matches = []
    for item_id in range(model.ngeom if object_type == mujoco.mjtObj.mjOBJ_GEOM else model.njnt):
        name = mujoco.mj_id2name(model, object_type, item_id)
        if name and (name == suffix or name.endswith("/" + suffix)):
            matches.append(item_id)
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {object_type} named/ending in {suffix!r}; "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def set_free_body_pose(model, data, joint_id, x, y, z, yaw=0.0):
    qpos_adr = int(model.jnt_qposadr[joint_id])
    qvel_adr = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos_adr:qpos_adr + 3] = (x, y, z)
    data.qpos[qpos_adr + 3:qpos_adr + 7] = (
        np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)
    )
    data.qvel[qvel_adr:qvel_adr + 6] = 0.0


def set_contact_friction(model, geom_suffix, friction):
    geom_id = find_id_by_suffix(model, mujoco.mjtObj.mjOBJ_GEOM, geom_suffix)
    model.geom_friction[geom_id, :] = friction


def table_top_height(model, data):
    """Read actual world-space top height from the compiled table_top box."""
    geom_id = find_id_by_suffix(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    # table_top is an axis-aligned box; geom_size[2] is its half-thickness.
    return float(data.geom_xpos[geom_id, 2] + model.geom_size[geom_id, 2])


def table_bounds(model, data):
    geom_id = find_id_by_suffix(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    center_x, center_y = data.geom_xpos[geom_id, :2]
    half_x, half_y = model.geom_size[geom_id, :2]
    return (
        float(center_x - half_x), float(center_x + half_x),
        float(center_y - half_y), float(center_y + half_y),
    )


def footprints_overlap(cube_x, cube_y, container_x, container_y):
    overlap_x = abs(cube_x - container_x) < CUBE_HALF_X + CONTAINER_HALF_X
    overlap_y = abs(cube_y - container_y) < CUBE_HALF_Y + CONTAINER_HALF_Y
    return overlap_x and overlap_y


def randomize_object_positions(model, data, rng):
    cube_joint = find_id_by_suffix(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    container_joint = find_id_by_suffix(
        model, mujoco.mjtObj.mjOBJ_JOINT, "container_free"
    )

    # Forward kinematics first, then derive table bounds/top from the compiled
    # table geom. This makes object placement follow edits to table.xml.
    mujoco.mj_forward(model, data)
    x_min, x_max, y_min, y_max = table_bounds(model, data)
    top_z = table_top_height(model, data)

    cube_x_min = x_min + CUBE_HALF_X + EDGE_MARGIN
    cube_x_max = x_max - CUBE_HALF_X - EDGE_MARGIN
    cube_y_min = y_min + CUBE_HALF_Y + EDGE_MARGIN
    cube_y_max = y_max - CUBE_HALF_Y - EDGE_MARGIN

    container_x_min = x_min + CONTAINER_HALF_X + EDGE_MARGIN
    container_x_max = x_max - CONTAINER_HALF_X - EDGE_MARGIN
    container_y_min = y_min + CONTAINER_HALF_Y + EDGE_MARGIN
    container_y_max = y_max - CONTAINER_HALF_Y - EDGE_MARGIN

    for attempt in range(1, MAX_SAMPLING_ATTEMPTS + 1):
        cube_x = rng.uniform(cube_x_min, cube_x_max)
        cube_y = rng.uniform(cube_y_min, cube_y_max)
        container_x = rng.uniform(container_x_min, container_x_max)
        container_y = rng.uniform(container_y_min, container_y_max)

        distance = float(np.hypot(cube_x - container_x, cube_y - container_y))
        if distance < MIN_OBJECT_CENTER_DISTANCE:
            continue
        if footprints_overlap(cube_x, cube_y, container_x, container_y):
            continue

        # Place objects slightly above the top; simulation gravity settles them
        # onto the table. Yaw=0 keeps all square sides parallel.
        set_free_body_pose(
            model, data, cube_joint, cube_x, cube_y,
            top_z + CUBE_HALF_Z + INITIAL_CLEARANCE_Z, OBJECT_YAW
        )
        set_free_body_pose(
            model, data, container_joint, container_x, container_y,
            top_z + CONTAINER_BASE_HALF_Z + INITIAL_CLEARANCE_Z, OBJECT_YAW
        )
        mujoco.mj_forward(model, data)

        print("\nRandomized placement:")
        print(f"  Table top Z:      {top_z:.4f} m (read from table.xml geometry)")
        print(f"  Cube XY:          ({cube_x:.3f}, {cube_y:.3f}) m")
        print(f"  Container XY:     ({container_x:.3f}, {container_y:.3f}) m")
        print(f"  Center distance:  {distance:.3f} m")
        print(f"  Minimum distance: {MIN_OBJECT_CENTER_DISTANCE:.3f} m")
        print("  Cube/container yaw: 0 degrees (sides parallel)")
        print(f"  Sampling attempts: {attempt}")
        return


    raise RuntimeError(
        f"No valid placement found in {MAX_SAMPLING_ATTEMPTS} attempts."
    )


def main():
    print("Building HEAL pick-and-place scene...")
    model = construct_scene()
    data = mujoco.MjData(model)

    # Explicitly enable normal downward gravity for the entire simulation.
    model.opt.gravity[:] = (0.0, 0.0, -9.81)

    # Ensure friction on table, cube and container contact surfaces. Friction
    # stays enabled throughout; it is not toggled on only during grasping.
    set_contact_friction(model, "table_top", (1.0, 0.005, 0.0001))
    set_contact_friction(model, "cube_geom", (1.0, 0.005, 0.0001))
    set_contact_friction(model, "container_base", (1.0, 0.005, 0.0001))

    rng = np.random.default_rng(RANDOM_SEED)
    mujoco.mj_forward(model, data)
    randomize_object_positions(model, data, rng)

    print("\nScene ready.")
    print(f"  Gravity: {model.opt.gravity}")
    print("  Table/cube/container contact friction: enabled")
    print(f"  Bodies: {model.nbody} | Joints: {model.njnt} | Actuators: {model.nu}")
    print("\nOpening viewer and advancing physics. Close the viewer to exit.")
    print("NOTE: no robot controller is running yet; the uncommanded arm may move/fall.")

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        while viewer.is_running():
            compensate_gravity(model, data)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(VIEWER_SLEEP_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nCould not build the scene: {exc}", file=sys.stderr)
        raise