#!/usr/bin/env python3
"""
Franka Panda pick-and-place scene preview.

Loads the Franka Panda arm + hand from robot-descriptions/franka/mjx_panda.xml,
then adds the same table, cube, and container components used by the HEAL scene.
The table is not moved or resized: its position and dimensions come from table.xml.

The cube and container are randomized on the tabletop with independent yaw angles,
fully-on-table footprint checks, and a minimum clearance between the cube centre
and the nearest container edge. The robot is held at its XML neutral configuration
and the gripper is left open while gravity/contact physics settle the objects.

Run from repository root:
  python3 Lab/lab03-ik-pick-and-place-heal-frnaka/scripts/franka_pick_place_scene.py
Options:
  --seed N          reproduce a particular randomized layout
  --no-randomize    use the positions in cube.xml/container.xml
  --no-random-yaw   randomize positions but set object yaw to zero
  --headless        simulate for 2 seconds and print a settled-scene check
"""
from pathlib import Path
import argparse
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
LAB_DIR = SCRIPT_DIR.parent
DESCRIPTIONS_DIR = LAB_DIR / "robot-descriptions"
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

from utils.mj_velocity_control.mj_velocity_ctrl import JointVelocityController  # noqa: E402

# This XML contains the Franka arm and its two-finger hand in one model.
# Keep the XML beside its "assets" directory so its meshdir="assets" resolves.
ARM_XML = DESCRIPTIONS_DIR / "franka" / "mjx_panda_new.xml"
TABLE_XML = DESCRIPTIONS_DIR / "table_fr.xml"
CUBE_XML = DESCRIPTIONS_DIR / "cube.xml"
CONTAINER_XML = DESCRIPTIONS_DIR / "container.xml"

# Match the HEAL scene's object dimensions and placement policy.
CUBE_HALF_X = 0.02
CUBE_HALF_Y = 0.02
CUBE_HALF_Z = 0.02
CONTAINER_HALF_X = 0.06
CONTAINER_HALF_Y = 0.06
CONTAINER_BASE_HALF_Z = 0.006
CUBE_EDGE_MARGIN = 0.03
CONTAINER_EDGE_MARGIN = 0.01
GRIPPER_CLEARANCE = 0.10
YAW_RANGE_DEG = 90.0
MAX_SAMPLING_ATTEMPTS = 10000
INITIAL_CLEARANCE_Z = 0.001

CONTROL_DT = 0.01
VELOCITY_KD = 50.0
JOINT_HOLD_KP = 5.0
MAX_JOINT_SPEED = 1.0
# mjx_panda.xml's gripper actuator has ctrlrange 0..0.04, where 0 is closed
# and 0.04 is open. Unlike the HEAL Robotiq actuator, this is not 0..255.
GRIPPER_OPEN_CMD = 0.04
FRICTION = (1.0, 0.005, 0.0001)
ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]


def require_files():
    required = [ARM_XML, TABLE_XML, CUBE_XML, CONTAINER_XML]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required scene file(s) not found:\n  - " + "\n  - ".join(missing)
            + "\nExpected Franka XML at robot-descriptions/franka/mjx_panda.xml "
              "with its sibling assets/ directory."
        )
    assets = ARM_XML.parent / "assets"
    if not assets.is_dir():
        raise FileNotFoundError(
            f"Franka mesh directory not found: {assets}\n"
            "Keep the supplied assets/ directory beside mjx_panda.xml."
        )


def attach_world_component(main_spec, component_path, prefix, frame_name):
    component_spec = mujoco.MjSpec.from_file(str(component_path))
    frame = main_spec.worldbody.add_frame(name=frame_name)
    main_spec.attach(component_spec, prefix=prefix, frame=frame)


def construct_scene():
    """Compile the standalone Franka+hand model with the shared scene objects."""
    require_files()
    spec = mujoco.MjSpec.from_file(str(ARM_XML))

    attach_world_component(spec, TABLE_XML, "table_component/", "table_frame")
    attach_world_component(spec, CUBE_XML, "cube_component/", "cube_frame")
    attach_world_component(spec, CONTAINER_XML, "container_component/", "container_frame")
    return spec.compile()


def find_id_by_suffix(model, object_type, suffix):
    if object_type == mujoco.mjtObj.mjOBJ_GEOM:
        count = model.ngeom
    elif object_type == mujoco.mjtObj.mjOBJ_JOINT:
        count = model.njnt
    elif object_type == mujoco.mjtObj.mjOBJ_ACTUATOR:
        count = model.nu
    else:
        raise ValueError(f"Unsupported object type: {object_type}")

    matches = []
    for item_id in range(count):
        name = mujoco.mj_id2name(model, object_type, item_id)
        if name and (name == suffix or name.endswith("/" + suffix)):
            matches.append(item_id)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {object_type} named/ending in {suffix!r}; "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def find_body_joint(model, suffix):
    """Find a named joint and return its ID."""
    return find_id_by_suffix(model, mujoco.mjtObj.mjOBJ_JOINT, suffix)


def set_free_body_pose(model, data, joint_id, x, y, z, yaw=0.0):
    qadr = int(model.jnt_qposadr[joint_id])
    dadr = int(model.jnt_dofadr[joint_id])
    data.qpos[qadr:qadr + 3] = (x, y, z)
    # MuJoCo free-joint quaternion order is w, x, y, z.
    data.qpos[qadr + 3:qadr + 7] = (
        np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)
    )
    data.qvel[dadr:dadr + 6] = 0.0


def set_contact_friction(model, geom_suffix, friction):
    geom_id = find_id_by_suffix(model, mujoco.mjtObj.mjOBJ_GEOM, geom_suffix)
    model.geom_friction[geom_id, :] = friction


def table_top_height(model, data):
    geom_id = find_id_by_suffix(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    return float(data.geom_xpos[geom_id, 2] + model.geom_size[geom_id, 2])


def table_bounds(model, data):
    geom_id = find_id_by_suffix(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    cx, cy = data.geom_xpos[geom_id, :2]
    hx, hy = model.geom_size[geom_id, :2]
    return float(cx - hx), float(cx + hx), float(cy - hy), float(cy + hy)


def container_gap(cube_x, cube_y, box_x, box_y, box_yaw_deg=0.0):
    """Distance in XY from cube centre to the nearest edge of a rotated container."""
    yaw = np.deg2rad(box_yaw_deg)
    dx, dy = cube_x - box_x, cube_y - box_y
    local_x = np.cos(yaw) * dx + np.sin(yaw) * dy
    local_y = -np.sin(yaw) * dx + np.cos(yaw) * dy
    gx = max(abs(local_x) - CONTAINER_HALF_X, 0.0)
    gy = max(abs(local_y) - CONTAINER_HALF_Y, 0.0)
    return float(np.hypot(gx, gy))


def yaw_extent(half_x, half_y, yaw_deg):
    c = abs(np.cos(np.deg2rad(yaw_deg)))
    s = abs(np.sin(np.deg2rad(yaw_deg)))
    return half_x * c + half_y * s, half_x * s + half_y * c


def randomize_object_positions(model, data, rng, randomize_yaw=True):
    """Sample valid cube/container poses using the same rules as the HEAL scene."""
    cube_joint = find_body_joint(model, "cube_free")
    container_joint = find_body_joint(model, "container_free")
    mujoco.mj_forward(model, data)
    x0, x1, y0, y1 = table_bounds(model, data)
    top_z = table_top_height(model, data)

    for attempt in range(1, MAX_SAMPLING_ATTEMPTS + 1):
        cube_yaw = rng.uniform(-YAW_RANGE_DEG, YAW_RANGE_DEG) if randomize_yaw else 0.0
        box_yaw = rng.uniform(-YAW_RANGE_DEG, YAW_RANGE_DEG) if randomize_yaw else 0.0
        cube_hx, cube_hy = yaw_extent(CUBE_HALF_X, CUBE_HALF_Y, cube_yaw)
        box_hx, box_hy = yaw_extent(CONTAINER_HALF_X, CONTAINER_HALF_Y, box_yaw)

        cube_x = rng.uniform(x0 + cube_hx + CUBE_EDGE_MARGIN,
                             x1 - cube_hx - CUBE_EDGE_MARGIN)
        cube_y = rng.uniform(y0 + cube_hy + CUBE_EDGE_MARGIN,
                             y1 - cube_hy - CUBE_EDGE_MARGIN)
        box_x = rng.uniform(x0 + box_hx + CONTAINER_EDGE_MARGIN,
                            x1 - box_hx - CONTAINER_EDGE_MARGIN)
        box_y = rng.uniform(y0 + box_hy + CONTAINER_EDGE_MARGIN,
                            y1 - box_hy - CONTAINER_EDGE_MARGIN)

        gap = container_gap(cube_x, cube_y, box_x, box_y, box_yaw)
        if gap < GRIPPER_CLEARANCE:
            continue

        set_free_body_pose(
            model, data, cube_joint, cube_x, cube_y,
            top_z + CUBE_HALF_Z + INITIAL_CLEARANCE_Z, np.deg2rad(cube_yaw)
        )
        set_free_body_pose(
            model, data, container_joint, box_x, box_y,
            top_z + CONTAINER_BASE_HALF_Z + INITIAL_CLEARANCE_Z, np.deg2rad(box_yaw)
        )
        mujoco.mj_forward(model, data)

        print("\nRandomized placement:")
        print(f"  Table top Z: {top_z:.4f} m")
        print(f"  Table X range: {x0:.3f} .. {x1:.3f} m")
        print(f"  Table Y range: {y0:.3f} .. {y1:.3f} m")
        print(f"  Cube XY: ({cube_x:.3f}, {cube_y:.3f}) m, yaw {cube_yaw:+.1f} deg")
        print(f"  Container XY: ({box_x:.3f}, {box_y:.3f}) m, yaw {box_yaw:+.1f} deg")
        print(f"  Cube-centre to container edge: {gap * 100:.1f} cm "
              f"(minimum {GRIPPER_CLEARANCE * 100:.0f} cm); attempts: {attempt}")
        return

    raise RuntimeError(
        f"Could not find a valid layout with {GRIPPER_CLEARANCE:.2f} m clearance "
        f"after {MAX_SAMPLING_ATTEMPTS} attempts. Check table dimensions."
    )


def build_world(randomize=True, seed=None, randomize_yaw=True):
    model = construct_scene()
    data = mujoco.MjData(model)
    model.opt.gravity[:] = (0.0, 0.0, -9.81)

    for geom_name in ("table_top", "cube_geom", "container_base"):
        set_contact_friction(model, geom_name, FRICTION)

    mujoco.mj_forward(model, data)
    if randomize:
        randomize_object_positions(model, data, np.random.default_rng(seed), randomize_yaw)
    else:
        print("\nUsing the default XML positions for cube and container.")
    return model, data


class NeutralHold:
    """Hold Franka's seven arm joints near qpos0; leave its gripper open."""

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.joint_ids = [find_body_joint(model, name) for name in ARM_JOINT_NAMES]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.joint_ids], dtype=int)
        self.dofadr = np.array([model.jnt_dofadr[j] for j in self.joint_ids], dtype=int)

        # Identify actuators by the joint they control, rather than assuming IDs.
        self.arm_actuator_ids = []
        for joint_id in self.joint_ids:
            found = [
                aid for aid in range(model.nu)
                if int(model.actuator_trnid[aid, 0]) == joint_id
            ]
            if len(found) != 1:
                raise ValueError(
                    f"Expected one actuator for joint "
                    f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)!r}; "
                    f"found actuator IDs {found}."
                )
            self.arm_actuator_ids.append(found[0])

        grip_candidates = []
        for aid in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or ""
            joint_id = int(model.actuator_trnid[aid, 0])
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) if joint_id >= 0 else ""
            if name == "gripper" or joint_name == "finger_joint1":
                grip_candidates.append(aid)
        if len(grip_candidates) != 1:
            raise ValueError(f"Could not uniquely identify Franka gripper actuator: {grip_candidates}")
        self.gripper_id = grip_candidates[0]

        self.controller = JointVelocityController(
            model, data, kd=VELOCITY_KD, gripper_ids=[self.gripper_id]
        )
        # Controller maps actuator IDs to DOFs. Feed-forward factor matches the
        # HEAL scene's gravity-compensation + D-control hold.
        self.ff = 1.0 + model.dof_damping[self.dofadr] / VELOCITY_KD
        self.q_hold = data.qpos[self.qadr].copy()
        self.n_sub = max(1, int(round(CONTROL_DT / model.opt.timestep)))
        self.dt = self.n_sub * model.opt.timestep

    def step(self):
        err = self.q_hold - self.data.qpos[self.qadr]
        velocity = np.clip(JOINT_HOLD_KP * err, -MAX_JOINT_SPEED, MAX_JOINT_SPEED)
        targets = np.zeros(self.model.nu)
        for i, actuator_id in enumerate(self.arm_actuator_ids):
            targets[actuator_id] = velocity[i] * self.ff[i]
        self.controller.set_velocity_target(targets)

        for _ in range(self.n_sub):
            self.data.ctrl[self.gripper_id] = GRIPPER_OPEN_CMD
            mujoco.mj_step(self.model, self.data)


def report(model, data, hold):
    top = table_top_height(model, data)
    cube_joint = find_body_joint(model, "cube_free")
    container_joint = find_body_joint(model, "container_free")
    cube = data.qpos[int(model.jnt_qposadr[cube_joint]):][:3]
    container = data.qpos[int(model.jnt_qposadr[container_joint]):][:3]
    drift = float(np.max(np.abs(data.qpos[hold.qadr] - hold.q_hold)))
    print("\nSettled-scene check:")
    print(f"  Cube centre Z:      {cube[2]:.4f} m (expected about {top + CUBE_HALF_Z:.4f})")
    print(f"  Container base Z:   {container[2]:.4f} m (expected about {top + CONTAINER_BASE_HALF_Z:.4f})")
    print(f"  Franka joint drift: {drift:.5f} rad")
    print(f"  Gripper command:    open ({GRIPPER_OPEN_CMD:.3f} m)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=None, help="seed for reproducible object placement")
    parser.add_argument("--no-randomize", action="store_true", help="use XML object positions")
    parser.add_argument("--no-random-yaw", action="store_true", help="randomize positions but keep yaw at zero")
    parser.add_argument("--headless", action="store_true", help="settle physics for 2 seconds without viewer")
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % 2**31)

    print("Building Franka Panda pick-and-place scene...")
    print(f"Franka XML: {ARM_XML}")
    if not args.no_randomize:
        print(f"Layout seed: {seed} (rerun with --seed {seed} to reproduce)")
    model, data = build_world(
        randomize=not args.no_randomize,
        seed=seed,
        randomize_yaw=not args.no_random_yaw,
    )
    hold = NeutralHold(model, data)
    mujoco.set_mjcb_control(hold.controller.control_callback)

    print("\nScene ready.")
    print(f"  Gravity: {model.opt.gravity}")
    print(f"  Table/cube/container friction: {FRICTION}")
    print(f"  Bodies: {model.nbody} | Joints: {model.njnt} | Actuators: {model.nu}")
    print(f"  Control period: {hold.dt * 1000:.0f} ms; arm held near qpos0; gripper open")

    try:
        if args.headless:
            for _ in range(max(1, int(round(2.0 / hold.dt)))):
                hold.step()
            report(model, data, hold)
            return

        print("\nOpening MuJoCo viewer and advancing physics. Close the viewer to exit.")
        with mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            reported = False
            while viewer.is_running():
                hold.step()
                viewer.sync()
                if not reported and data.time >= 2.0:
                    report(model, data, hold)
                    reported = True
                time.sleep(hold.dt)
    finally:
        mujoco.set_mjcb_control(None)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nCould not build the Franka scene: {exc}", file=sys.stderr)
        raise
