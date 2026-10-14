#!/usr/bin/env python3
"""Demo of the HEAL + Robotiq 2F-85 pick-and-place scene used by main_heal_mink/dls/qp.py.

Shows exactly the environment the task scripts start from: the HEAL arm with the gripper open at its
neutral configuration, the fixed table, and the cube and container placed randomly on the table
(random position AND yaw, rotated footprints fully on the table, and at least GRIPPER_CLEARANCE between
the cube centre and the container so the open gripper cannot touch the container while picking).
Gravity, contact friction and the velocity controller (gravity compensation + D term, arm held at its
neutral joints) match the task scripts. No pick-and-place controller runs here.

This module also provides the scene helpers the task scripts import (construct_scene, find_id_by_suffix,
set_free_body_pose, set_contact_friction, table_top_height, table_bounds, geometry constants), so keep
those names when editing. The placement constants/logic below mirror the task scripts' layout sampler;
if you change one, change the other.

Run from the repository root:
  python Lab/lab03-ik-pick-and-place-heal-frnaka/scripts/heal_pick_place_scene.py [--seed N]
        [--no-random-yaw] [--no-randomize] [--headless]
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

# ---- random placement (same values as the task scripts) ----
CUBE_EDGE_MARGIN = 0.03        # m, keeps the open fingers over the table around the cube
CONTAINER_EDGE_MARGIN = 0.01   # m
GRIPPER_CLEARANCE = 0.10       # m, cube centre -> nearest container edge
YAW_RANGE_DEG = 90.0           # random yaw uniform in +-this (squares repeat every 90 deg)
MAX_SAMPLING_ATTEMPTS = 10000
OBJECT_YAW = 0.0               # yaw (rad) when yaw randomization is off
INITIAL_CLEARANCE_Z = 0.001

# ---- control (same as the task scripts) ----
CONTROL_DT = 0.01
VELOCITY_KD = 50.0
JOINT_HOLD_KP = 5.0
MAX_JOINT_SPEED = 1.0
GRIPPER_OPEN_CMD = 0.0
FRICTION = (1.0, 0.005, 0.0001)


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


def container_gap(cube_x, cube_y, box_x, box_y, box_yaw_deg=0.0):
    """Horizontal distance from the cube centre to the nearest edge of the (rotated) container."""
    yaw = np.deg2rad(box_yaw_deg)
    dx, dy = cube_x - box_x, cube_y - box_y
    lx = np.cos(yaw) * dx + np.sin(yaw) * dy      # cube centre in the container frame
    ly = -np.sin(yaw) * dx + np.cos(yaw) * dy
    gx = max(abs(lx) - CONTAINER_HALF_X, 0.0)
    gy = max(abs(ly) - CONTAINER_HALF_Y, 0.0)
    return float(np.hypot(gx, gy))


def _yaw_extent(half_x, half_y, yaw_deg):
    """Half-extent of a rotated rectangle's axis-aligned bounding box."""
    c, s = abs(np.cos(np.deg2rad(yaw_deg))), abs(np.sin(np.deg2rad(yaw_deg)))
    return half_x * c + half_y * s, half_x * s + half_y * c


def randomize_object_positions(model, data, rng, randomize_yaw=True):
    """Random cube/container pose on the table with gripper clearance between them.

    Table bounds and height are read from the compiled table geometry. The rotated footprints must
    stay fully on the table (with edge margins). Objects start 1 mm above the table and settle under
    gravity and contact.
    """
    cube_joint = find_id_by_suffix(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    box_joint = find_id_by_suffix(model, mujoco.mjtObj.mjOBJ_JOINT, "container_free")
    mujoco.mj_forward(model, data)
    x0, x1, y0, y1 = table_bounds(model, data)
    top_z = table_top_height(model, data)

    for attempt in range(1, MAX_SAMPLING_ATTEMPTS + 1):
        cyaw = rng.uniform(-YAW_RANGE_DEG, YAW_RANGE_DEG) if randomize_yaw else 0.0
        byaw = rng.uniform(-YAW_RANGE_DEG, YAW_RANGE_DEG) if randomize_yaw else 0.0
        chx, chy = _yaw_extent(CUBE_HALF_X, CUBE_HALF_Y, cyaw)
        bhx, bhy = _yaw_extent(CONTAINER_HALF_X, CONTAINER_HALF_Y, byaw)
        cx = rng.uniform(x0 + chx + CUBE_EDGE_MARGIN, x1 - chx - CUBE_EDGE_MARGIN)
        cy = rng.uniform(y0 + chy + CUBE_EDGE_MARGIN, y1 - chy - CUBE_EDGE_MARGIN)
        bx = rng.uniform(x0 + bhx + CONTAINER_EDGE_MARGIN, x1 - bhx - CONTAINER_EDGE_MARGIN)
        by = rng.uniform(y0 + bhy + CONTAINER_EDGE_MARGIN, y1 - bhy - CONTAINER_EDGE_MARGIN)
        gap = container_gap(cx, cy, bx, by, byaw)
        if gap < GRIPPER_CLEARANCE:
            continue

        set_free_body_pose(model, data, cube_joint, cx, cy,
                           top_z + CUBE_HALF_Z + INITIAL_CLEARANCE_Z, np.deg2rad(cyaw))
        set_free_body_pose(model, data, box_joint, bx, by,
                           top_z + CONTAINER_BASE_HALF_Z + INITIAL_CLEARANCE_Z, np.deg2rad(byaw))
        mujoco.mj_forward(model, data)

        print("\nRandomized placement (table-relative limits respected):")
        print(f"  Table top Z:    {top_z:.4f} m (read from table.xml geometry)")
        print(f"  Table X range:  {x0:.3f} .. {x1:.3f} m | Y range: {y0:.3f} .. {y1:.3f} m")
        print(f"  Cube XY:        ({cx:.3f}, {cy:.3f}) m, yaw {cyaw:+.1f} deg")
        print(f"  Container XY:   ({bx:.3f}, {by:.3f}) m, yaw {byaw:+.1f} deg")
        print(f"  Cube centre -> container edge: {gap * 100:.1f} cm "
              f"(required >= {GRIPPER_CLEARANCE * 100:.0f} cm) | sampling attempts: {attempt}")
        return
    raise RuntimeError(
        f"No valid placement with {GRIPPER_CLEARANCE * 100:.0f} cm gripper clearance found "
        f"in {MAX_SAMPLING_ATTEMPTS} attempts."
    )


def build_world(randomize=True, seed=None, randomize_yaw=True):
    """Compile the scene and apply the task environment: gravity, friction, object layout."""
    model = construct_scene()
    data = mujoco.MjData(model)

    # Explicitly enable normal downward gravity for the entire simulation.
    model.opt.gravity[:] = (0.0, 0.0, -9.81)

    # Friction stays enabled throughout; it is not toggled on only during grasping.
    for geom in ("table_top", "cube_geom", "container_base"):
        set_contact_friction(model, geom, FRICTION)

    mujoco.mj_forward(model, data)
    if randomize:
        randomize_object_positions(model, data, np.random.default_rng(seed), randomize_yaw)
    else:
        print("\nUsing the default XML positions for cube and container (no randomization).")
    return model, data


class NeutralHold:
    """Holds the arm at its initial joints with the task's JointVelocityController; gripper open."""

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.arm_joints = [find_id_by_suffix(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i}") for i in range(1, 7)]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.arm_joints])
        self.dofadr = np.array([model.jnt_dofadr[j] for j in self.arm_joints])
        acts = [a for a in range(model.nu) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "")
                .endswith("fingers_actuator")]
        if len(acts) != 1 or list(range(6)) != [a for a in range(model.nu) if a != acts[0]][:6]:
            raise ValueError("Unexpected actuator layout (expected arm actuators 0-5 and one gripper actuator)")
        self.grip = acts[0]
        self.controller = JointVelocityController(model, data, kd=VELOCITY_KD, gripper_ids=[self.grip])
        self.ff = 1.0 + model.dof_damping[self.dofadr] / VELOCITY_KD  # same damping feed-forward as the task
        self.q_hold = data.qpos[self.qadr].copy()
        self.n_sub = max(1, int(round(CONTROL_DT / model.opt.timestep)))
        self.dt = self.n_sub * model.opt.timestep

    def step(self):
        """One control period: hold neutral joints, gripper open."""
        err = self.q_hold - self.data.qpos[self.qadr]
        v = np.clip(JOINT_HOLD_KP * err, -MAX_JOINT_SPEED, MAX_JOINT_SPEED)
        v7 = np.zeros(self.model.nu)
        v7[:6] = v * self.ff
        self.controller.set_velocity_target(v7)
        for _ in range(self.n_sub):
            self.data.ctrl[self.grip] = GRIPPER_OPEN_CMD
            mujoco.mj_step(self.model, self.data)


def report(model, data, hold):
    """Check the settled scene: objects at rest on the table, arm still at neutral."""
    top = table_top_height(model, data)
    cube = data.qpos[model.jnt_qposadr[find_id_by_suffix(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")]:][:3]
    box = data.qpos[model.jnt_qposadr[find_id_by_suffix(model, mujoco.mjtObj.mjOBJ_JOINT, "container_free")]:][:3]
    drift = float(np.max(np.abs(data.qpos[hold.qadr] - hold.q_hold)))
    print("\nSettled scene check:")
    print(f"  Cube z {cube[2]:.4f} m (resting height {top + CUBE_HALF_Z:.4f})")
    print(f"  Container z {box[2]:.4f} m (resting height {top + CONTAINER_BASE_HALF_Z:.4f})")
    print(f"  Arm drift from neutral: {drift:.5f} rad | gripper command: open ({GRIPPER_OPEN_CMD:.0f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=None, help="layout seed (printed if omitted)")
    ap.add_argument("--no-randomize", action="store_true", help="keep the default XML object positions")
    ap.add_argument("--no-random-yaw", action="store_true", help="random positions only; yaw stays 0")
    ap.add_argument("--headless", action="store_true", help="no viewer: settle for 2 s, print a check, exit")
    args = ap.parse_args()
    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % 2**31)

    print("Building HEAL pick-and-place scene...")
    if not args.no_randomize:
        print(f"Layout seed: {seed}  (rerun with --seed {seed} to reproduce)")
    model, data = build_world(not args.no_randomize, seed, not args.no_random_yaw)
    hold = NeutralHold(model, data)
    mujoco.set_mjcb_control(hold.controller.control_callback)

    print("\nScene ready.")
    print(f"  Gravity: {model.opt.gravity}")
    print("  Table/cube/container contact friction: enabled")
    print(f"  Bodies: {model.nbody} | Joints: {model.njnt} | Actuators: {model.nu}")
    print(f"  Control period {hold.dt * 1000:.0f} ms | arm held at neutral | gripper open")

    try:
        if args.headless:
            for _ in range(int(2.0 / hold.dt)):
                hold.step()
            report(model, data, hold)
            return
        print("\nOpening viewer and advancing physics. Close the viewer to exit.")
        with mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            settled_reported = False
            while viewer.is_running():
                hold.step()
                viewer.sync()
                if not settled_reported and data.time > 2.0:
                    report(model, data, hold)
                    settled_reported = True
                time.sleep(hold.dt)
    finally:
        mujoco.set_mjcb_control(None)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nCould not build the scene: {exc}", file=sys.stderr)
        raise