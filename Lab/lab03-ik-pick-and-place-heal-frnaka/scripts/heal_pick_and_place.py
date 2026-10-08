#!/usr/bin/env python3

import os
import time
import numpy as np
import mujoco
import mujoco.viewer


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DESC_DIR = os.path.join(REPO_ROOT, "robot_descriptions")

ARM_XML = os.path.join(
    DESC_DIR, "single_arm_heal_effort_actuation_rs_mj.xml"
)
GRIPPER_XML = os.path.join(
    DESC_DIR, "robotiq_2f85_v4", "2f85.xml"
)


# ============================================================
# ROBOT / GRIPPER
# ============================================================

ARM_JOINTS = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6",
]

EE_SITE = "right_center"

# Your existing gripper example has six HEAL actuators followed
# by the Robotiq actuator.
GRIPPER_ACTUATOR_ID = 6
GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 255.0


# ============================================================
# TASK SCENE
# ============================================================

# The HEAL zero pose puts the right_center site at roughly
# z = 0.48 m, so a 0.42 m table top gives useful clearance.
TABLE_CENTER = np.array([0.38, 0.02, 0.39])
TABLE_HALF_X = 0.35
TABLE_HALF_Y = 0.35
TABLE_HALF_Z = 0.05
TABLE_TOP_Z = TABLE_CENTER[2] + TABLE_HALF_Z

TABLE_COLOR = [0.45, 0.22, 0.07, 1.0]

CUBE_HALF = 0.04
CUBE_MASS = 0.10
CUBE_COLOR = [0.90, 0.02, 0.02, 1.0]

CONTAINER_CENTER = np.array([0.60, 0.18, TABLE_TOP_Z])
CONTAINER_X = 0.24
CONTAINER_Y = 0.24
CONTAINER_WALL = 0.025
CONTAINER_BASE = 0.02
CONTAINER_HEIGHT = 0.10
CONTAINER_COLOR = [0.01, 0.01, 0.01, 1.0]


# ============================================================
# CONTROLLER PARAMETERS
# ============================================================

KP = np.array([180.0, 180.0, 150.0, 80.0, 70.0, 50.0])
KD = np.array([35.0, 35.0, 30.0, 15.0, 12.0, 8.0])
MAX_TORQUE = 250.0

IK_MAX_ITERS = 300
IK_STEP = 0.45
IK_DAMPING = 0.05

POSITION_TOL = 0.012
ORIENTATION_TOL = 0.08

APPROACH_HEIGHT = 0.14
LIFT_HEIGHT = 0.16
DROP_SITE_HEIGHT = 0.12


# ============================================================
# SCENE BUILDING
# ============================================================

def add_box(body, name, pos, half_size, rgba,
            friction=None, mass=None):
    geom = body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=pos,
        size=half_size,
        rgba=rgba,
    )

    if friction is not None:
        geom.friction = friction

    if mass is not None:
        geom.mass = mass

    return geom


def add_table(spec):
    table = spec.worldbody.add_body(
        name="pick_place_table",
        pos=TABLE_CENTER,
    )

    add_box(
        table,
        "table_top",
        [0, 0, 0],
        [TABLE_HALF_X, TABLE_HALF_Y, TABLE_HALF_Z],
        TABLE_COLOR,
        friction=[1.2, 0.01, 0.001],
    )


def add_container(spec):
    container = spec.worldbody.add_body(
        name="black_container",
        pos=CONTAINER_CENTER,
    )

    hx = CONTAINER_X / 2.0
    hy = CONTAINER_Y / 2.0
    w = CONTAINER_WALL
    h = CONTAINER_HEIGHT

    add_box(
        container, "container_bottom",
        [0, 0, CONTAINER_BASE / 2],
        [hx, hy, CONTAINER_BASE / 2],
        CONTAINER_COLOR,
    )

    add_box(
        container, "container_left",
        [-(hx - w / 2), 0, CONTAINER_BASE + h / 2],
        [w / 2, hy, h / 2],
        CONTAINER_COLOR,
    )

    add_box(
        container, "container_right",
        [(hx - w / 2), 0, CONTAINER_BASE + h / 2],
        [w / 2, hy, h / 2],
        CONTAINER_COLOR,
    )

    add_box(
        container, "container_front",
        [0, -(hy - w / 2), CONTAINER_BASE + h / 2],
        [hx, w / 2, h / 2],
        CONTAINER_COLOR,
    )

    add_box(
        container, "container_back",
        [0, (hy - w / 2), CONTAINER_BASE + h / 2],
        [hx, w / 2, h / 2],
        CONTAINER_COLOR,
    )


def choose_cube_position(rng):
    for _ in range(1000):
        x = rng.uniform(
            TABLE_CENTER[0] - 0.22,
            TABLE_CENTER[0] + 0.12,
        )
        y = rng.uniform(
            TABLE_CENTER[1] - 0.20,
            TABLE_CENTER[1] + 0.20,
        )

        if np.linalg.norm(
            np.array([x, y]) - CONTAINER_CENTER[:2]
        ) > 0.18:
            return np.array([
                x,
                y,
                TABLE_TOP_Z + CUBE_HALF,
            ])

    raise RuntimeError("Could not find cube spawn position.")


def add_cube(spec, cube_pos):
    cube = spec.worldbody.add_body(
        name="red_cube",
        pos=cube_pos,
    )

    cube.add_freejoint()

    add_box(
        cube,
        "red_cube_geom",
        [0, 0, 0],
        [CUBE_HALF, CUBE_HALF, CUBE_HALF],
        CUBE_COLOR,
        friction=[1.5, 0.02, 0.002],
        mass=CUBE_MASS,
    )


def build_model(cube_pos):
    arm = mujoco.MjSpec.from_file(ARM_XML)
    gripper = mujoco.MjSpec.from_file(GRIPPER_XML)

    # Same HEAL + Robotiq attachment as your existing code.
    arm.attach(
        gripper,
        prefix="gripper/",
        site=arm.site(EE_SITE),
    )

    add_table(arm)
    add_container(arm)
    add_cube(arm, cube_pos)

    return arm.compile()


# ============================================================
# JOINT / IK UTILITIES
# ============================================================

def get_joint_addresses(model):
    qpos_adr = []
    dof_adr = []

    for name in ARM_JOINTS:
        jid = model.joint(name).id
        qpos_adr.append(model.jnt_qposadr[jid])
        dof_adr.append(model.jnt_dofadr[jid])

    return (
        np.array(qpos_adr, dtype=int),
        np.array(dof_adr, dtype=int),
    )


def rotation_error(current_R, target_R):
    return 0.5 * (
        np.cross(current_R[:, 0], target_R[:, 0])
        + np.cross(current_R[:, 1], target_R[:, 1])
        + np.cross(current_R[:, 2], target_R[:, 2])
    )


def solve_ik(model, start_q, target_pos, target_R,
             qpos_adr, dof_adr):

    ik_data = mujoco.MjData(model)
    ik_data.qpos[:] = model.qpos0
    ik_data.qvel[:] = 0.0
    ik_data.qpos[qpos_adr] = start_q

    site_id = model.site(EE_SITE).id
    q = start_q.copy()

    for _ in range(IK_MAX_ITERS):
        mujoco.mj_forward(model, ik_data)

        current_pos = ik_data.site(EE_SITE).xpos.copy()
        current_R = ik_data.site(EE_SITE).xmat.reshape(3, 3).copy()

        pos_err = target_pos - current_pos
        rot_err = rotation_error(current_R, target_R)
        error = np.concatenate([pos_err, rot_err])

        if (
            np.linalg.norm(pos_err) < 0.002
            and np.linalg.norm(rot_err) < 0.015
        ):
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        mujoco.mj_jacSite(
            model,
            ik_data,
            jacp,
            jacr,
            site_id,
        )

        J = np.vstack([
            jacp[:, dof_adr],
            jacr[:, dof_adr],
        ])

        A = J @ J.T + (IK_DAMPING ** 2) * np.eye(6)
        dq = J.T @ np.linalg.solve(A, error)

        step_norm = np.linalg.norm(dq)
        if step_norm > 0.08:
            dq *= 0.08 / step_norm

        q += IK_STEP * dq

        for i, name in enumerate(ARM_JOINTS):
            jid = model.joint(name).id
            low, high = model.jnt_range[jid]
            q[i] = np.clip(q[i], low + 0.02, high - 0.02)

        ik_data.qpos[qpos_adr] = q
        ik_data.qvel[:] = 0.0

    mujoco.mj_forward(model, ik_data)
    final_pos = ik_data.site(EE_SITE).xpos.copy()

    if np.linalg.norm(target_pos - final_pos) > 0.025:
        print(
            "WARNING: IK position error = "
            f"{np.linalg.norm(target_pos - final_pos):.3f} m"
        )

    return q


# ============================================================
# TORQUE / GRIPPER CONTROL
# ============================================================

def apply_arm_pd(model, data, target_q, qpos_adr, dof_adr):
    q = data.qpos[qpos_adr]
    qd = data.qvel[dof_adr]

    tau = (
        data.qfrc_bias[dof_adr]
        + KP * (target_q - q)
        - KD * qd
    )

    tau = np.clip(tau, -MAX_TORQUE, MAX_TORQUE)

    for i in range(6):
        data.ctrl[i] = tau[i]


def set_gripper(data, closed):
    data.ctrl[GRIPPER_ACTUATOR_ID] = (
        GRIPPER_CLOSE if closed else GRIPPER_OPEN
    )


# ============================================================
# PICK-AND-PLACE STATE MACHINE
# ============================================================

class PickPlaceTask:

    def __init__(self, model, data, cube_pos,
                 qpos_adr, dof_adr):

        self.model = model
        self.data = data
        self.qpos_adr = qpos_adr
        self.dof_adr = dof_adr
        self.cube_pos = cube_pos.copy()

        mujoco.mj_forward(model, data)

        self.home_q = data.qpos[qpos_adr].copy()
        self.home_pos = data.site(EE_SITE).xpos.copy()

        # Keep the gripper in the same downward-facing orientation
        # during the entire pick-and-place motion.
        self.target_R = (
            data.site(EE_SITE).xmat.reshape(3, 3).copy()
        )

        # -------- Pick waypoints --------

        approach_cube = cube_pos.copy()
        approach_cube[2] += APPROACH_HEIGHT

        grasp_cube = cube_pos.copy()

        lift_cube = cube_pos.copy()
        lift_cube[2] += LIFT_HEIGHT

        # -------- Container waypoints --------

        approach_container = CONTAINER_CENTER.copy()
        approach_container[2] = (
            TABLE_TOP_Z
            + CONTAINER_HEIGHT
            + DROP_SITE_HEIGHT
        )

        drop_container = CONTAINER_CENTER.copy()
        drop_container[2] = (
            TABLE_TOP_Z
            + CONTAINER_BASE
            + 0.055
        )

        retreat = drop_container.copy()
        retreat[2] += 0.18

        self.waypoints = [
            ("APPROACH CUBE", approach_cube, False),
            ("GRASP CUBE", grasp_cube, False),
            ("CLOSE GRIPPER", grasp_cube, True),
            ("LIFT CUBE", lift_cube, True),
            ("MOVE OVER CONTAINER", approach_container, True),
            ("LOWER INTO CONTAINER", drop_container, True),
            ("OPEN GRIPPER", drop_container, False),
            ("RETREAT", retreat, False),
            ("HOME", self.home_pos, False),
        ]

        self.target_q = []
        current_q = self.home_q.copy()

        for name, pos, _ in self.waypoints:
            q_target = solve_ik(
                model,
                current_q,
                pos,
                self.target_R,
                qpos_adr,
                dof_adr,
            )

            self.target_q.append(q_target.copy())
            current_q = q_target.copy()

        self.state = 0
        self.hold_start = None

        print("\n========== PICK AND PLACE ==========")
        print("Random cube:", self.cube_pos)
        print("Container:", CONTAINER_CENTER)
        print("====================================\n")

    def name(self):
        if self.state >= len(self.waypoints):
            return "DONE"
        return self.waypoints[self.state][0]

    def step(self):
        if self.state >= len(self.waypoints):
            set_gripper(self.data, False)
            return True

        name, target_pos, closed = self.waypoints[self.state]
        target_q = self.target_q[self.state]

        set_gripper(self.data, closed)

        apply_arm_pd(
            self.model,
            self.data,
            target_q,
            self.qpos_adr,
            self.dof_adr,
        )

        mujoco.mj_forward(self.model, self.data)

        current_pos = self.data.site(EE_SITE).xpos.copy()
        current_R = self.data.site(EE_SITE).xmat.reshape(3, 3)

        pos_error = np.linalg.norm(target_pos - current_pos)
        rot_error = np.linalg.norm(
            rotation_error(current_R, self.target_R)
        )

        reached = (
            pos_error < POSITION_TOL
            and rot_error < ORIENTATION_TOL
        )

        if reached:
            if self.hold_start is None:
                self.hold_start = self.data.time

            # Dwell briefly before changing state.
            if self.data.time - self.hold_start > 0.20:
                self.state += 1
                self.hold_start = None

                if self.state < len(self.waypoints):
                    print(
                        f"[{self.data.time:6.2f}s] -> "
                        f"{self.waypoints[self.state][0]}"
                    )
                else:
                    print(
                        f"[{self.data.time:6.2f}s] -> DONE"
                    )
        else:
            self.hold_start = None

        return False


# ============================================================
# VIEWER OVERLAY
# ============================================================

def update_overlay(viewer, data, task):
    pos = data.site(EE_SITE).xpos

    text = (
        "HEAL PICK & PLACE\n\n"
        f"TASK: {task.name()}\n\n"
        f"EE x = {pos[0]: .3f} m\n"
        f"EE y = {pos[1]: .3f} m\n"
        f"EE z = {pos[2]: .3f} m\n\n"
        f"CUBE x = {task.cube_pos[0]: .3f}\n"
        f"CUBE y = {task.cube_pos[1]: .3f}\n"
        f"CUBE z = {task.cube_pos[2]: .3f}"
    )

    viewer.set_texts(
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            text,
            "",
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    rng = np.random.default_rng()
    cube_pos = choose_cube_position(rng)

    print("Random cube spawn:", cube_pos)

    model = build_model(cube_pos)
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    qpos_adr, dof_adr = get_joint_addresses(model)

    if model.nu <= GRIPPER_ACTUATOR_ID:
        raise RuntimeError(
            "Robotiq actuator 6 was not found. "
            f"Model has {model.nu} actuators."
        )

    print("\nActuators:")
    for aid in range(model.nu):
        print(f"  {aid}: {model.actuator(aid).name}")

    task = PickPlaceTask(
        model,
        data,
        cube_pos,
        qpos_adr,
        dof_adr,
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:

        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.distance = 1.8
        viewer.cam.lookat[:] = [0.35, 0.05, 0.35]

        last_time = time.time()

        while viewer.is_running():

            # Update gravity/Coriolis bias before computing torque.
            mujoco.mj_forward(model, data)

            done = task.step()

            mujoco.mj_step(model, data)

            update_overlay(viewer, data, task)
            viewer.sync()

            now = time.time()
            elapsed = now - last_time

            if elapsed < 0.005:
                time.sleep(0.005 - elapsed)

            last_time = time.time()

            if done:
                # Hold the final pose with the gripper open.
                while viewer.is_running():

                    mujoco.mj_forward(model, data)

                    apply_arm_pd(
                        model,
                        data,
                        task.target_q[-1],
                        qpos_adr,
                        dof_adr,
                    )

                    set_gripper(data, False)

                    mujoco.mj_step(model, data)

                    update_overlay(viewer, data, task)
                    viewer.sync()

                    time.sleep(0.01)

                break


if __name__ == "__main__":
    main()