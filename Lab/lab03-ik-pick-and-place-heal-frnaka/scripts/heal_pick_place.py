#!/usr/bin/env python3
"""HEAL pick-and-place with a selectable IK method: Mink, DLS, or QP.

Run from the repository root:
  python Lab/lab03-ik-pick-and-place-heal-frnaka/scripts/HealPickPlace.py
Optional flags:
  --ik {mink,dls,qp} --seed N --no-viewer --no-randomize --no-random-yaw
If --ik is omitted, the script asks the user to choose interactively.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
LAB_DIR = SCRIPT_DIR.parent
for _p in (LAB_DIR, SCRIPT_DIR):  # SCRIPT_DIR ends up first, LAB_DIR second
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import heal_pick_place_scene as scene  # noqa: E402
from utils.mink_ik.mink_ik import MinkIK, downward_rotation
from utils.dls_ik.dls_ik import DLSIK, IKFailure as DLSIKFailure
from utils.qp_ik.qp_ik import QPIK, IKFailure as QPIKFailure
from utils.mj_velocity_control.mj_velocity_ctrl import JointVelocityController 

# Robot-specific configuration belongs in this task file, not in the shared IK utility.
HEAL_JOINT_NAMES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
)

IK_METHOD = "mink"
IK_METHOD_LABELS = {
    "mink": "Mink IK",
    "dls": "DLS IK",
    "qp": "QP IK",
}
MAX_CONSECUTIVE_IK_FAILURES = 50


class GuardedIK:
    """Count IK calls and hold the arm still during isolated custom-solver failures."""

    def __init__(self, solver, catch=()):
        self.solver = solver
        self.catch = tuple(catch)
        self.consecutive_failures = 0

    def compute_joint_velocities(self, target_position, target_rotation):
        try:
            velocities = self.solver.compute_joint_velocities(
                target_position, target_rotation
            )
            self.consecutive_failures = 0
            return velocities
        except self.catch as exc:
            self.consecutive_failures += 1
            if self.consecutive_failures >= MAX_CONSECUTIVE_IK_FAILURES:
                raise TaskError(
                    f"{IK_METHOD_LABELS[IK_METHOD]} failed for "
                    f"{self.consecutive_failures} consecutive IK updates: {exc}"
                ) from exc
            return np.zeros(len(HEAL_JOINT_NAMES), dtype=float)

# --------------------------------------------------------------------------------------
# Task parameters
# --------------------------------------------------------------------------------------
APPROACH_HEIGHT = 0.10          # m, TCP height above object reference point
# Even with the TCP measured at closure width, the lowest pad edge ends only ~0.4 mm above the
# tabletop when the TCP is exactly at Zc (pad boxes are ~20 mm tall, cube centre is 20 mm up).
# This extra height keeps the pads off the table while closing. Set 0.0 for exactly (Xc, Yc, Zc).
GRASP_Z_CLEARANCE = 0.003
GRIPPER_OPEN_CMD = 0.0          # same convention as example_controlling_gripper.py
GRIPPER_CLOSE_CMD = 255.0

CONTROL_DT = 0.01               # s, IK/controller update period (multiple of model timestep)
VELOCITY_KD = 50.0              # D gain of JointVelocityController (N*m*s/rad)
IK_GAIN = 0.3                   # FrameTask gain (per control period)
IK_DAMPING = 1e-3
MAX_JOINT_SPEED = 1.0           # rad/s, IK output limit (uniform scaling)
JOINT_HOLD_KP = 5.0             # 1/s, joint-space position hold / joint-space moves

SPEED_TRANSIT = 0.15            # m/s TCP speed, free-space moves
SPEED_PRECISE = 0.06            # m/s TCP speed, descend / lift
YAW_RATE = 0.5                  # rad/s peak tool-yaw rate (a hard wrist slew makes the cube slip)
SPEED_JOINT_MOVE = 0.8          # rad/s, joint-space moves (return to neutral)

POS_TOL = 0.004                 # m   TCP position tolerance at every IK target
TILT_TOL = np.deg2rad(2.0)      # rad tool-axis deviation from vertical
JOINT_TOL = 0.01                # rad joint-space move tolerance
SETTLE_SPEED = 0.02             # rad/s max joint speed counted as "settled"
SETTLE_TIME = 0.25              # s the settled condition must persist
MOVE_TIMEOUT_MARGIN = 8.0       # s allowed beyond the nominal duration of a move
OBJECT_SETTLE_TIMEOUT = 4.0     # s
GRIPPER_CLOSE_TIMEOUT = 3.0     # s
GRIPPER_OPEN_TIMEOUT = 2.0      # s
CUBE_LAND_TIMEOUT = 4.0         # s for the released cube to come to rest
YAW_TOL = np.deg2rad(1.0)       # rad, finger axis vs. cube faces must be parallel within this

CUBE_HALF = 0.02
CONTAINER_BASE_HALF_Z = 0.006
CONTAINER_INNER_HALF = 0.048    # inner cavity half-width (walls start at 0.048)


# Random placement (both objects stay fully on the table, with these edge margins)
CUBE_EDGE_MARGIN = 0.03         # m, keeps the open fingers (~+-0.06 m around the cube) over the table
CONTAINER_EDGE_MARGIN = 0.01    # m
# Required horizontal distance from the cube centre to the nearest edge of the container. The open
# gripper reaches ~0.06 m from the cube centre (fingers/followers) and its yaw varies with joint_1,
# so a circular zone is used; 0.10 m includes ~4 cm of safety margin.
GRIPPER_CLEARANCE = 0.10
YAW_RANGE_DEG = 90.0            # random object yaw is uniform in +-this (squares repeat every 90 deg)
PLACE_YAW_TOL = np.deg2rad(3.0) # cube must rest parallel to the container walls within this


class TaskError(RuntimeError):
    """A stage failed; message says which and why."""


# --------------------------------------------------------------------------------------
# Model helpers
# --------------------------------------------------------------------------------------
def find_by_suffix(model, obj_type, suffix):
    """Id of the single object whose (possibly attach-prefixed) name ends with `suffix`."""
    count = {
        mujoco.mjtObj.mjOBJ_BODY: model.nbody,
        mujoco.mjtObj.mjOBJ_GEOM: model.ngeom,
        mujoco.mjtObj.mjOBJ_JOINT: model.njnt,
        mujoco.mjtObj.mjOBJ_ACTUATOR: model.nu,
    }[obj_type]
    hits = []
    for i in range(count):
        name = mujoco.mj_id2name(model, obj_type, i)
        if name and (name == suffix or name.endswith("/" + suffix)):
            hits.append(i)
    if len(hits) != 1:
        raise TaskError(f"Expected exactly one {suffix!r} in model, found {len(hits)}")
    return hits[0]


def _wrap90(angle_deg):
    """Wrap an angle to [-45, 45) deg (square symmetry)."""
    return ((angle_deg + 45.0) % 90.0) - 45.0


def container_gap(cube_x, cube_y, box_x, box_y, box_yaw_deg=0.0):
    """Horizontal distance from the cube centre to the nearest edge of the (rotated) container."""
    yaw = np.deg2rad(box_yaw_deg)
    dx, dy = cube_x - box_x, cube_y - box_y
    lx = np.cos(yaw) * dx + np.sin(yaw) * dy      # cube centre in the container frame
    ly = -np.sin(yaw) * dx + np.cos(yaw) * dy
    gx = max(abs(lx) - scene.CONTAINER_HALF_X, 0.0)
    gy = max(abs(ly) - scene.CONTAINER_HALF_Y, 0.0)
    return float(np.hypot(gx, gy))


def _yaw_extent(half_x, half_y, yaw_deg):
    """Half-extent of a rotated rectangle's axis-aligned bounding box."""
    c, s = abs(np.cos(np.deg2rad(yaw_deg))), abs(np.sin(np.deg2rad(yaw_deg)))
    return half_x * c + half_y * s, half_x * s + half_y * c


def randomize_layout(model, data, rng, randomize_yaw=True):
    """Random cube/container pose on the table with gripper clearance between them.

    Table bounds and height come from the compiled table geometry (via the scene module). The
    rotated footprints must stay fully on the table (with edge margins). Objects are dropped 1 mm
    above the table and settle under gravity and contact.
    """
    cube_joint = scene.find_id_by_suffix(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    box_joint = scene.find_id_by_suffix(model, mujoco.mjtObj.mjOBJ_JOINT, "container_free")
    mujoco.mj_forward(model, data)
    x0, x1, y0, y1 = scene.table_bounds(model, data)
    top_z = scene.table_top_height(model, data)

    for attempt in range(1, scene.MAX_SAMPLING_ATTEMPTS + 1):
        cyaw = rng.uniform(-YAW_RANGE_DEG, YAW_RANGE_DEG) if randomize_yaw else 0.0
        byaw = rng.uniform(-YAW_RANGE_DEG, YAW_RANGE_DEG) if randomize_yaw else 0.0
        chx, chy = _yaw_extent(scene.CUBE_HALF_X, scene.CUBE_HALF_Y, cyaw)
        bhx, bhy = _yaw_extent(scene.CONTAINER_HALF_X, scene.CONTAINER_HALF_Y, byaw)
        cx = rng.uniform(x0 + chx + CUBE_EDGE_MARGIN, x1 - chx - CUBE_EDGE_MARGIN)
        cy = rng.uniform(y0 + chy + CUBE_EDGE_MARGIN, y1 - chy - CUBE_EDGE_MARGIN)
        bx = rng.uniform(x0 + bhx + CONTAINER_EDGE_MARGIN, x1 - bhx - CONTAINER_EDGE_MARGIN)
        by = rng.uniform(y0 + bhy + CONTAINER_EDGE_MARGIN, y1 - bhy - CONTAINER_EDGE_MARGIN)
        gap = container_gap(cx, cy, bx, by, byaw)
        if gap < GRIPPER_CLEARANCE:
            continue
        scene.set_free_body_pose(model, data, cube_joint, cx, cy,
                                 top_z + scene.CUBE_HALF_Z + scene.INITIAL_CLEARANCE_Z,
                                 np.deg2rad(cyaw))
        scene.set_free_body_pose(model, data, box_joint, bx, by,
                                 top_z + scene.CONTAINER_BASE_HALF_Z + scene.INITIAL_CLEARANCE_Z,
                                 np.deg2rad(byaw))
        mujoco.mj_forward(model, data)
        print("Randomized placement (table-relative limits respected):")
        print(f"  Cube XY:       ({cx:.3f}, {cy:.3f}) m, yaw {cyaw:+.1f} deg")
        print(f"  Container XY:  ({bx:.3f}, {by:.3f}) m, yaw {byaw:+.1f} deg")
        print(f"  Cube centre -> container edge: {gap * 100:.1f} cm "
              f"(required >= {GRIPPER_CLEARANCE * 100:.0f} cm) | sampling attempts: {attempt}")
        return
    raise TaskError(f"No cube/container layout with {GRIPPER_CLEARANCE * 100:.0f} cm gripper clearance "
                    f"found in {scene.MAX_SAMPLING_ATTEMPTS} attempts")


def build_world(randomize: bool, seed, randomize_yaw: bool = True):
    """Same scene + physics setup as heal_pick_place_scene.main(), reusing its functions."""
    model = scene.construct_scene()
    data = mujoco.MjData(model)
    model.opt.gravity[:] = (0.0, 0.0, -9.81)
    for geom in ("table_top", "cube_geom", "container_base"):
        scene.set_contact_friction(model, geom, (1.0, 0.005, 0.0001))
    mujoco.mj_forward(model, data)
    if randomize:
        randomize_layout(model, data, np.random.default_rng(seed), randomize_yaw)
    return model, data


# --------------------------------------------------------------------------------------
# Task runner
# --------------------------------------------------------------------------------------
class PickPlace:
    def __init__(self, model, data, viewer=None, realtime=True):
        self.model, self.data = model, data
        self.viewer, self.realtime = viewer, realtime

        # ---- timing: use the actual model timestep ----
        self.n_sub = max(1, int(round(CONTROL_DT / model.opt.timestep)))
        self.dt = self.n_sub * model.opt.timestep

        # ---- arm / gripper mapping (verified by name, not assumed) ----
        self.arm_joint_ids = [find_by_suffix(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                              for n in HEAL_JOINT_NAMES]
        self.arm_qadr = np.array([model.jnt_qposadr[j] for j in self.arm_joint_ids])
        self.arm_dofadr = np.array([model.jnt_dofadr[j] for j in self.arm_joint_ids])
        self.arm_act = []
        for j, name in zip(self.arm_joint_ids, HEAL_JOINT_NAMES):
            acts = [a for a in range(model.nu)
                    if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
                    and model.actuator_trnid[a, 0] == j]
            if len(acts) != 1:
                raise TaskError(f"{name}: expected one actuator, found {len(acts)}")
            self.arm_act.append(acts[0])
        self.grip_act = find_by_suffix(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
        if self.arm_act != list(range(6)) or self.grip_act != 6:
            raise TaskError(f"Unexpected actuator order: arm={self.arm_act}, gripper={self.grip_act} "
                            "(example_controlling_gripper.py assumes arm 0-5, gripper 6)")
        self.grip_cmd = GRIPPER_OPEN_CMD
        self._q_hold = None
        self.grip_driver_qadr = [model.jnt_qposadr[find_by_suffix(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                                 for n in ("left_driver_joint", "right_driver_joint")]

        # ---- objects ----
        self.cube_body = find_by_suffix(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.box_body = find_by_suffix(model, mujoco.mjtObj.mjOBJ_BODY, "container")
        self.cube_geom = find_by_suffix(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        # Arm + gripper bodies are everything compiled between base_link and the table body.
        self.robot_bodies = set(range(find_by_suffix(model, mujoco.mjtObj.mjOBJ_BODY, "base_link"),
                                      find_by_suffix(model, mujoco.mjtObj.mjOBJ_BODY, "table")))
        self.left_pads = {find_by_suffix(model, mujoco.mjtObj.mjOBJ_GEOM, n)
                          for n in ("left_pad1", "left_pad2")}
        self.right_pads = {find_by_suffix(model, mujoco.mjtObj.mjOBJ_GEOM, n)
                           for n in ("right_pad1", "right_pad2")}
        cube_joint = find_by_suffix(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        self.cube_qadr = int(model.jnt_qposadr[cube_joint])
        self.cube_dadr = int(model.jnt_dofadr[cube_joint])

        # ---- velocity controller (existing class): gravity comp + D term; gripper untouched ----
        self.controller = JointVelocityController(model, data, kd=VELOCITY_KD,
                                                  gripper_ids=[self.grip_act])
        # Joint damping (55 N*m*s/rad in the XML) dominates the D term, so a raw velocity
        # target would only be ~half realised. Feed forward the damping: target' = v*(1+b/kd)
        # gives steady-state joint speed == requested speed.
        self.ff_scale = 1.0 + model.dof_damping[self.arm_dofadr] / VELOCITY_KD

        # ---- IK: choose one solver while keeping the pick-and-place logic unchanged ----
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ft_site_eff")
        if IK_METHOD == "mink":
            solver = MinkIK(
                model, data, joint_names=HEAL_JOINT_NAMES,
                site_name="ft_site_eff", dt=self.dt,
                damping=IK_DAMPING, gain=IK_GAIN,
                max_joint_velocity=MAX_JOINT_SPEED,
            )
            self.ik = solver
        elif IK_METHOD == "dls":
            solver = DLSIK(
                model, data, joint_names=HEAL_JOINT_NAMES,
                site_name="ft_site_eff", dt=self.dt,
                damping=IK_DAMPING, gain=IK_GAIN,
                max_joint_velocity=MAX_JOINT_SPEED,
            )
            self.ik = GuardedIK(solver, catch=(DLSIKFailure,))
        elif IK_METHOD == "qp":
            solver = QPIK(
                model, data, joint_names=HEAL_JOINT_NAMES,
                site_name="ft_site_eff", solver="daqp", dt=self.dt,
                damping=IK_DAMPING, gain=IK_GAIN,
                max_joint_velocity=MAX_JOINT_SPEED,
            )
            self.ik = GuardedIK(solver, catch=(QPIKFailure,))
        else:
            raise ValueError(f"Unknown IK method: {IK_METHOD}")
        self.target_yaw = 0.0  # deg; set from the cube's yaw in choose_grasp_yaw()

        self.q_neutral = data.qpos[self.arm_qadr].copy()
        self.tcp_offset = self._measure_tcp_offset()
        self._wall0 = time.perf_counter()

    # ------------------------------------------------------------------ utilities
    def log(self, msg):
        print(f"[t={self.data.time:7.2f}s] {msg}", flush=True)

    def _measure_tcp_offset(self):
        """Grasp point (centroid of the 4 pad geoms) in the ft_site_eff frame.

        The 2F-85 pads travel ~10 mm along the tool axis while closing, so the offset is taken at
        the finger separation reached when closed on the cube (not with the gripper open). It is
        measured on a scratch MjData with gravity off, so the real simulation state is untouched.
        """
        model = self.model
        pads = list(self.left_pads | self.right_pads)
        pad1_l = find_by_suffix(model, mujoco.mjtObj.mjOBJ_GEOM, "left_pad1")
        pad1_r = find_by_suffix(model, mujoco.mjtObj.mjOBJ_GEOM, "right_pad1")
        target_gap = 2 * CUBE_HALF + 2 * model.geom_size[pad1_l][0]  # pad-centre separation

        saved_gravity = model.opt.gravity.copy()
        model.opt.gravity[:] = 0.0
        scratch = mujoco.MjData(model)
        gaps, centroids = [], []
        try:
            mujoco.mj_forward(model, scratch)
            for cmd in np.linspace(GRIPPER_OPEN_CMD, GRIPPER_CLOSE_CMD, 18):
                scratch.ctrl[self.grip_act] = cmd
                for _ in range(int(0.4 / model.opt.timestep)):
                    mujoco.mj_step(model, scratch)
                rot = scratch.site_xmat[self.site_id].reshape(3, 3)
                local = rot.T @ (np.mean([scratch.geom_xpos[g] for g in pads], axis=0)
                                 - scratch.site_xpos[self.site_id])
                gaps.append(float(np.linalg.norm(scratch.geom_xpos[pad1_l] - scratch.geom_xpos[pad1_r])))
                centroids.append(local)
        finally:
            model.opt.gravity[:] = saved_gravity
        gaps, centroids = np.array(gaps), np.array(centroids)
        if not (gaps.min() < target_gap < gaps.max()):
            raise TaskError(f"Cannot measure TCP: pad gap {target_gap * 1000:.0f} mm is outside "
                            f"the gripper stroke ({gaps.min() * 1000:.0f}-{gaps.max() * 1000:.0f} mm)")
        order = np.argsort(gaps)
        offset = np.array([np.interp(target_gap, gaps[order], centroids[order, k]) for k in range(3)])
        self.tcp_offset_open_z = float(centroids[order][-1, 2])
        return offset

    def site_pose(self):
        mujoco.mj_kinematics(self.model, self.data)
        return (self.data.site_xpos[self.site_id].copy(),
                self.data.site_xmat[self.site_id].reshape(3, 3).copy())

    def tcp(self):
        pos, rot = self.site_pose()
        return pos + rot @ self.tcp_offset

    def tilt(self):
        """Angle between tool axis and straight down (rad)."""
        _, rot = self.site_pose()
        return float(np.arccos(np.clip(-rot[2, 2], -1.0, 1.0)))

    def tool_yaw_deg(self):
        _, rot = self.site_pose()
        return float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))  # finger-closing axis

    def yaw_error(self):
        """|finger-axis yaw - target yaw| in rad, wrapped to [0, pi]."""
        d = np.deg2rad(self.tool_yaw_deg() - self.target_yaw)
        return float(abs(np.arctan2(np.sin(d), np.cos(d))))

    def q_arm(self):
        return self.data.qpos[self.arm_qadr].copy()

    def qd_arm(self):
        return self.data.qvel[self.arm_dofadr].copy()

    def cube_pos(self):
        return self.data.qpos[self.cube_qadr:self.cube_qadr + 3].copy()

    def cube_speed(self):
        return float(np.linalg.norm(self.data.qvel[self.cube_dadr:self.cube_dadr + 6]))

    def box_pos(self):
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.xpos[self.box_body].copy()

    def _body_yaw_deg(self, body):
        mujoco.mj_kinematics(self.model, self.data)
        rot = self.data.xmat[body].reshape(3, 3)
        return float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))

    def cube_yaw_deg(self):
        return self._body_yaw_deg(self.cube_body)

    def box_yaw_deg(self):
        return self._body_yaw_deg(self.box_body)

    # ------------------------------------------------------------------ stepping
    def _advance(self, arm_velocity):
        """Apply 6 arm joint-velocity targets (rad/s) for one control period."""
        v7 = np.zeros(self.model.nu)
        v7[self.arm_act] = np.asarray(arm_velocity) * self.ff_scale
        self.controller.set_velocity_target(v7)
        for _ in range(self.n_sub):
            self.data.ctrl[self.grip_act] = self.grip_cmd  # separate from arm commands
            mujoco.mj_step(self.model, self.data)
        self._check_container_collision()
        if self.viewer is not None:
            if not self.viewer.is_running():
                raise KeyboardInterrupt("Viewer closed")
            self.viewer.sync()
        if self.realtime:
            ahead = (self._wall0 + self.data.time) - time.perf_counter()
            if ahead > 0:
                time.sleep(ahead)

    def _check_container_collision(self):
        """Fail loudly if any arm/gripper geom touches the container (it must only hold the cube)."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if (b1 == self.box_body and b2 in self.robot_bodies) or \
                    (b2 == self.box_body and b1 in self.robot_bodies):
                names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
                         for g in (c.geom1, c.geom2)]
                raise TaskError(f"Robot collided with the container ({names[0]} <-> {names[1]})")

    def hold(self, seconds):
        q_hold = self.q_arm()
        for _ in range(int(round(seconds / self.dt))):
            self._advance(np.clip(JOINT_HOLD_KP * (q_hold - self.q_arm()),
                                  -MAX_JOINT_SPEED, MAX_JOINT_SPEED))

    # ------------------------------------------------------------------ motions
    def move_joints(self, label, q_goal, timeout=20.0):
        """Joint-space move with the velocity controller (no IK)."""
        q_goal = np.asarray(q_goal, dtype=float)
        t0, settled = self.data.time, 0.0
        while True:
            err = q_goal - self.q_arm()
            v = JOINT_HOLD_KP * err
            peak = np.max(np.abs(v)) / SPEED_JOINT_MOVE
            if peak > 1.0:
                v = v / peak
            self._advance(v)
            if np.max(np.abs(q_goal - self.q_arm())) < JOINT_TOL and \
                    np.max(np.abs(self.qd_arm())) < SETTLE_SPEED:
                settled += self.dt
                if settled >= SETTLE_TIME:
                    return
            else:
                settled = 0.0
            if self.data.time - t0 > timeout:
                raise TaskError(f"{label}: joint move timed out; max joint error "
                                f"{np.max(np.abs(q_goal - self.q_arm())):.4f} rad")

    def move_tcp(self, label, goal, speed):
        """Straight-line Cartesian move of the grasp point using iterative Mink IK."""
        goal = np.asarray(goal, dtype=float)
        start = self.tcp()
        vec = goal - start
        dist = float(np.linalg.norm(vec))
        # Yaw moves along the shortest rotation to the target yaw, on the same smooth profile.
        yaw0 = np.deg2rad(self.tool_yaw_deg())
        d_yaw = float(np.arctan2(np.sin(np.deg2rad(self.target_yaw) - yaw0),
                                 np.cos(np.deg2rad(self.target_yaw) - yaw0)))
        # Quintic (minimum-jerk) profile: peak rate = 1.875 x mean rate. Duration is set so the peak
        # TCP speed equals `speed` and the peak yaw rate equals YAW_RATE, whichever is slower.
        duration = 1.875 * max(dist / speed, abs(d_yaw) / YAW_RATE)
        t0, settled = self.data.time, 0.0
        self.log(f"{label}: TCP {np.round(start, 4)} -> {np.round(goal, 4)}  ({dist * 100:.1f} cm)")
        while True:
            tau = 1.0 if duration <= 0 else min(1.0, (self.data.time - t0) / duration)
            s = tau ** 3 * (10.0 - 15.0 * tau + 6.0 * tau ** 2)
            carrot = start + s * vec
            rot_target = downward_rotation(np.rad2deg(yaw0 + s * d_yaw))
            site_pos, site_rot = self.site_pose()
            site_target = carrot - site_rot @ self.tcp_offset  # TCP target -> flange target
            v = self.ik.compute_joint_velocities(site_target, rot_target)
            self._advance(v)

            pos_err = float(np.linalg.norm(goal - self.tcp()))
            tilt = self.tilt()
            yaw_err = self.yaw_error()
            if tau >= 1.0 and pos_err < POS_TOL and tilt < TILT_TOL and yaw_err < YAW_TOL and \
                    np.max(np.abs(self.qd_arm())) < SETTLE_SPEED:
                settled += self.dt
                if settled >= SETTLE_TIME:
                    self.log(f"{label}: reached (pos err {pos_err * 1000:.2f} mm, "
                             f"tilt {np.degrees(tilt):.2f} deg, yaw err {np.degrees(yaw_err):.2f} deg, "
                             f"joint_6 {np.degrees(self.q_arm()[5]):.2f} deg)")
                    return
            else:
                settled = 0.0
            if self.data.time - t0 > duration + MOVE_TIMEOUT_MARGIN:
                raise TaskError(
                    f"{label}: timed out. Target likely unreachable (or blocked) with the required "
                    f"finger yaw. pos err {pos_err * 1000:.1f} mm (tol {POS_TOL * 1000:.1f}), "
                    f"tilt {np.degrees(tilt):.2f} deg, yaw err {np.degrees(yaw_err):.2f} deg, "
                    f"q = {np.round(self.q_arm(), 3)}")

    # ------------------------------------------------------------------ gripper
    def _wait(self, label, predicate, timeout):
        t0 = self.data.time
        while not predicate():
            self.hold_step()
            if self.data.time - t0 > timeout:
                raise TaskError(f"{label}: timed out after {timeout:.1f}s")

    def hold_step(self):
        """One control period holding the arm still (joint-space hold at the latest pose)."""
        if self._q_hold is None:
            self._q_hold = self.q_arm()
        self._advance(np.clip(JOINT_HOLD_KP * (self._q_hold - self.q_arm()),
                              -MAX_JOINT_SPEED, MAX_JOINT_SPEED))

    def _driver_pos(self):
        return float(np.mean([self.data.qpos[a] for a in self.grip_driver_qadr]))

    def _driver_speed(self):
        return float(np.max(np.abs(self.data.qvel[[self.model.jnt_dofadr[
            find_by_suffix(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in ("left_driver_joint", "right_driver_joint")]])))

    def contacts_cube(self):
        left = right = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self.cube_geom:
                other = c.geom2
            elif c.geom2 == self.cube_geom:
                other = c.geom1
            else:
                continue
            left |= other in self.left_pads
            right |= other in self.right_pads
        return left, right

    def open_gripper(self, label="open gripper"):
        self.grip_cmd = GRIPPER_OPEN_CMD
        self._q_hold = self.q_arm()
        self.log(f"{label}: command {GRIPPER_OPEN_CMD:.0f}")
        self._wait(label, lambda: self._driver_pos() < 0.02 and self._driver_speed() < 0.02,
                   GRIPPER_OPEN_TIMEOUT)
        self.log(f"{label}: open (driver {self._driver_pos():.3f} rad)")

    def close_gripper(self, label="close gripper"):
        self.grip_cmd = GRIPPER_CLOSE_CMD
        self._q_hold = self.q_arm()
        self.log(f"{label}: command {GRIPPER_CLOSE_CMD:.0f}")
        # Wait until the fingers stop moving (blocked by the cube or at full closure).
        t_still = 0.0
        t0 = self.data.time
        while t_still < 0.3:
            self.hold_step()
            t_still = t_still + self.dt if self._driver_speed() < 0.01 else 0.0
            if self.data.time - t0 > GRIPPER_CLOSE_TIMEOUT:
                raise TaskError(f"{label}: fingers never came to rest")
        left, right = self.contacts_cube()
        drv = self._driver_pos()
        self.log(f"{label}: stopped (driver {drv:.3f} rad, pad contact left={left} right={right})")
        if not (left and right):
            raise TaskError("Grasp failed: both finger pads are not in contact with the cube "
                            f"(left={left}, right={right}, driver={drv:.3f} rad; ~0.82 means fully "
                            "closed on nothing).")

    # ------------------------------------------------------------------ stages
    def settle_objects(self):
        self.log("STAGE 0a: letting cube and container settle on the table")
        self._q_hold = self.q_arm()
        self._wait("object settle",
                   lambda: self.data.time > 0.5 and self.cube_speed() < 1e-3 and
                   float(np.linalg.norm(self.data.qvel[self.model.jnt_dofadr[
                       find_by_suffix(self.model, mujoco.mjtObj.mjOBJ_JOINT, "container_free")]:][:6])) < 1e-3,
                   OBJECT_SETTLE_TIMEOUT)
        cube = self.cube_pos()
        top = scene.table_top_height(self.model, self.data)
        if abs(cube[2] - (top + CUBE_HALF)) > 0.005:
            raise TaskError(f"Cube is not resting on the table (z={cube[2]:.4f}, expected "
                            f"{top + CUBE_HALF:.4f})")

    def choose_yaw(self, aligned_yaw_deg, xy, label):
        """Set the IK finger-axis yaw to aligned_yaw + n*90 deg (all four are equivalent).

        For a downward tool, yaw ~ base + joint_1 + joint_6, so the candidate needing the smallest
        joint_6 at the target location is used (predicted from the current joint angles); the IK
        then solves the actual joint_6 value.
        """
        q = self.q_arm()
        d_q1 = np.arctan2(xy[1], xy[0]) - q[0]
        best = None
        for n in range(4):
            yaw = aligned_yaw_deg + 90.0 * n
            d_yaw = np.deg2rad(yaw - self.tool_yaw_deg())
            d_yaw = np.arctan2(np.sin(d_yaw), np.cos(d_yaw))       # shortest rotation
            q6 = q[5] + d_yaw - d_q1
            q6 = float(np.arctan2(np.sin(q6), np.cos(q6)))
            if best is None or abs(q6) < abs(best[1]):
                best = (yaw, q6)
        self.target_yaw = float(best[0])
        self.log(f"{label}: finger-axis target yaw {self.target_yaw:+.2f} deg "
                 f"(predicted joint_6 {np.degrees(best[1]):+.1f} deg)")

    def run(self):
        self.log(f"Control period {self.dt * 1000:.1f} ms ({self.n_sub} x model dt "
                 f"{self.model.opt.timestep * 1000:.1f} ms) | TCP offset in flange frame "
                 f"{np.round(self.tcp_offset, 4)} m (measured at cube-closure width; "
                 f"gripper-open value would be z={self.tcp_offset_open_z:.4f})")
        self.grip_cmd = GRIPPER_OPEN_CMD
        self.log(f"Initial arm config q = {np.round(self.q_neutral, 4)} | gripper open")
        self.settle_objects()

        # ---- STAGE 1: read cube, move above it ----
        cx, cy, cz = self.cube_pos()
        cube_yaw = self.cube_yaw_deg()
        self.log(f"STAGE 0b: cube yaw {cube_yaw:+.2f} deg -> plates must be parallel to its faces")
        self.choose_yaw(cube_yaw, (cx, cy), "STAGE 0b")
        self.log(f"STAGE 1: cube at (Xc, Yc, Zc) = ({cx:.4f}, {cy:.4f}, {cz:.4f})")
        above_cube = np.array([cx, cy, cz + APPROACH_HEIGHT])
        self.move_tcp("move above cube", above_cube, SPEED_TRANSIT)
        mis = ((self.tool_yaw_deg() - self.cube_yaw_deg() + 45.0) % 90.0) - 45.0
        self.log(f"  finger plates vs cube faces: {mis:+.2f} deg off parallel "
                 f"(joint_6 = {np.degrees(self.q_arm()[5]):+.1f} deg)")
        if abs(np.deg2rad(mis)) > YAW_TOL:
            raise TaskError(f"Finger plates not parallel to cube faces ({mis:+.2f} deg); aborting "
                            "before descent.")

        # ---- STAGE 2: descend ----
        self.log(f"STAGE 2: descend to cube (grasp point Zc + {GRASP_Z_CLEARANCE * 1000:.1f} mm clearance)")
        self.move_tcp("descend", np.array([cx, cy, cz + GRASP_Z_CLEARANCE]), SPEED_PRECISE)

        # ---- STAGE 3: close ----
        self.log("STAGE 3: grasp")
        self.close_gripper()

        # ---- STAGE 4: lift ----
        self.log("STAGE 4: lift")
        self.move_tcp("lift", above_cube, SPEED_PRECISE)
        held = self.cube_pos() - self.tcp()
        if np.linalg.norm(held[:2]) > 0.02 or abs(held[2]) > 0.02 or \
                self.cube_pos()[2] < cz + 0.5 * APPROACH_HEIGHT:
            raise TaskError(f"Grasp lost during lift: cube offset from TCP = {np.round(held, 4)}, "
                            f"cube z = {self.cube_pos()[2]:.4f}")
        self.log(f"  cube held (offset from TCP {np.round(held, 4)} m)")

        # ---- STAGE 5: transfer ----
        bx, by, bz = self.box_pos()
        self.log(f"STAGE 5: container at (Xb, Yb, Zb) = ({bx:.4f}, {by:.4f}, {bz:.4f})")
        # Cube yaw relative to the finger axis (small slip during grasp/lift), mod 90 deg. The tool
        # yaw is chosen so that the cube's actual yaw ends up parallel to the container walls.
        in_hand = _wrap90(self.cube_yaw_deg() - self.tool_yaw_deg())
        box_yaw = self.box_yaw_deg()
        self.log(f"  container yaw {box_yaw:+.2f} deg, cube yaw in hand {in_hand:+.2f} deg off finger axis")
        self.choose_yaw(box_yaw - in_hand, (bx, by), "  drop yaw")
        carry_z = above_cube[2]
        self.move_tcp("transfer (horizontal)", np.array([bx, by, carry_z]), SPEED_TRANSIT)
        self.move_tcp("align to drop height", np.array([bx, by, bz + APPROACH_HEIGHT]), SPEED_PRECISE)
        if np.linalg.norm(self.cube_pos() - self.tcp()) > 0.03:
            raise TaskError("Cube slipped out of the gripper during transfer.")

        # ---- STAGE 6: release ----
        self.log("STAGE 6: release")
        self.open_gripper("release")
        self.log("  waiting for cube to come to rest")
        self._wait("cube landing", lambda: self.cube_speed() < 1e-3, CUBE_LAND_TIMEOUT)
        self.hold(0.3)

        # ---- verify placement ----
        ok, report = self.verify_placement()
        self.log("PLACEMENT " + ("VERIFIED: " if ok else "FAILED: ") + report)

        # ---- STAGE 7: return ----
        self.log("STAGE 7: return to initial/neutral configuration")
        self.move_joints("return to neutral", self.q_neutral)
        self.log(f"  final q = {np.round(self.q_arm(), 4)} "
                 f"(max error {np.max(np.abs(self.q_arm() - self.q_neutral)):.4f} rad)")
        if not ok:
            raise TaskError("Cube was NOT placed inside the container: " + report)

    def verify_placement(self):
        cube, box = self.cube_pos(), self.box_pos()
        yaw_b = np.deg2rad(self.box_yaw_deg())
        dx, dy = cube[0] - box[0], cube[1] - box[1]
        lx = np.cos(yaw_b) * dx + np.sin(yaw_b) * dy       # cube offset in the container frame
        ly = -np.sin(yaw_b) * dx + np.cos(yaw_b) * dy
        rest_z = box[2] + CONTAINER_BASE_HALF_Z + CUBE_HALF
        dz = cube[2] - rest_z
        rel_yaw = _wrap90(self.cube_yaw_deg() - self.box_yaw_deg())
        inside_xy = abs(lx) <= CONTAINER_INNER_HALF - CUBE_HALF and \
            abs(ly) <= CONTAINER_INNER_HALF - CUBE_HALF
        on_base = abs(dz) < 0.01
        parallel = abs(np.deg2rad(rel_yaw)) <= PLACE_YAW_TOL
        report = (f"cube ({cube[0]:.3f}, {cube[1]:.3f}, {cube[2]:.3f}) vs container "
                  f"({box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}): offset in container frame "
                  f"({lx * 1000:+.1f}, {ly * 1000:+.1f}) mm, dz_from_base_rest={dz * 1000:+.1f} mm, "
                  f"yaw vs container walls {rel_yaw:+.2f} deg (tol {np.degrees(PLACE_YAW_TOL):.0f})")
        if not parallel:
            report += " -> NOT parallel to container sides"
        return bool(inside_xy and on_base and parallel), report


# --------------------------------------------------------------------------------------
def main():
    global IK_METHOD

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ik", choices=["mink", "dls", "qp"], default=None,
                    help="IK method: mink, dls, or qp (if omitted, prompt interactively)")
    ap.add_argument("--no-viewer", action="store_true", help="run headless (also runs faster than real time)")
    ap.add_argument("--no-randomize", action="store_true",
                    help="keep the default XML object positions instead of random placement")
    ap.add_argument("--no-random-yaw", action="store_true",
                    help="random positions only; cube and container yaw stay 0")
    ap.add_argument("--seed", type=int, default=None, help="random-layout seed (printed if omitted)")
    args = ap.parse_args()

    if args.ik is None:
        print("\nChoose the inverse-kinematics method:")
        print("  1. Mink IK")
        print("  2. DLS IK")
        print("  3. QP IK")
        while True:
            choice = input("Enter 1, 2, or 3: ").strip()
            method_by_choice = {"1": "mink", "2": "dls", "3": "qp"}
            if choice in method_by_choice:
                IK_METHOD = method_by_choice[choice]
                break
            print("Invalid choice. Please enter 1, 2, or 3.")
    else:
        IK_METHOD = args.ik

    print(f"Selected method: {IK_METHOD_LABELS[IK_METHOD]}")
    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % 2**31)
    if not args.no_randomize:
        print(f"Layout seed: {seed}  (rerun with --seed {seed} to reproduce)")

    print(f"Building HEAL pick-and-place scene using {IK_METHOD_LABELS[IK_METHOD]}...")
    model, data = build_world(not args.no_randomize, seed, not args.no_random_yaw)
    viewer_cm = None
    viewer = None
    if not args.no_viewer:
        viewer_cm = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
        viewer = viewer_cm.__enter__()

    exit_code = 0
    runner = None
    try:
        runner = PickPlace(model, data, viewer=viewer, realtime=viewer is not None)
        mujoco.set_mjcb_control(runner.controller.control_callback)  # gravity comp + D control
        runner.run()
        runner.log("EPISODE COMPLETE: cube placed in container and robot returned to neutral.")
    except KeyboardInterrupt as exc:
        print(f"\nInterrupted: {exc}")
        exit_code = 130
    except TaskError as exc:
        print(f"\nEPISODE FAILED: {exc}", file=sys.stderr)
        exit_code = 1
    except Exception as exc:  # IK solver failures etc. must not be reported as success
        print(f"\nEPISODE ABORTED by unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        exit_code = 2
    finally:
        mujoco.set_mjcb_control(None)

    if viewer_cm is not None:
        if exit_code in (0, 1, 2):
            print("Close the viewer window to exit.")
            try:
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.05)
            except KeyboardInterrupt:
                pass
        viewer_cm.__exit__(None, None, None)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
