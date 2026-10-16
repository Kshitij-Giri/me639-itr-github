#!/usr/bin/env python3
"""Franka Panda pick-and-place workspace analysis and table optimization.

This script replaces the earlier "IK-reachability-only" Franka workspace script.
It keeps the HEAL workflow (grid sampling -> per-height analysis -> largest
usable rectangle -> plots + summary) but scores tabletop cells by *task*
feasibility of a downward cube pick-and-place, not by wrist reachability.

Feasibility levels recorded for every sampled cell (all world frame):

  L0  ik_grasp_reachable   Mink IK reaches the grasp TCP pose (downward, some
                           tested yaw) within tolerance and inside joint limits.
                           This is the old HEAL-style metric, kept for comparison.
  L1  pick_ok              Full pick chain verified kinematically + geometrically:
                           pre-grasp IK, straight vertical approach (IK-tracked,
                           1 cm steps, joint-jump bounded), finger closure to
                           the cube faces (pad/cube contact geometry verified),
                           vertical lift with the cube attached, with collision
                           (signed-distance) checks at every waypoint.
  L1  place_ok             Same for placing into a container centred on the cell:
                           pre-place IK, vertical descent of the held cube into
                           the container, finger opening inside the walls,
                           vertical retract.
  L2  validated_*          L1 chains re-run against the FULL candidate table
                           (top + legs) and the full robot collision geometry.
  L3  pair checks          For sampled (cube cell, container cell) pairs: pick
                           path re-checked with the container present, a
                           joint-space transfer (with held cube) collision-checked
                           at fine joint increments, and the place path.
  L4  physics replay       A few pairs are replayed in MuJoCo physics: the arm
                           tracks the planned joint path (PD + gravity comp.),
                           the gripper actuator closes/opens, and the cube must
                           end inside the container.

Collision checking uses signed distances (mujoco.mj_geomDistance, with a GJK
re-check of exact-zero answers, see Context.gdist) on the Panda collision geoms
(group 3: link meshes [convex hulls], finger meshes, fingertip pad boxes, the
hand mesh and the hand capsule).  NOTE: in mjx_panda_new.xml most robot
collision geoms have contype=conaffinity=0, so MuJoCo's own contact pipeline
would NOT report arm/table collisions; this script therefore measures pairwise
signed distances explicitly instead of relying on data.contact.

Intended vs unwanted contact:
  * finger geoms <-> cube are allowed to touch (small penetration tolerance)
    only while the gripper is closing on / holding the cube;
  * cube <-> tabletop contact is allowed only while the cube is resting on
    the table (grasp and the start of the lift);
  * everything else must keep a positive clearance margin.

Limitations (also written to the summary file):
  * Geometry is verified only at the sampled grid points and waypoints; the
    continuous region between samples is assumed by interpolation.
  * Panda self-collision is checked only for distal (link6, link7, hand,
    fingers) vs proximal (link0-link3) pairs that are separated at the home pose.
  * Mesh distances use MuJoCo's convex-hull mesh collision (conservative).
  * Cube yaw is assumed aligned to the tested grasp yaws (0/90/180/-90 deg); a
    cube with arbitrary yaw needs the corresponding wrist rotation.
  * The transfer is a joint-space straight line (with a raised via-point
    fallback), not a general motion planner.
  * The physics replay uses a joint PD + gravity-compensation tracker; it
    validates grasp/contact behaviour, not the lab's own controller.

table_fr.xml is NOT modified; recommended parameters and a ready-to-paste XML
snippet are written to the summary file.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import math
import multiprocessing as mp
import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import mujoco
import mink

# --------------------------------------------------------------------------
# Paths (robust to the current working directory)
# --------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
LAB_DIR = SCRIPT_DIR.parent
DESC = LAB_DIR / "robot-descriptions"
RESULTS = LAB_DIR / "results"
ARM_XML = DESC / "franka" / "mjx_panda_new.xml"
TABLE_XML = DESC / "table_fr.xml"
CUBE_XML = DESC / "cube.xml"
CONTAINER_XML = DESC / "container.xml"

# --------------------------------------------------------------------------
# Model names (verified against mjx_panda_new.xml / cube.xml / container.xml /
# table_fr.xml, and the prefixes used by franka_pick_place_scene.py)
# --------------------------------------------------------------------------
SITE_NAME = "gripper"                      # site on body "hand", pos 0 0 0.1
HAND_BODY = "hand"
BASE_BODY = "link0"
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["finger_joint1", "finger_joint2"]   # coupled by <equality>
FINGER_BODIES = ("left_finger", "right_finger")
PAD_GEOMS = ("left_finger_pad", "right_finger_pad")  # only finger geoms with
#   non-zero conaffinity (3) -> the only finger surfaces that make physical
#   contact with the cube in the lab scene.
GRIPPER_ACTUATOR = "gripper"   # general actuator on finger_joint1, ctrl 0..0.04
PROXIMAL_BODIES = ("link0", "link1", "link2", "link3")
DISTAL_SELF_BODIES = ("link6", "link7", "hand", "left_finger", "right_finger")
SCREEN_BODIES = ("link4", "link5", "link6", "link7", "hand",
                 "left_finger", "right_finger")
FLOOR_EXEMPT_BODIES = ("link0", "link1")   # base sits on the floor

TABLE_PREFIX, CUBE_PREFIX, CONTAINER_PREFIX = (
    "table_component/", "cube_component/", "container_component/")

# Panda "ready" pose.  qpos0 is NOT usable: joint4=0 lies outside its range
# [-3.0718, -0.0698] in the XML, so it is never used as an IK seed.
Q_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
FINGER_OPEN = 0.04             # finger_joint range 0..0.04 (each finger)

# --------------------------------------------------------------------------
# Object geometry (read from the XML files at start-up and cross-checked)
# --------------------------------------------------------------------------
CUBE_EDGE_MARGIN = 0.03        # same placement rules as franka_pick_place_scene.py
CONTAINER_EDGE_MARGIN = 0.01
CUBE_CONTAINER_GAP = 0.10      # scene GRIPPER_CLEARANCE
TABLE_HALF_THICK = 0.02        # from table_fr.xml
LEG_HALF_XY = 0.025
LEG_INSET = (0.035, 0.040)     # table_fr.xml: 0.225-0.190, 0.410-0.370

# --------------------------------------------------------------------------
# Task parameters
# --------------------------------------------------------------------------
GRASP_YAWS_DEG = (0.0, 90.0, -90.0, 180.0)   # cube symmetric under 90 deg
APPROACH_HEIGHT = 0.10         # pre-grasp TCP height above grasp TCP
LIFT_HEIGHT = 0.12             # cube bottom 12 cm above table after lift
PLACE_APPROACH = 0.10          # pre-place TCP height above release TCP
RELEASE_GAP = 0.005            # cube bottom above container base at release
FINGER_WALL_CLEARANCE = 0.006  # lowest finger point above wall top at release
MIN_CUBE_INSIDE_WALLS = 0.003  # cube bottom must be below wall top by this much
FINGERTIP_TABLE_CLEARANCE = 0.005
MIN_PAD_OVERLAP = 0.010        # vertical pad/cube-face overlap required
OPEN_SIDE_CLEARANCE = 0.008    # per side, open fingers vs cube faces
PATH_STEP = 0.01               # Cartesian waypoint spacing (m)
FINGER_STEPS = 5
TRANSFER_JOINT_STEP = 0.04     # rad, max joint change between checked configs
TRANSFER_VIA_RISE = 0.15

# Clearance margins (m)
ROBOT_MARGIN = 0.005           # robot vs table/legs/container/floor/cube
CUBE_MARGIN = 0.003            # held cube vs table/container
FINGER_CUBE_TOL = 0.003        # allowed finger/cube penetration while grasping
#   (finger mesh convex hull reaches 1.5 mm past the pad face)
PAD_CONTACT_TOL = 0.002        # pads must be within this of the cube faces
SELF_MARGIN = 0.0

# IK
IK_DT = 0.1
IK_GAIN = 1.0
IK_LM_DAMPING = 1e-3
IK_DAMPING = 1e-3
IK_POSTURE_COST = 1e-3
IK_MAX_STEP = 0.3              # rad, per-iteration step clip
IK_ITERS_KEY = 200
IK_ITERS_TRACK = 40
POS_TOL = 0.002                # m   (old script: 15 mm - too loose to grasp)
ORI_TOL = 0.02                 # rad (old script: 0.12 rad)
MAX_JOINT_JUMP = 0.25          # rad between 1 cm waypoints (branch flip guard)
MAX_REACH_BOUND = 1.20         # sum of link lengths + TCP; quick reject only
SOLVER = "daqp"

# Screening patch (local tabletop used before the table size is known)
PATCH_HALF = 0.15

# Optimization defaults
DEFAULTS = dict(
    x_min=0.10, x_max=0.85, y_min=-0.60, y_max=0.60, xy_step=0.05,
    z_min=0.05, z_max=0.45, z_coarse=0.10, z_fine=0.05,
    min_table_side=0.30, transfer_pairs=40, physics_pairs=4,
    validation_rounds=6, seed=639,
)


# ==========================================================================
# Small math helpers
# ==========================================================================
def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def downward_rotation(yaw_rad):
    """Site/hand orientation with hand +Z (finger approach axis) along world -Z.

    diag(1,-1,-1) is a proper rotation (pi about X).  Hand +Y (the finger
    sliding axis in the XML) then lies in the horizontal plane, so yaw sets the
    closing direction; the cube faces are normal to the closing axis when the
    yaw is a multiple of 90 deg.
    """
    return rotz(yaw_rad) @ np.diag([1.0, -1.0, -1.0])


def rot_angle(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(math.acos(max(-1.0, min(1.0, c))))


def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).reshape(9))
    return q   # w, x, y, z (MuJoCo order)


def wrap_into_range(a, lo, hi):
    """Return an angle equivalent to a (mod 2pi) inside [lo, hi], else clip."""
    best = None
    for k in range(-2, 3):
        v = a + 2.0 * math.pi * k
        if lo <= v <= hi and (best is None or abs(v) < abs(best)):
            best = v
    return float(np.clip(a, lo, hi)) if best is None else best


# ==========================================================================
# File checks and XML-derived object geometry
# ==========================================================================
def require_files():
    missing = [str(p) for p in (ARM_XML, TABLE_XML, CUBE_XML, CONTAINER_XML)
               if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required file(s):\n  " + "\n  ".join(missing))
    if not (ARM_XML.parent / "assets").is_dir():
        raise FileNotFoundError(
            f"Franka mesh directory not found: {ARM_XML.parent / 'assets'} "
            "(meshdir='assets' in mjx_panda_new.xml is relative to the XML).")


def read_object_geometry():
    """Read cube/container dimensions from their XML files."""
    def geom_size(path, name):
        root = ET.parse(path).getroot()
        g = root.find(f".//geom[@name='{name}']")
        if g is None:
            raise ValueError(f"geom '{name}' not found in {path.name}")
        return np.fromstring(g.get("size"), sep=" "), np.fromstring(
            g.get("pos", "0 0 0"), sep=" ")

    cube_half, _ = geom_size(CUBE_XML, "cube_geom")
    base_half, _ = geom_size(CONTAINER_XML, "container_base")
    wall_half, wall_pos = geom_size(CONTAINER_XML, "container_wall_x_pos")
    wall_y_half, _ = geom_size(CONTAINER_XML, "container_wall_y_pos")
    geo = dict(
        cube_half=float(cube_half[0]),
        cont_half=float(base_half[0]),
        cont_base_half_z=float(base_half[2]),
        wall_thick=float(2 * wall_half[1]),
        wall_top_local=float(wall_pos[2] + wall_half[2]),
    )
    if not np.allclose(cube_half, cube_half[0]):
        raise ValueError("cube_geom is not a cube; this analysis assumes a cube.")
    geo["cont_inner_half"] = geo["cont_half"] - geo["wall_thick"]
    if abs(wall_y_half[0] * 2 - geo["wall_thick"]) > 1e-9:
        raise ValueError("Container walls have unequal thickness.")
    return geo


# ==========================================================================
# Models
# ==========================================================================
def build_ik_model():
    """Panda-only model for Mink (the scene model has free joints)."""
    return mujoco.MjSpec.from_file(str(ARM_XML)).compile()


def table_layout(cx, cy, top_z, lx, ly):
    """Table body/geom parameters in table_fr.xml conventions.

    Body origin at the tabletop centre (z = top - half thickness); legs run
    from the floor (z=0) up to the body origin, inset as in table_fr.xml.
    """
    th = TABLE_HALF_THICK
    leg_half_z = max(1e-4, (top_z - th) / 2.0)
    legs = []
    for sx, sy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        legs.append(((sx * (lx / 2 - LEG_INSET[0]), sy * (ly / 2 - LEG_INSET[1]), -leg_half_z),
                     (LEG_HALF_XY, LEG_HALF_XY, leg_half_z)))
    return dict(body_pos=(cx, cy, top_z - th), top_size=(lx / 2.0, ly / 2.0, th), legs=legs)


def build_scene_model(table=None):
    """Panda + table + cube + container, assembled like franka_pick_place_scene.

    If table=(cx, cy, top_z, lx, ly) is given, the table is resized in the spec
    before compiling.  This is required for physics: MuJoCo builds a static
    mid-phase BVH for the (joint-less) table body at compile time, so editing
    geom sizes/positions on a compiled model would leave contact detection
    using stale bounds.  (Signed-distance queries via mj_geomDistance do not
    use that BVH, so the geometric checker may resize the table in place.)
    """
    spec = mujoco.MjSpec.from_file(str(ARM_XML))
    for path, prefix, fname in ((TABLE_XML, TABLE_PREFIX, "table_frame"),
                                (CUBE_XML, CUBE_PREFIX, "cube_frame"),
                                (CONTAINER_XML, CONTAINER_PREFIX, "container_frame")):
        comp = mujoco.MjSpec.from_file(str(path))
        frame = spec.worldbody.add_frame(name=fname)
        spec.attach(comp, prefix=prefix, frame=frame)
    if table is not None:
        lay = table_layout(*table)
        body = spec.body(TABLE_PREFIX + "table")
        if body is None:
            raise ValueError("table body not found in spec")
        body.pos = lay["body_pos"]
        spec.geom(TABLE_PREFIX + "table_top").size = lay["top_size"]
        for k, (pos, size) in enumerate(lay["legs"], 1):
            g = spec.geom(TABLE_PREFIX + f"table_leg_{k}")
            g.pos, g.size = pos, size
    return spec.compile()


def obj_id(model, objtype, name):
    i = mujoco.mj_name2id(model, objtype, name)
    if i < 0:
        raise ValueError(f"'{name}' not found in compiled model")
    return i


class Context:
    """Everything a worker needs: models, ids, derived grasp geometry, IK."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.geo = read_object_geometry()
        # ---------------- IK model ----------------
        self.ikm = build_ik_model()
        self.ikd = mujoco.MjData(self.ikm)
        self.ik_arm_q = np.array([self.ikm.jnt_qposadr[obj_id(self.ikm, mujoco.mjtObj.mjOBJ_JOINT, j)]
                                  for j in ARM_JOINTS])
        self.ik_fing_q = np.array([self.ikm.jnt_qposadr[obj_id(self.ikm, mujoco.mjtObj.mjOBJ_JOINT, j)]
                                   for j in FINGER_JOINTS])
        jids = [obj_id(self.ikm, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINTS]
        self.q_lo = self.ikm.jnt_range[jids, 0].copy()
        self.q_hi = self.ikm.jnt_range[jids, 1].copy()
        if not np.all((Q_HOME >= self.q_lo) & (Q_HOME <= self.q_hi)):
            raise ValueError("Home pose is outside the XML joint limits.")
        self.mcfg = mink.Configuration(self.ikm)
        self.frame_task = mink.FrameTask(
            frame_name=SITE_NAME, frame_type="site", position_cost=1.0,
            orientation_cost=1.0, gain=IK_GAIN, lm_damping=IK_LM_DAMPING)
        self.posture = mink.PostureTask(self.ikm, cost=IK_POSTURE_COST)
        self.posture.set_target(self.full_ik_q(Q_HOME, FINGER_OPEN))
        self.limits = [mink.ConfigurationLimit(self.ikm)]

        # ---------------- scene model ----------------
        self.m = build_scene_model()
        self.d = mujoco.MjData(self.m)
        m = self.m
        # The checker model is used only for signed-distance queries through
        # self.gdist().  MuJoCo's native CCD is kept enabled (the libccd
        # fallback gives grossly wrong mesh-box distances, e.g. -85 mm for a
        # finger 21 mm above the table), but some releases (observed: 3.11.0)
        # return a spurious exact 0.0 for separated convex pairs; gdist()
        # re-checks every exact-zero answer with an independent GJK.
        J = mujoco.mjtObj.mjOBJ_JOINT
        self.arm_q = np.array([m.jnt_qposadr[obj_id(m, J, j)] for j in ARM_JOINTS])
        self.arm_v = np.array([m.jnt_dofadr[obj_id(m, J, j)] for j in ARM_JOINTS])
        self.fing_q = np.array([m.jnt_qposadr[obj_id(m, J, j)] for j in FINGER_JOINTS])
        self.cube_q = m.jnt_qposadr[obj_id(m, J, CUBE_PREFIX + "cube_free")]
        self.cube_v = m.jnt_dofadr[obj_id(m, J, CUBE_PREFIX + "cube_free")]
        self.cont_q = m.jnt_qposadr[obj_id(m, J, CONTAINER_PREFIX + "container_free")]
        self.cont_v = m.jnt_dofadr[obj_id(m, J, CONTAINER_PREFIX + "container_free")]
        self.site = obj_id(m, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)
        self.hand = obj_id(m, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
        self.base_body = obj_id(m, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
        self.gripper_act = obj_id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR)
        G = mujoco.mjtObj.mjOBJ_GEOM
        self.table_body = obj_id(m, mujoco.mjtObj.mjOBJ_BODY, TABLE_PREFIX + "table")
        self.table_top = obj_id(m, G, TABLE_PREFIX + "table_top")
        self.table_legs = [obj_id(m, G, TABLE_PREFIX + f"table_leg_{i}") for i in range(1, 5)]
        self.cube_geom = obj_id(m, G, CUBE_PREFIX + "cube_geom")
        self.cont_base = obj_id(m, G, CONTAINER_PREFIX + "container_base")
        self.cont_walls = [obj_id(m, G, CONTAINER_PREFIX + n) for n in (
            "container_wall_x_pos", "container_wall_x_neg",
            "container_wall_y_pos", "container_wall_y_neg")]
        self.floor = obj_id(m, G, "floor")
        self.pads = [obj_id(m, G, n) for n in PAD_GEOMS]
        # sanity: XML cube size equals compiled size
        if abs(m.geom_size[self.cube_geom, 0] - self.geo["cube_half"]) > 1e-9:
            raise ValueError("Cube size mismatch between XML parse and model.")

        # Robot collision geoms: group 3 (class "collision") in the Panda subtree.
        body_name = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        self.robot_geoms, self.geom_body = [], {}
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if b == 0 or m.geom_group[g] != 3:
                continue
            root = b
            while m.body_parentid[root] != 0:
                root = m.body_parentid[root]
            if root == self.base_body:
                self.robot_geoms.append(g)
                self.geom_body[g] = body_name(b)
        if not self.robot_geoms:
            raise RuntimeError("No Panda collision geoms (group 3) found.")
        rg = self.robot_geoms
        self.finger_geoms = [g for g in rg if self.geom_body[g] in FINGER_BODIES]
        self.nonfinger_geoms = [g for g in rg if g not in self.finger_geoms]
        self.screen_geoms = [g for g in rg if self.geom_body[g] in SCREEN_BODIES]
        self.floor_geoms = [g for g in rg if self.geom_body[g] not in FLOOR_EXEMPT_BODIES]
        self.cont_geoms = [self.cont_base] + self.cont_walls
        self.table_geoms = [self.table_top] + self.table_legs

        self._self_test_distance()
        self._derive_grasp_geometry()
        self._derive_release_geometry()
        self._derive_self_pairs()
        self._derive_yaw_model()
        self.park_objects()

    # ------------------------------------------------------------------
    def _geom_points(self, g):
        """World-frame support points and inflation radius of a convex geom.

        mesh -> hull vertices, box -> 8 corners, capsule -> segment end points
        (+radius), sphere -> centre (+radius).  Returns (None, 0) otherwise.
        """
        m, d = self.m, self.d
        t = m.geom_type[g]
        R = d.geom_xmat[g].reshape(3, 3)
        c = d.geom_xpos[g]
        sz = m.geom_size[g]
        G = mujoco.mjtGeom
        if t == G.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            adr, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            return c + m.mesh_vert[adr:adr + n] @ R.T, 0.0
        if t == G.mjGEOM_BOX:
            s = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
            return c + (s * sz) @ R.T, 0.0
        if t == G.mjGEOM_CAPSULE:
            ax = R[:, 2] * sz[1]
            return np.array([c - ax, c + ax]), float(sz[0])
        if t == G.mjGEOM_SPHERE:
            return c[None, :].copy(), float(sz[0])
        return None, 0.0

    @staticmethod
    def _closest_on_simplex(W):
        """Closest point to the origin on conv(W) (<= 4 points) by subset search."""
        best, best_idx = None, None
        n = len(W)
        for mask in range(1, 1 << n):
            idx = [i for i in range(n) if mask >> i & 1]
            P = W[idx]
            if len(idx) == 1:
                lam = np.array([1.0])
            else:
                E = (P[1:] - P[0]).T
                try:
                    mu = np.linalg.lstsq(E, -P[0], rcond=None)[0]
                except np.linalg.LinAlgError:
                    continue
                lam = np.r_[1.0 - mu.sum(), mu]
                if np.any(lam < -1e-12):
                    continue
            x = lam @ P
            if best is None or x @ x < best @ best - 1e-15:
                best, best_idx = x, idx
        return best, best_idx

    def _gjk(self, A, B, iters=64):
        """Euclidean distance between conv(A) and conv(B); 0 if they intersect."""
        v = A[0] - B[0]
        W = np.empty((0, 3))
        for _ in range(iters):
            vv = float(v @ v)
            if vv < 1e-18:
                return 0.0
            w = A[np.argmin(A @ v)] - B[np.argmax(B @ v)]
            if vv - float(v @ w) <= 1e-10 * max(vv, 1e-12) + 1e-14:
                return math.sqrt(vv)
            W = np.vstack([W, w])
            v, idx = self._closest_on_simplex(W)
            W = W[idx]
            if len(W) == 4:
                return 0.0
        return math.sqrt(float(v @ v))

    def gdist(self, a, b, cap):
        """Signed distance between geoms a and b (capped at cap).

        Uses mj_geomDistance; an exact-zero answer (the failure mode seen in
        MuJoCo 3.11.0) is re-checked with GJK on the geoms' convex support
        points.  GJK cannot measure penetration, so a confirmed contact stays 0.
        """
        dd = mujoco.mj_geomDistance(self.m, self.d, a, b, cap, None)
        if dd != 0.0:
            return dd
        G = mujoco.mjtGeom
        m = self.m
        if G.mjGEOM_PLANE in (m.geom_type[a], m.geom_type[b]):
            pl, other = (a, b) if m.geom_type[a] == G.mjGEOM_PLANE else (b, a)
            P, r = self._geom_points(other)
            if P is None:
                return dd
            n = self.d.geom_xmat[pl].reshape(3, 3)[:, 2]
            return min(cap, float(np.min((P - self.d.geom_xpos[pl]) @ n)) - r)
        A, ra = self._geom_points(a)
        B, rb = self._geom_points(b)
        if A is None or B is None:
            return dd
        sep = self._gjk(A, B)
        if sep <= 0.0:
            return 0.0
        return min(cap, sep - ra - rb)

    def _self_test_distance(self):
        """Check mj_geomDistance against analytic box-box distances.

        A pad box is moved over the (axis-aligned) tabletop box at known
        separations and penetrations, with the hand pointing down (pad axes
        parallel to the table's).  Fails loudly if the installed MuJoCo returns
        wrong signed distances, instead of silently rejecting every grasp.
        """
        m, d = self.m, self.d
        self.park_objects()
        top = 0.20
        self.set_table(0.45, 0.0, top, 0.6, 0.6, legs=False)
        self.set_robot(Q_HOME, FINGER_OPEN)
        self.fk()
        pad = self.pads[0]
        Rp = d.geom_xmat[pad].reshape(3, 3)
        # pad lowest point along world Z (box half extents projected)
        half_z = float(np.sum(np.abs(Rp[2, :]) * m.geom_size[pad]))
        base_bz = m.body_pos[self.table_body].copy()
        worst = 0.0
        for gap in (0.30, 0.10, 0.05, 0.02, 0.012, 0.004, -0.002):
            pad_bottom = float(d.geom_xpos[pad][2]) - half_z
            # move the table so that its top is 'gap' below the pad bottom
            new_top = pad_bottom - gap
            m.body_pos[self.table_body] = (float(d.geom_xpos[pad][0]),
                                           float(d.geom_xpos[pad][1]),
                                           new_top - TABLE_HALF_THICK)
            mujoco.mj_kinematics(m, d)
            dd = self.gdist(pad, self.table_top, 0.05)
            expect = min(gap, 0.05)
            worst = max(worst, abs(dd - expect))
        # finger mesh (convex hull) vs tabletop: exact answer = lowest hull
        # vertex height above the top face (table far larger than the finger)
        fg = [g for g in self.finger_geoms if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH][0]
        pts, _ = self._geom_points(fg)
        for gap in (0.30, 0.04, 0.021, 0.012, 0.006):
            new_top = float(pts[:, 2].min()) - gap
            m.body_pos[self.table_body] = (float(d.geom_xpos[fg][0]),
                                           float(d.geom_xpos[fg][1]),
                                           new_top - TABLE_HALF_THICK)
            mujoco.mj_kinematics(m, d)
            dd = self.gdist(fg, self.table_top, 0.05)
            worst = max(worst, abs(dd - min(gap, 0.05)))
        m.body_pos[self.table_body] = base_bz
        self.park_objects()
        if worst > 1e-3:
            raise RuntimeError(
                f"mj_geomDistance self-test failed (max error {worst * 1000:.2f} mm) "
                f"with MuJoCo {mujoco.__version__}; collision checks would be unreliable.")
        self.distance_selftest_err = worst

    def full_ik_q(self, q7, qf):
        q = np.zeros(self.ikm.nq)
        q[self.ik_arm_q] = q7
        q[self.ik_fing_q] = qf
        return q

    def set_robot(self, q7, qf):
        self.d.qpos[self.arm_q] = q7
        self.d.qpos[self.fing_q] = qf

    def set_free(self, qadr, pos, quat):
        self.d.qpos[qadr:qadr + 3] = pos
        self.d.qpos[qadr + 3:qadr + 7] = quat

    def park_objects(self):
        """Move cube/container/table far away (they are then simply not checked)."""
        self.set_free(self.cube_q, (5.0, 5.0, 1.0), (1, 0, 0, 0))
        self.set_free(self.cont_q, (-5.0, 5.0, 1.0), (1, 0, 0, 0))
        self.set_table(10.0, 10.0, 0.5, 0.2, 0.2, legs=False)

    def fk(self):
        mujoco.mj_kinematics(self.m, self.d)

    # ------------------------------------------------------------------
    def set_table(self, cx, cy, top_z, lx, ly, legs=True):
        """Resize/move the table in the compiled checker model (distance
        queries only - see build_scene_model for why physics recompiles)."""
        m = self.m
        lay = table_layout(cx, cy, top_z, lx, ly)
        m.body_pos[self.table_body] = lay["body_pos"]
        size = np.array(lay["top_size"])
        m.geom_size[self.table_top] = size
        m.geom_rbound[self.table_top] = float(np.linalg.norm(size))
        m.geom_aabb[self.table_top] = np.r_[0, 0, 0, size]
        for g, (pos, lsize) in zip(self.table_legs, lay["legs"]):
            lsize = np.array(lsize)
            m.geom_pos[g] = pos if legs else (0.0, 0.0, -50.0)
            m.geom_size[g] = lsize
            m.geom_rbound[g] = float(np.linalg.norm(lsize))
            m.geom_aabb[g] = np.r_[0, 0, 0, lsize]

    # ------------------------------------------------------------------
    def _derive_grasp_geometry(self):
        """Derive the grasp TCP and finger geometry from the actual model.

        TCP choice: midpoint of the two named fingertip pads.  These pads are
        the only finger geoms that collide with the cube in the lab scene
        (conaffinity=3; all other finger collision geoms have contype =
        conaffinity = 0), so the pad midpoint is the physical contact centre.
        The pad midpoint alone does not fix the grasp *depth*: the finger
        meshes extend below the pads, so the grasp height is chosen such that
        (a) the fingertips keep FINGERTIP_TABLE_CLEARANCE above the table and
        (b) the pads overlap the cube side faces by >= MIN_PAD_OVERLAP.
        """
        m, d = self.m, self.d
        self.set_robot(Q_HOME, 0.0)
        self.fk()
        Rh = d.xmat[self.hand].reshape(3, 3).copy()
        ph = d.xpos[self.hand].copy()
        Rs = d.site_xmat[self.site].reshape(3, 3).copy()
        ps = d.site_xpos[self.site].copy()
        if not np.allclose(Rh, Rs, atol=1e-9):
            raise ValueError("Site 'gripper' is expected to share the hand orientation.")
        to_hand = lambda p: Rh.T @ (p - ph)
        pad_c = [to_hand(d.geom_xpos[g]) for g in self.pads]
        tcp_hand = 0.5 * (pad_c[0] + pad_c[1])
        self.tcp_offset_site = Rs.T @ (Rh @ tcp_hand + ph - ps)

        # Pad inner-face position (hand +/-Y) at q_finger = 0, and pad extents.
        def extent_along(g, axis_hand):
            Rg = Rh.T @ d.geom_xmat[g].reshape(3, 3)
            return float(np.sum(np.abs(axis_hand @ Rg) * m.geom_size[g]))
        ey = np.array([0.0, 1.0, 0.0])
        ez = np.array([0.0, 0.0, 1.0])
        inner = [abs(pc[1]) - extent_along(g, ey) for g, pc in zip(self.pads, pad_c)]
        self.pad_inner_at_zero = float(np.mean(inner))    # ~0 for this model
        self.pad_half_z = float(np.mean([extent_along(g, ez) for g in self.pads]))

        # Lowest finger point (largest hand-frame z) over all finger geoms.
        tip = -np.inf
        signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
        for g in self.finger_geoms:
            Rg = Rh.T @ d.geom_xmat[g].reshape(3, 3)
            c = to_hand(d.geom_xpos[g])
            aabb = m.geom_aabb[g]
            corners = (Rg @ (aabb[:3] + signs * aabb[3:]).T).T + c
            tip = max(tip, float(corners[:, 2].max()))
        self.tip_below_tcp = tip - float(tcp_hand[2])

        a = self.geo["cube_half"]
        # TCP height above cube centre needed for fingertip/table clearance.
        self.grasp_dz = max(0.0, self.tip_below_tcp + FINGERTIP_TABLE_CLEARANCE - a)
        lo = max(a + self.grasp_dz - self.pad_half_z, 0.0)
        hi = min(a + self.grasp_dz + self.pad_half_z, 2 * a)
        self.pad_overlap = hi - lo
        if self.pad_overlap < MIN_PAD_OVERLAP:
            raise RuntimeError(
                f"Pad/cube overlap {self.pad_overlap*1000:.1f} mm < required "
                f"{MIN_PAD_OVERLAP*1000:.0f} mm: the cube is too small for the "
                "Panda fingertips with the requested table clearance.")
        # Finger joint value at which the pad faces touch the cube faces.
        self.q_contact = a - self.pad_inner_at_zero
        open_gap = 2.0 * (FINGER_OPEN + self.pad_inner_at_zero)
        if not (0.0 < self.q_contact < FINGER_OPEN):
            raise RuntimeError("Cube width is outside the Panda finger stroke.")
        if open_gap < 2 * a + 2 * OPEN_SIDE_CLEARANCE:
            raise RuntimeError(
                f"Open finger gap {open_gap*1000:.1f} mm leaves less than "
                f"{OPEN_SIDE_CLEARANCE*1000:.0f} mm per side around the cube.")
        self.open_gap = open_gap
        self.tcp_hand = tcp_hand

    def _derive_release_geometry(self):
        """Decide whether the fingers can open inside the container opening.

        Half-width of the finger assembly along the closing axis (hand Y), from
        the finger collision-geom AABBs, at the cube-contact and fully-open
        finger positions, compared with the container's inner half-width.
        """
        m, d = self.m, self.d
        out = {}
        for tag, qf in (("contact", self.q_contact), ("open", FINGER_OPEN)):
            self.set_robot(Q_HOME, qf)
            self.fk()
            Rh = d.xmat[self.hand].reshape(3, 3)
            ph = d.xpos[self.hand]
            signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
            w = 0.0
            for g in self.finger_geoms:
                Rg = Rh.T @ d.geom_xmat[g].reshape(3, 3)
                c = Rh.T @ (d.geom_xpos[g] - ph)
                aabb = m.geom_aabb[g]
                corners = (Rg @ (aabb[:3] + signs * aabb[3:]).T).T + c
                w = max(w, float(np.abs(corners[:, 1]).max()))
            out[tag] = w
        self.finger_halfwidth = out
        inner = self.geo["cont_inner_half"]
        self.release_above_walls = out["open"] > inner - ROBOT_MARGIN
        a = self.geo["cube_half"]
        wall_h = self.geo["wall_top_local"] - self.geo["cont_base_half_z"]
        if self.release_above_walls:
            # cube bottom relative to the wall top when the fingertips clear it
            cube_bottom_rel_wall = (FINGER_WALL_CLEARANCE + self.tip_below_tcp
                                    - self.grasp_dz - a)
            if cube_bottom_rel_wall > -MIN_CUBE_INSIDE_WALLS:
                raise RuntimeError(
                    "Fingers cannot open inside the container and the held cube "
                    "would not be inside the walls when released above them.")
            self.release_drop = wall_h + cube_bottom_rel_wall
        else:
            self.release_drop = RELEASE_GAP

    def _derive_self_pairs(self):
        self.set_robot(Q_HOME, FINGER_OPEN)
        self.fk()
        pairs, dropped = [], []
        dist = [g for g in self.robot_geoms if self.geom_body[g] in DISTAL_SELF_BODIES]
        prox = [g for g in self.robot_geoms if self.geom_body[g] in PROXIMAL_BODIES]
        for a in dist:
            for b in prox:
                dd = self.gdist(a, b, 0.05)
                (pairs if dd > SELF_MARGIN else dropped).append((a, b))
        self.self_pairs = pairs
        self.self_pairs_dropped = dropped

    def _derive_yaw_model(self):
        """Hand yaw as a function of (q1, q7) near the home pose, measured by FK."""
        def yaw_at(q7):
            self.set_robot(q7, FINGER_OPEN)
            self.fk()
            R = self.d.site_xmat[self.site].reshape(3, 3)
            return math.atan2(R[1, 0], R[0, 0])
        q = Q_HOME.copy()
        self.yaw_home = yaw_at(q)
        dq = 0.1
        q2 = q.copy(); q2[6] += dq
        self.k7 = (math.remainder(yaw_at(q2) - self.yaw_home, 2 * math.pi)) / dq
        q3 = q.copy(); q3[0] += dq
        self.k1 = (math.remainder(yaw_at(q3) - self.yaw_home, 2 * math.pi)) / dq

    # ------------------------------------------------------------------
    def seed_for(self, x, y, yaw):
        q = Q_HOME.copy()
        q[0] = float(np.clip(math.atan2(y, x), self.q_lo[0], self.q_hi[0]))
        dpsi = math.remainder(yaw - self.yaw_home - self.k1 * q[0], 2 * math.pi)
        q[6] = wrap_into_range(Q_HOME[6] + dpsi / self.k7, self.q_lo[6], self.q_hi[6])
        return q

    def ordered_yaws(self, x, y):
        """Yaw candidates, those leaving joint7 nearest mid-range first."""
        out = []
        for yd in GRASP_YAWS_DEG:
            q = self.seed_for(x, y, math.radians(yd))
            out.append((abs(q[6] - 0.5 * (self.q_lo[6] + self.q_hi[6])), yd, q))
        out.sort(key=lambda t: t[0])
        return [(yd, q) for _, yd, q in out]

    # ------------------------------------------------------------------
    def site_target(self, tcp_pos, R):
        return tcp_pos - R @ self.tcp_offset_site

    def ik(self, tcp_pos, R, seed7, iters):
        """Mink IK for the TCP pose.  Errors are measured by forward kinematics.

        Returns (ok, q7, pos_err, ori_err, reason).
        """
        site_p = self.site_target(tcp_pos, R)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = site_p
        self.frame_task.set_target(mink.SE3.from_matrix(T))
        self.mcfg.update(self.full_ik_q(np.clip(seed7, self.q_lo, self.q_hi), FINGER_OPEN))
        pe = oe = float("inf")
        for _ in range(iters):
            try:
                v = mink.solve_ik(self.mcfg, [self.frame_task, self.posture], IK_DT,
                                  SOLVER, damping=IK_DAMPING, limits=self.limits)
            except Exception:
                return False, None, pe, oe, "ik_solver_error"
            step = np.abs(v * IK_DT).max()
            if step > IK_MAX_STEP:
                v = v * (IK_MAX_STEP / step)
            self.mcfg.integrate_inplace(v, IK_DT)
            Tc = self.mcfg.get_transform_frame_to_world(SITE_NAME, "site")
            pe = float(np.linalg.norm(Tc.translation() - site_p))
            oe = rot_angle(R, Tc.rotation().as_matrix())
            if pe <= POS_TOL and oe <= ORI_TOL:
                q7 = self.mcfg.q[self.ik_arm_q].copy()
                if np.any(q7 < self.q_lo - 1e-6) or np.any(q7 > self.q_hi + 1e-6):
                    return False, None, pe, oe, "joint_limit"
                return True, q7, pe, oe, ""
            if step < 1e-6:
                break
        return False, None, pe, oe, "ik_no_convergence"

    # ------------------------------------------------------------------
    def min_dist(self, A, B, thresh):
        """Return (violation_label or None, min distance) for geom sets A x B.

        Pairs separated by more than thresh+0.05 via bounding spheres are skipped.
        """
        m, d = self.m, self.d
        dmin = np.inf
        cap = thresh + 0.05
        for a in A:
            pa, ra = d.geom_xpos[a], m.geom_rbound[a]
            for b in B:
                if m.geom_type[b] != mujoco.mjtGeom.mjGEOM_PLANE and m.geom_type[a] != mujoco.mjtGeom.mjGEOM_PLANE:
                    if np.linalg.norm(pa - d.geom_xpos[b]) - ra - m.geom_rbound[b] > cap:
                        continue
                dd = self.gdist(a, b, cap)
                if dd < dmin:
                    dmin = dd
                if dd < thresh:
                    return (self.gname(a) + "|" + self.gname(b)), dd
        return None, dmin

    def gname(self, g):
        return mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or (
            f"{self.geom_body.get(g, 'geom')}#{g}")

    def check_state(self, robot_vs_table, cube_mode, container=False,
                    cube_table_allowed=False, cube_cont_check=False,
                    finger_cube_allowed=False):
        """Collision checks for the current data.qpos.  Returns failure label or None.

        robot_vs_table: list of robot geoms checked against the table geoms set
        in self.active_table (patch top only, or top + legs).
        cube_mode: None (cube parked), "on_table", "held", "released".
        """
        self.fk()
        checks = []
        if self.active_table:
            checks.append((robot_vs_table, self.active_table, ROBOT_MARGIN, "robot-table"))
        checks.append((self.floor_geoms, [self.floor], ROBOT_MARGIN, "robot-floor"))
        if cube_mode is not None:
            if finger_cube_allowed:
                checks.append((self.finger_geoms, [self.cube_geom], -FINGER_CUBE_TOL, "finger-cube"))
                checks.append((self.nonfinger_geoms, [self.cube_geom], ROBOT_MARGIN, "robot-cube"))
            else:
                checks.append((self.robot_geoms, [self.cube_geom], ROBOT_MARGIN, "robot-cube"))
            if cube_mode == "held" and self.active_table:
                lim = -0.0005 if cube_table_allowed else CUBE_MARGIN
                checks.append(([self.cube_geom], [self.table_top], lim, "cube-table"))
        if container:
            checks.append((self.robot_geoms, self.cont_geoms, ROBOT_MARGIN, "robot-container"))
            if cube_cont_check and cube_mode == "held":
                checks.append(([self.cube_geom], self.cont_geoms, CUBE_MARGIN, "cube-container"))
        for A, B, th, label in checks:
            v, _ = self.min_dist(A, B, th)
            if v is not None:
                return f"collision_{label}:{v}"
        for a, b in self.self_pairs:
            dd = self.gdist(a, b, 0.02)
            if dd < SELF_MARGIN:
                return f"collision_self:{self.gname(a)}|{self.gname(b)}"
        return None

    # ------------------------------------------------------------------
    def held_cube_pose(self):
        """World cube pose for the current robot state using self.grip_rel."""
        p = self.d.site_xpos[self.site]
        R = self.d.site_xmat[self.site].reshape(3, 3)
        rp, rR = self.grip_rel
        return p + R @ rp, mat2quat(R @ rR)


# ==========================================================================
# Pick / place chains
# ==========================================================================
class ChainResult:
    __slots__ = ("ok", "stage", "yaw", "q_key", "q_end", "pe", "oe", "states",
                 "grip_rel", "ik_reachable", "ik_pe", "ik_oe", "q_ik")

    def __init__(self):
        self.ok = False
        self.stage = "not_tested"
        self.yaw = float("nan")
        self.q_key = self.q_end = self.q_ik = None
        self.pe = self.oe = self.ik_pe = self.ik_oe = float("nan")
        self.states = []
        self.grip_rel = None
        self.ik_reachable = False


def vertical_track(ctx, start_q, xy, z0, z1, R, on_state):
    """Track a vertical TCP line z0->z1 with IK; call on_state(q) per waypoint."""
    n = max(1, int(math.ceil(abs(z1 - z0) / PATH_STEP)))
    q = start_q
    for k in range(1, n + 1):
        z = z0 + (z1 - z0) * k / n
        ok, qn, pe, oe, why = ctx.ik(np.array([xy[0], xy[1], z]), R, q, IK_ITERS_TRACK)
        if not ok:
            return None, f"path_{why}"
        if np.max(np.abs(qn - q)) > MAX_JOINT_JUMP:
            return None, "path_joint_jump"
        err = on_state(qn)
        if err:
            return None, err
        q = qn
    return q, ""


def try_pick(ctx, cx, cy, top, yaw_deg, seed, screen, container_xy=None):
    res = ChainResult()
    res.yaw = yaw_deg
    a = ctx.geo["cube_half"]
    R = downward_rotation(math.radians(yaw_deg))
    cube_c = np.array([cx, cy, top + a])
    cube_quat = np.array([1.0, 0.0, 0.0, 0.0])
    z_grasp = top + a + ctx.grasp_dz
    z_pre = z_grasp + APPROACH_HEIGHT
    z_lift = z_grasp + LIFT_HEIGHT
    rv = ctx.screen_geoms if screen else ctx.robot_geoms
    cont = container_xy is not None

    # L0: IK of the grasp pose itself.
    ok, qg, pe, oe, why = ctx.ik(np.array([cx, cy, z_grasp]), R, seed, IK_ITERS_KEY)
    res.ik_pe, res.ik_oe = pe, oe
    if not ok:
        res.stage = f"grasp_{why}"
        return res
    res.ik_reachable, res.q_ik = True, qg

    # Pre-grasp, seeded from the grasp solution (same IK branch).
    ok, qp, _, _, why = ctx.ik(np.array([cx, cy, z_pre]), R, qg, IK_ITERS_KEY)
    if not ok:
        res.stage = f"pregrasp_{why}"
        return res
    states = []

    def rec(q7, qf, cpos, cquat, tag):
        states.append((q7.copy(), qf, np.array(cpos, float), np.array(cquat, float), tag))

    ctx.set_free(ctx.cube_q, cube_c, cube_quat)
    ctx.set_robot(qp, FINGER_OPEN)
    err = ctx.check_state(rv, "on_table", container=cont)
    if err:
        res.stage = "pregrasp_" + err
        return res
    rec(qp, FINGER_OPEN, cube_c, cube_quat, "pregrasp")

    def approach_state(q7):
        ctx.set_robot(q7, FINGER_OPEN)
        e = ctx.check_state(rv, "on_table", container=cont)
        if not e:
            rec(q7, FINGER_OPEN, cube_c, cube_quat, "approach")
        return ("approach_" + e) if e else ""
    q_at, err = vertical_track(ctx, qp, (cx, cy), z_pre, z_grasp, R, approach_state)
    if q_at is None:
        res.stage = "approach_" + err if not err.startswith("approach") else err
        return res

    # Closure: fingers from open to cube contact.
    for k in range(1, FINGER_STEPS + 1):
        qf = FINGER_OPEN + (ctx.q_contact - FINGER_OPEN) * k / FINGER_STEPS
        ctx.set_robot(q_at, qf)
        err = ctx.check_state(rv, "on_table", container=cont, finger_cube_allowed=True)
        if err:
            res.stage = "grasp_close_" + err
            return res
        rec(q_at, qf, cube_c, cube_quat, "close")
    # Grasp geometry verification (pads straddle the cube centre, touch faces).
    ctx.fk()
    for g in ctx.pads:
        dd = ctx.gdist(g, ctx.cube_geom, 0.05)
        if abs(dd) > PAD_CONTACT_TOL:
            res.stage = f"grasp_geometry_pad_gap_{dd*1000:.1f}mm"
            return res
    mid = 0.5 * (ctx.d.geom_xpos[ctx.pads[0]] + ctx.d.geom_xpos[ctx.pads[1]])
    if np.linalg.norm(mid[:2] - cube_c[:2]) > 0.003:
        res.stage = "grasp_geometry_off_centre"
        return res
    p = ctx.d.site_xpos[ctx.site].copy()
    Rs = ctx.d.site_xmat[ctx.site].reshape(3, 3).copy()
    ctx.grip_rel = (Rs.T @ (cube_c - p), Rs.T @ np.eye(3))
    res.grip_rel = ctx.grip_rel
    res.q_key = q_at.copy()

    # Lift with the cube attached.
    def lift_state(q7):
        ctx.set_robot(q7, ctx.q_contact)
        ctx.fk()
        cp, cq = ctx.held_cube_pose()
        ctx.set_free(ctx.cube_q, cp, cq)
        allow = cp[2] - a - top < CUBE_MARGIN + 0.002
        e = ctx.check_state(rv, "held", container=cont, cube_table_allowed=allow,
                            cube_cont_check=True, finger_cube_allowed=True)
        if not e:
            rec(q7, ctx.q_contact, cp, cq, "lift")
        return ("lift_" + e) if e else ""
    q_l, err = vertical_track(ctx, q_at, (cx, cy), z_grasp, z_lift, R, lift_state)
    if q_l is None:
        res.stage = err if err.startswith("lift") else "lift_" + err
        return res
    res.ok, res.stage, res.q_end, res.states = True, "ok", q_l, states
    return res


def try_place(ctx, bx, by, top, yaw_deg, seed, screen):
    res = ChainResult()
    res.yaw = yaw_deg
    a = ctx.geo["cube_half"]
    R = downward_rotation(math.radians(yaw_deg))
    cont_pos = np.array([bx, by, top + ctx.geo["cont_base_half_z"]])
    base_top = top + 2 * ctx.geo["cont_base_half_z"]
    wall_top = top + ctx.geo["cont_base_half_z"] + ctx.geo["wall_top_local"]
    # Release height.  Ideally the cube is set down RELEASE_GAP above the base,
    # but the open Panda fingers are wider than the container opening (checked
    # in Context._derive_release_geometry), so by default the gripper opens with
    # its lowest finger point FINGER_WALL_CLEARANCE above the wall top and the
    # cube drops the remaining distance inside the walls.
    z_rel = max(base_top + RELEASE_GAP + a + ctx.grasp_dz,
                wall_top + FINGER_WALL_CLEARANCE + ctx.tip_below_tcp) \
        if ctx.release_above_walls else base_top + RELEASE_GAP + a + ctx.grasp_dz
    z_pre = z_rel + PLACE_APPROACH
    if z_pre - ctx.grasp_dz - a < wall_top + CUBE_MARGIN:
        raise RuntimeError("PLACE_APPROACH does not clear the container walls.")
    rv = ctx.screen_geoms if screen else ctx.robot_geoms
    ctx.set_free(ctx.cont_q, cont_pos, (1, 0, 0, 0))
    # nominal held-cube transform: cube centre a+dz below TCP, axes aligned.
    Rs_nom = R
    ctx.grip_rel = (ctx.tcp_offset_site + Rs_nom.T @ np.array([0, 0, -ctx.grasp_dz]),
                    Rs_nom.T @ np.eye(3))
    res.grip_rel = ctx.grip_rel

    ok, qr, pe, oe, why = ctx.ik(np.array([bx, by, z_rel]), R, seed, IK_ITERS_KEY)
    res.ik_pe, res.ik_oe = pe, oe
    if not ok:
        res.stage = f"release_{why}"
        return res
    res.ik_reachable, res.q_ik = True, qr
    ok, qp, _, _, why = ctx.ik(np.array([bx, by, z_pre]), R, qr, IK_ITERS_KEY)
    if not ok:
        res.stage = f"preplace_{why}"
        return res
    states = []

    def rec(q7, qf, cp, cq, tag):
        states.append((q7.copy(), qf, np.array(cp, float), np.array(cq, float), tag))

    def held(q7, tag):
        ctx.set_robot(q7, ctx.q_contact)
        ctx.fk()
        cp, cq = ctx.held_cube_pose()
        ctx.set_free(ctx.cube_q, cp, cq)
        e = ctx.check_state(rv, "held", container=True, cube_cont_check=True,
                            finger_cube_allowed=True)
        if not e:
            rec(q7, ctx.q_contact, cp, cq, tag)
        return (tag + "_" + e) if e else ""

    err = held(qp, "preplace")
    if err:
        res.stage = err
        return res
    q_d, err = vertical_track(ctx, qp, (bx, by), z_pre, z_rel, R,
                              lambda q: held(q, "descend"))
    if q_d is None:
        res.stage = err if err.startswith("descend") else "descend_" + err
        return res
    # Cube footprint inside the container's inner opening (geometric check).
    ctx.fk()
    cp, cq = ctx.held_cube_pose()
    inner = ctx.geo["cont_inner_half"] - a
    if np.max(np.abs(cp[:2] - cont_pos[:2])) > inner - 0.002:
        res.stage = "release_cube_outside_opening"
        return res
    res.q_key = q_d.copy()
    # Release: fingers open; cube stays at the release pose.
    for k in range(1, FINGER_STEPS + 1):
        qf = ctx.q_contact + (FINGER_OPEN - ctx.q_contact) * k / FINGER_STEPS
        ctx.set_robot(q_d, qf)
        ctx.set_free(ctx.cube_q, cp, cq)
        e = ctx.check_state(rv, "released", container=True, finger_cube_allowed=True)
        if e:
            res.stage = "release_open_" + e
            return res
        rec(q_d, qf, cp, cq, "release")

    def retract_state(q7):
        ctx.set_robot(q7, FINGER_OPEN)
        ctx.set_free(ctx.cube_q, cp, cq)
        e = ctx.check_state(rv, "released", container=True)
        if not e:
            rec(q7, FINGER_OPEN, cp, cq, "retract")
        return ("retract_" + e) if e else ""
    q_r, err = vertical_track(ctx, q_d, (bx, by), z_rel, z_pre, R, retract_state)
    if q_r is None:
        res.stage = err if err.startswith("retract") else "retract_" + err
        return res
    res.ok, res.stage, res.q_end, res.states = True, "ok", qp, states
    return res


def run_chain(ctx, kind, x, y, top, screen):
    """Try the yaw candidates in order; return the first fully feasible chain,
    otherwise the attempt that progressed furthest (for diagnostics)."""
    if np.linalg.norm(np.array([x, y, top]) - np.array([0, 0, 0.333])) > MAX_REACH_BOUND:
        r = ChainResult(); r.stage = "outside_max_reach_bound"
        return r
    if screen:
        hx = hy = PATCH_HALF
        ctx.set_table(x, y, top, 2 * hx, 2 * hy, legs=False)
        ctx.active_table = [ctx.table_top]
    best = None
    for yd, seed in ctx.ordered_yaws(x, y):
        ctx.park_objects()
        if screen:
            ctx.set_table(x, y, top, 2 * PATCH_HALF, 2 * PATCH_HALF, legs=False)
        else:
            ctx.set_table(*ctx.full_table, legs=True)
        if kind == "pick":
            r = try_pick(ctx, x, y, top, yd, seed, screen)
        else:
            r = try_place(ctx, x, y, top, yd, seed, screen)
        if r.ok:
            return r
        if best is None or (r.ik_reachable and not best.ik_reachable):
            best = r
    return best


# ==========================================================================
# Worker plumbing
# ==========================================================================
_CTX = None


def _init_worker(cfg):
    global _CTX
    _CTX = Context(cfg)
    _CTX.active_table = []


def _screen_task(args):
    kind, top, ix, iy, x, y = args
    r = run_chain(_CTX, kind, x, y, top, screen=True)
    return (kind, round(top, 6), ix, iy, r.ok, r.stage, r.yaw, r.ik_reachable,
            r.ik_pe, r.ik_oe, None if r.q_ik is None else r.q_ik.tolist())


# ==========================================================================
# Table optimization
# ==========================================================================
def rect_sum(P, i0, i1, j0, j1):
    """Inclusive-index sum on a 2D prefix-sum array P (shape ny+1, nx+1)."""
    return P[j1 + 1, i1 + 1] - P[j0, i1 + 1] - P[j1 + 1, i0] + P[j0, i0]


def prefix(mask):
    P = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int64)
    P[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.int64), 0), 1)
    return P


def best_table_for_height(pick, place, blocked, xs, ys, geo, min_side, exclude=()):
    """Largest table whose randomizer regions are fully task-feasible.

    Objective: maximise tabletop area.
    Constraints (grid-sampled):
      * every sampled cube position the scene randomizer can produce (table
        shrunk by cube half + CUBE_EDGE_MARGIN) is pick-feasible;
      * every grid cell whose cell area overlaps the container-centre region
        (table shrunk by container half + CONTAINER_EDGE_MARGIN) is
        place-feasible;
      * no table cell is blocked by the robot at the home pose;
      * both sides >= min_side, and a cube/container layout with the scene's
        CUBE_CONTAINER_GAP exists.
    Returns dict or None.
    """
    s = float(xs[1] - xs[0])
    cube_m = geo["cube_half"] + CUBE_EDGE_MARGIN
    cont_m = geo["cont_half"] + CONTAINER_EDGE_MARGIN
    shrink = cont_m - cube_m            # container region = cube region shrunk
    k_in = int(math.ceil((shrink - s / 2) / s - 1e-9)) if shrink > s / 2 else 0
    okp = pick & ~blocked
    P1, P2 = prefix(okp), prefix(place & ~blocked)
    nx, ny = len(xs), len(ys)
    best = None
    for j0 in range(ny):
        for j1 in range(j0, ny):
            hy = (ys[j1] - ys[j0]) + 2 * cube_m
            if hy < min_side:
                continue
            for i0 in range(nx):
                for i1 in range(i0, nx):
                    n = (i1 - i0 + 1) * (j1 - j0 + 1)
                    if rect_sum(P1, i0, i1, j0, j1) != n:
                        break  # extending in x keeps the infeasible cell
                    hx = (xs[i1] - xs[i0]) + 2 * cube_m
                    if hx < min_side:
                        continue
                    a0, a1, b0, b1 = i0 + k_in, i1 - k_in, j0 + k_in, j1 - k_in
                    if a0 > a1 or b0 > b1:
                        continue
                    if rect_sum(P2, a0, a1, b0, b1) != (a1 - a0 + 1) * (b1 - b0 + 1):
                        continue
                    span = max(xs[i1] - xs[i0], ys[j1] - ys[j0]) - shrink - geo["cont_half"]
                    if span < CUBE_CONTAINER_GAP:
                        continue
                    if (i0, i1, j0, j1) in exclude:
                        continue
                    area = hx * hy
                    score = (round(area, 9), n, -abs(0.5 * (ys[j0] + ys[j1])))
                    if best is None or score > best["score"]:
                        best = dict(score=score, area=area, idx=(i0, i1, j0, j1),
                                    cont_idx=(a0, a1, b0, b1))
    if best is None:
        return None
    i0, i1, j0, j1 = best["idx"]
    best["cube_region"] = (xs[i0], xs[i1], ys[j0], ys[j1])
    best["table"] = (xs[i0] - cube_m, xs[i1] + cube_m, ys[j0] - cube_m, ys[j1] + cube_m)
    best["cont_region"] = (xs[i0] + shrink, xs[i1] - shrink, ys[j0] + shrink, ys[j1] - shrink)
    return best


def blocked_mask(ctx, top, xs, ys, geo):
    """Cells where a table slab (cell +/- table overhang) hits the robot at home."""
    cube_m = geo["cube_half"] + CUBE_EDGE_MARGIN
    s = float(xs[1] - xs[0])
    ctx.park_objects()
    ctx.set_robot(Q_HOME, FINGER_OPEN)
    out = np.zeros((len(ys), len(xs)), dtype=bool)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            half = s / 2 + cube_m
            ctx.set_table(x, y, top, 2 * half, 2 * half, legs=False)
            ctx.fk()
            v, _ = ctx.min_dist(ctx.robot_geoms, [ctx.table_top], ROBOT_MARGIN)
            out[j, i] = v is not None
    ctx.park_objects()
    return out


# ==========================================================================
# Validation of a candidate table
# ==========================================================================
def recheck_states(ctx, states, container_pos, held_flags=None):
    """Re-check stored chain states with an extra container present."""
    ctx.set_free(ctx.cont_q, container_pos, (1, 0, 0, 0))
    for q7, qf, cp, cq, tag in states:
        ctx.set_robot(q7, qf)
        ctx.set_free(ctx.cube_q, cp, cq)
        allow_fc = tag in ("close", "lift", "release")
        mode = "held" if tag == "lift" else "on_table"
        allow_ct = tag == "lift" and cp[2] - ctx.geo["cube_half"] - ctx.cur_top < CUBE_MARGIN + 0.002
        e = ctx.check_state(ctx.robot_geoms, mode, container=True,
                            cube_cont_check=(tag == "lift"),
                            cube_table_allowed=allow_ct, finger_cube_allowed=allow_fc)
        if e:
            return f"{tag}_{e}"
    return None


def transfer(ctx, q_a, q_b, grip_rel, container_pos):
    """Joint-space transfer with the held cube; returns (ok, path, reason)."""
    ctx.grip_rel = grip_rel
    ctx.set_free(ctx.cont_q, container_pos, (1, 0, 0, 0))

    def seg(qs, qe, path):
        n = max(1, int(math.ceil(np.max(np.abs(qe - qs)) / TRANSFER_JOINT_STEP)))
        for k in range(1, n + 1):
            q = qs + (qe - qs) * k / n
            ctx.set_robot(q, ctx.q_contact)
            ctx.fk()
            cp, cq = ctx.held_cube_pose()
            ctx.set_free(ctx.cube_q, cp, cq)
            e = ctx.check_state(ctx.robot_geoms, "held", container=True,
                                cube_cont_check=True, finger_cube_allowed=True)
            if e:
                return e
            path.append(q.copy())
        return None

    path = []
    e = seg(q_a, q_b, path)
    if e is None:
        return True, path, "direct"
    # Fallback: via-point above the midpoint, IK-solved, two straight segments.
    ctx.set_robot(q_a, ctx.q_contact); ctx.fk()
    pa = ctx.d.site_xpos[ctx.site].copy() + ctx.d.site_xmat[ctx.site].reshape(3, 3) @ ctx.tcp_offset_site
    Ra = ctx.d.site_xmat[ctx.site].reshape(3, 3).copy()
    ctx.set_robot(q_b, ctx.q_contact); ctx.fk()
    pb = ctx.d.site_xpos[ctx.site].copy() + ctx.d.site_xmat[ctx.site].reshape(3, 3) @ ctx.tcp_offset_site
    via = 0.5 * (pa + pb); via[2] = max(pa[2], pb[2]) + TRANSFER_VIA_RISE
    ok, qv, *_ = ctx.ik(via, Ra, 0.5 * (q_a + q_b), IK_ITERS_KEY)
    if not ok:
        return False, None, f"direct:{e};via:ik_failed"
    path = []
    e2 = seg(q_a, qv, path) or seg(qv, q_b, path)
    if e2:
        return False, None, f"direct:{e};via:{e2}"
    return True, path, "via"


def validate_candidate(ctx, top, cand, xs, ys, rng, n_pairs, log):
    """Level-2/3 validation of one candidate table.  Returns report dict."""
    tx0, tx1, ty0, ty1 = cand["table"]
    ctx.full_table = (0.5 * (tx0 + tx1), 0.5 * (ty0 + ty1), top, tx1 - tx0, ty1 - ty0)
    ctx.active_table = [ctx.table_top] + ctx.table_legs
    ctx.cur_top = top
    rep = dict(pick={}, place={}, pair_results=[], fail_pick=set(), fail_place=set(),
               table_home_ok=True)
    # Robot at home vs the full table.
    ctx.park_objects(); ctx.set_table(*ctx.full_table, legs=True)
    ctx.set_robot(Q_HOME, FINGER_OPEN)
    if ctx.check_state(ctx.robot_geoms, None) is not None:
        rep["table_home_ok"] = False
        return rep
    i0, i1, j0, j1 = cand["idx"]
    a0, a1, b0, b1 = cand["cont_idx"]
    for j in range(j0, j1 + 1):
        for i in range(i0, i1 + 1):
            r = run_chain(ctx, "pick", xs[i], ys[j], top, screen=False)
            rep["pick"][(i, j)] = r
            if not r.ok:
                rep["fail_pick"].add((i, j))
    for j in range(b0, b1 + 1):
        for i in range(a0, a1 + 1):
            r = run_chain(ctx, "place", xs[i], ys[j], top, screen=False)
            rep["place"][(i, j)] = r
            if not r.ok:
                rep["fail_place"].add((i, j))
    log(f"    cell validation: pick {len(rep['pick']) - len(rep['fail_pick'])}/"
        f"{len(rep['pick'])}, place {len(rep['place']) - len(rep['fail_place'])}/"
        f"{len(rep['place'])}")
    if rep["fail_pick"] or rep["fail_place"]:
        return rep

    # Level 3: pairs respecting the scene's cube-container gap.
    geo = ctx.geo
    pairs = []
    picks = sorted(rep["pick"]); places = sorted(rep["place"])
    for pc in picks:
        for bc in places:
            dx = max(abs(xs[pc[0]] - xs[bc[0]]) - geo["cont_half"], 0)
            dy = max(abs(ys[pc[1]] - ys[bc[1]]) - geo["cont_half"], 0)
            if math.hypot(dx, dy) >= CUBE_CONTAINER_GAP:
                pairs.append((pc, bc))
    if not pairs:
        rep["no_pairs"] = True
        return rep
    # extremes first (farthest and nearest valid pairs), then random sample.
    dist = [math.hypot(xs[p[0]] - xs[b[0]], ys[p[1]] - ys[b[1]]) for p, b in pairs]
    # Farthest and nearest pairs first, then a reproducible random sample that
    # prefers pick and place cells not used yet (spatial coverage).
    order = [int(np.argmax(dist)), int(np.argmin(dist))]
    seen_p = {pairs[k][0] for k in order}
    seen_b = {pairs[k][1] for k in order}
    perm = [int(k) for k in rng.permutation(len(pairs))]
    for k in perm:
        p, b = pairs[k]
        if k not in order and p not in seen_p and b not in seen_b:
            order.append(k); seen_p.add(p); seen_b.add(b)
    order += [k for k in perm if k not in order]
    order = order[:max(n_pairs, 2)]
    for k in order:
        pc, bc = pairs[k]
        pr, plr = rep["pick"][pc], rep["place"][bc]
        cont_pos = np.array([xs[bc[0]], ys[bc[1]], top + geo["cont_base_half_z"]])
        ctx.park_objects(); ctx.set_table(*ctx.full_table, legs=True)
        e = recheck_states(ctx, pr.states, cont_pos)
        res = dict(pick=pc, place=bc, ok=False, reason="", mode="", path=None)
        if e:
            res["reason"] = "pick_with_container:" + e
        else:
            ok, path, why = transfer(ctx, pr.q_end, plr.q_end, pr.grip_rel, cont_pos)
            res.update(ok=ok, reason="" if ok else "transfer:" + why,
                       mode=why if ok else "", path=path)
        rep["pair_results"].append(res)
    nfail = sum(not r["ok"] for r in rep["pair_results"])
    log(f"    pair checks: {len(rep['pair_results']) - nfail}/{len(rep['pair_results'])} ok")
    return rep


# ==========================================================================
# Physics replay (level 4)
# ==========================================================================
def physics_replay(ctx, top, pick_r, place_r, transfer_path, cube_xy, cont_xy):
    """Replay a planned pick-transfer-place in MuJoCo physics.

    The arm follows the planned joint path through its seven <motor> actuators
    with joint-space PD + gravity/Coriolis compensation (qfrc_bias), similar in
    spirit to the lab scene's gravity-compensated hold.  Fingers, cube and
    container are fully dynamic; the "gripper" actuator closes (ctrl 0) and
    opens (ctrl 0.04).  (Overwriting qpos kinematically was tested and is not
    used: the uncompensated arm then "falls" inside each constraint solve and
    drags the cube out of the pads.)
    """
    m = build_scene_model(table=ctx.full_table)
    for name, idx in (("cube_free", ctx.cube_q), ("container_free", ctx.cont_q)):
        pref = CUBE_PREFIX if name.startswith("cube") else CONTAINER_PREFIX
        if m.jnt_qposadr[obj_id(m, mujoco.mjtObj.mjOBJ_JOINT, pref + name)] != idx:
            raise RuntimeError("Physics model layout differs from the checker model.")
    d = mujoco.MjData(m)
    geo = ctx.geo
    dt = m.opt.timestep
    arm_act = []
    for jn in ARM_JOINTS:
        j = obj_id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        found = [a for a in range(m.nu) if int(m.actuator_trnid[a, 0]) == j]
        if len(found) != 1:
            raise RuntimeError(f"Expected one actuator on {jn}, found {found}")
        arm_act.append(found[0])
    kp = np.array([2000.0] * 4 + [500.0] * 3)
    kd = np.array([100.0] * 4 + [30.0] * 3)

    mujoco.mj_resetData(m, d)
    d.qpos[ctx.arm_q] = pick_r.states[0][0]
    d.qpos[ctx.fing_q] = FINGER_OPEN
    cq0 = ctx.cube_q
    d.qpos[cq0:cq0 + 3] = (cube_xy[0], cube_xy[1], top + geo["cube_half"] + 0.0005)
    d.qpos[cq0 + 3:cq0 + 7] = (1, 0, 0, 0)
    k0 = ctx.cont_q
    d.qpos[k0:k0 + 3] = (cont_xy[0], cont_xy[1], top + geo["cont_base_half_z"] + 0.0005)
    d.qpos[k0 + 3:k0 + 7] = (1, 0, 0, 0)
    mujoco.mj_forward(m, d)

    seq = [(s[0], FINGER_OPEN) for s in pick_r.states if s[4] in ("pregrasp", "approach")]
    seq.append(("dwell", 0.0, 0.8, pick_r.q_key))
    seq += [(s[0], 0.0) for s in pick_r.states if s[4] == "lift"]
    seq += [(q, 0.0) for q in transfer_path]
    seq += [(s[0], 0.0) for s in place_r.states if s[4] in ("preplace", "descend")]
    seq.append(("dwell", FINGER_OPEN, 0.8, place_r.q_key))
    seq += [(s[0], FINGER_OPEN) for s in place_r.states if s[4] == "retract"]

    vmax = 0.6                     # rad/s joint speed for the replay
    q_cur = d.qpos[ctx.arm_q].copy()
    track_err = 0.0
    cube_lifted = False

    def step_to(q_goal, grip, duration=None):
        nonlocal q_cur, track_err
        q_goal = np.asarray(q_goal, float)
        dist = float(np.max(np.abs(q_goal - q_cur)))
        T = duration if duration is not None else max(dist / vmax, 0.02)
        n = max(1, int(round(T / dt)))
        vel = (q_goal - q_cur) / (n * dt) if duration is None else np.zeros(7)
        for k in range(1, n + 1):
            qr = q_cur + (q_goal - q_cur) * min(1.0, k / n) if duration is None else q_goal
            q = d.qpos[ctx.arm_q]
            v = d.qvel[ctx.arm_v]
            d.ctrl[:] = 0.0
            d.ctrl[arm_act] = kp * (qr - q) + kd * (vel - v) + d.qfrc_bias[ctx.arm_v]
            d.ctrl[ctx.gripper_act] = grip
            mujoco.mj_step(m, d)
            track_err = max(track_err, float(np.max(np.abs(qr - d.qpos[ctx.arm_q]))))
        q_cur = q_goal.copy()

    step_to(q_cur.copy(), FINGER_OPEN, duration=0.3)          # settle objects
    for item in seq:
        if isinstance(item[0], str):
            _, grip, T, qh = item
            step_to(qh, grip, duration=T)
        else:
            step_to(item[0], item[1])
            if item[1] == 0.0 and d.qpos[cq0 + 2] > top + geo["cube_half"] + 0.05:
                cube_lifted = True
    step_to(q_cur.copy(), FINGER_OPEN, duration=0.5)
    cube = d.qpos[cq0:cq0 + 3].copy()
    cont = d.qpos[k0:k0 + 3].copy()
    inner = geo["cont_inner_half"]
    inside = bool(np.max(np.abs(cube[:2] - cont[:2])) < inner
                  and cube[2] < top + 2 * geo["cont_base_half_z"] + 2 * geo["cube_half"] + 0.01)
    cont_moved = float(np.linalg.norm(cont[:2] - np.asarray(cont_xy)))
    return dict(ok=bool(cube_lifted and inside and cont_moved < 0.02),
                cube_lifted=cube_lifted, cube_in_container=inside,
                container_displacement=cont_moved, cube_final=cube.tolist(),
                max_tracking_error=track_err)


# ==========================================================================
# Plotting
# ==========================================================================
def make_plots(out, xs, ys, heights, cells, height_scores, best_top, cand, vrep,
               base_pos, ctx, pairs_drawn, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # ---- Figure 1: 3D -----------------------------------------------------
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    reach = np.array([c["tcp"] for c in cells if c["ik_grasp_reachable"] and not c["task_ok"]])
    task = np.array([c["tcp"] for c in cells if c["task_ok"]])
    pick_only = np.array([c["tcp"] for c in cells if c["pick_ok"] and not c["task_ok"]])
    if len(reach):
        ax.scatter(reach[:, 0], reach[:, 1], reach[:, 2], s=6, c="tab:orange", alpha=0.35,
                   label="IK-reachable grasp pose only (L0)")
    if len(pick_only):
        ax.scatter(pick_only[:, 0], pick_only[:, 1], pick_only[:, 2], s=9, c="gold",
                   alpha=0.6, label="Pick chain feasible, place not (L1)")
    if len(task):
        ax.scatter(task[:, 0], task[:, 1], task[:, 2], s=12, c="tab:green", alpha=0.8,
                   label="Pick AND place chains feasible (L1, screened)")
    tx0, tx1, ty0, ty1 = cand["table"]
    xx, yy = np.meshgrid([tx0, tx1], [ty0, ty1])
    ax.plot_surface(xx, yy, np.full_like(xx, best_top), color="saddlebrown", alpha=0.2,
                    shade=False)
    ax.plot([tx0, tx1, tx1, tx0, tx0], [ty0, ty0, ty1, ty1, ty0], [best_top] * 5,
            color="saddlebrown", lw=2, label="Recommended tabletop")
    ax.scatter(*base_pos, marker="*", s=250, c="tab:blue", label="Franka base (link0)")
    ax.set_xlabel("World X (m)"); ax.set_ylabel("World Y (m)"); ax.set_zlabel("World Z (m)")
    ax.set_title("Franka downward-grasp workspace: IK reachability vs task feasibility\n"
                 "(points = grasp TCP positions)")
    ax.set_xlim(min(xs.min(), -0.1), xs.max()); ax.set_ylim(ys.min(), ys.max())
    ax.set_zlim(0, max(heights) + 0.15)
    ax.set_box_aspect((xs.max() - min(xs.min(), -0.1), ys.max() - ys.min(),
                       max(heights) + 0.15))
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    p3d = out / "franka_dexterous_workspace_3d.png"
    fig.savefig(p3d, dpi=200)

    # ---- Figure 2: height optimization ------------------------------------
    fig_h, ax_h = plt.subplots(figsize=(9.5, 6.0))
    hs = sorted(height_scores)
    area = [height_scores[h]["area"] for h in hs]
    ax_h.plot(hs, area, "o-", lw=2, color="tab:green",
              label="Objective: largest task-feasible table area (screened)")
    ax_h.set_xlabel("Tabletop height Z (m)")
    ax_h.set_ylabel("Table area (m²)")
    ax2 = ax_h.twinx()
    ax2.plot(hs, [height_scores[h]["n_ik"] for h in hs], "s--", color="tab:orange",
             alpha=0.8, label="# cells IK-reachable (L0)")
    ax2.plot(hs, [height_scores[h]["n_pick"] for h in hs], "^--", color="gold",
             label="# cells pick-feasible (L1)")
    ax2.plot(hs, [height_scores[h]["n_task"] for h in hs], "v--", color="tab:purple",
             label="# cells pick+place feasible (L1)")
    ax2.set_ylabel("Sampled cells")
    ax_h.axvline(best_top, color="k", ls=":", lw=1.5,
                 label=f"Selected height {best_top:.3f} m (validated)")
    h1, l1 = ax_h.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax_h.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center",
                bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    ax_h.set_title("Franka table height selection")
    ax_h.grid(alpha=0.3)
    plt.tight_layout()
    ph = out / "franka_dexterous_workspace_height_optimization.png"
    fig_h.savefig(ph, dpi=200, bbox_inches="tight")

    # ---- Figure 3: table plane -------------------------------------------
    fig2, a2 = plt.subplots(figsize=(9.5, 8.5))
    at = [c for c in cells if abs(c["table_top_z"] - best_top) < 1e-9]
    style = {
        "unreach": dict(marker="x", c="lightgray", s=18),
        "reach": dict(marker="o", c="tab:orange", s=20),
        "pick": dict(marker="o", c="gold", s=24),
        "place": dict(marker="o", c="tab:cyan", s=24),
        "task": dict(marker="o", c="tab:green", s=26),
        "blocked": dict(marker="s", c="k", s=16),
    }
    for c in at:
        if c["blocked_by_robot"]:
            k = "blocked"
        elif c["task_ok"]:
            k = "task"
        elif c["pick_ok"]:
            k = "pick"
        elif c["place_ok"]:
            k = "place"
        elif c["ik_grasp_reachable"]:
            k = "reach"
        else:
            k = "unreach"
        a2.scatter(c["x"], c["y"], **style[k])
    for (pc, bc) in pairs_drawn:
        a2.plot([xs[pc[0]], xs[bc[0]]], [ys[pc[1]], ys[bc[1]]], color="tab:gray",
                lw=0.5, alpha=0.4)
    a2.add_patch(plt.Rectangle((tx0, ty0), tx1 - tx0, ty1 - ty0, fill=False,
                               lw=2.5, ec="saddlebrown"))
    cx0, cx1, cy0, cy1 = cand["cube_region"]
    a2.add_patch(plt.Rectangle((cx0, cy0), cx1 - cx0, cy1 - cy0, fill=False,
                               lw=1.6, ls="--", ec="tab:red"))
    bx0, bx1, by0, by1 = cand["cont_region"]
    a2.add_patch(plt.Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False,
                               lw=1.6, ls=":", ec="tab:blue"))
    a2.scatter(base_pos[0], base_pos[1], marker="*", s=300, c="tab:blue")
    handles = [
        Line2D([0], [0], marker="o", ls="", color="tab:green", label="Pick+place feasible (L1)"),
        Line2D([0], [0], marker="o", ls="", color="gold", label="Pick only"),
        Line2D([0], [0], marker="o", ls="", color="tab:cyan", label="Place only"),
        Line2D([0], [0], marker="o", ls="", color="tab:orange", label="IK-reachable only (L0)"),
        Line2D([0], [0], marker="x", ls="", color="lightgray", label="IK not reached"),
        Line2D([0], [0], marker="s", ls="", color="k", label="Table would hit robot (home)"),
        Line2D([0], [0], color="saddlebrown", lw=2.5, label="Recommended tabletop"),
        Line2D([0], [0], color="tab:red", ls="--", label="Cube-centre region (validated L2)"),
        Line2D([0], [0], color="tab:blue", ls=":", label="Container-centre region (validated L2)"),
        Line2D([0], [0], color="tab:gray", lw=0.8, label="Validated pick→place pairs (L3)"),
        Line2D([0], [0], marker="*", ls="", color="tab:blue", ms=13, label="Franka base"),
    ]
    a2.legend(handles=handles, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    a2.set_aspect("equal", adjustable="box")
    a2.set_xlim(min(xs.min(), -0.15) - 0.05, xs.max() + 0.05)
    a2.set_ylim(ys.min() - 0.05, ys.max() + 0.05)
    a2.set_xlabel("World X (m)"); a2.set_ylabel("World Y (m)")
    a2.set_title(f"Top-down task feasibility at tabletop Z = {best_top:.3f} m\n"
                 f"Table {tx1 - tx0:.3f} × {ty1 - ty0:.3f} m, centre "
                 f"({0.5 * (tx0 + tx1):.3f}, {0.5 * (ty0 + ty1):.3f})")
    a2.grid(alpha=0.25)
    plt.tight_layout()
    p2d = out / "franka_dexterous_workspace_table_plane.png"
    fig2.savefig(p2d, dpi=200, bbox_inches="tight")
    return [p3d, ph, p2d], plt


# ==========================================================================
# Main
# ==========================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    D = DEFAULTS
    p.add_argument("--x-range", nargs=2, type=float, default=(D["x_min"], D["x_max"]))
    p.add_argument("--y-range", nargs=2, type=float, default=(D["y_min"], D["y_max"]))
    p.add_argument("--xy-step", type=float, default=D["xy_step"])
    p.add_argument("--z-range", nargs=2, type=float, default=(D["z_min"], D["z_max"]),
                   help="candidate tabletop heights (m)")
    p.add_argument("--z-coarse-step", type=float, default=D["z_coarse"])
    p.add_argument("--z-fine-step", type=float, default=D["z_fine"],
                   help="refinement step around the best coarse heights (0 = off)")
    p.add_argument("--min-table-side", type=float, default=D["min_table_side"])
    p.add_argument("--transfer-pairs", type=int, default=D["transfer_pairs"])
    p.add_argument("--physics-pairs", type=int, default=D["physics_pairs"])
    p.add_argument("--validation-rounds", type=int, default=D["validation_rounds"])
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1)))
    p.add_argument("--seed", type=int, default=D["seed"])
    p.add_argument("--no-show", action="store_true", help="do not open plot windows")
    return p.parse_args()


def grid(lo, hi, step):
    n = int(round((hi - lo) / step)) + 1
    return np.round(lo + step * np.arange(n), 6)


def main():
    args = parse_args()
    t_start = time.time()
    require_files()
    RESULTS.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(s, flush=True)

    ctx = Context(vars(args))
    ctx.active_table = []
    geo = ctx.geo
    base_pos = ctx.d.xpos[ctx.base_body].copy()
    xs = grid(*args.x_range, args.xy_step)
    ys = grid(*args.y_range, args.xy_step)
    if len(xs) < 2 or len(ys) < 2:
        raise ValueError("XY grid needs at least 2 samples per axis.")
    if args.z_range[0] < 2 * TABLE_HALF_THICK + 0.01:
        raise ValueError("Minimum tabletop height must exceed the table thickness.")

    log("=== Franka Panda pick-and-place workspace analysis ===")
    log(f"Franka XML: {ARM_XML}")
    log(f"Base (link0) world position: {np.round(base_pos, 4)}")
    log("Arm joint limits (rad): " + ", ".join(
        f"{n}[{lo:+.3f},{hi:+.3f}]" for n, lo, hi in zip(ARM_JOINTS, ctx.q_lo, ctx.q_hi)))
    log(f"MuJoCo {mujoco.__version__}: geom-distance self-test max error "
        f"{ctx.distance_selftest_err * 1000:.3f} mm")
    log(f"Robot collision geoms checked: {len(ctx.robot_geoms)} "
        f"(fingers {len(ctx.finger_geoms)}); self-collision pairs: {len(ctx.self_pairs)} "
        f"(dropped {len(ctx.self_pairs_dropped)} touching at home)")
    log(f"TCP (pad-contact midpoint) offset in '{SITE_NAME}' site frame: "
        f"{np.round(ctx.tcp_offset_site, 4)} m")
    log(f"Lowest finger point below TCP: {ctx.tip_below_tcp * 1000:.1f} mm; "
        f"grasp TCP above cube centre: {ctx.grasp_dz * 1000:.1f} mm; "
        f"pad/face overlap: {ctx.pad_overlap * 1000:.1f} mm")
    log(f"Finger joint at cube contact: {ctx.q_contact:.4f} m (open {FINGER_OPEN} m, "
        f"open gap {ctx.open_gap * 1000:.1f} mm vs cube {2 * geo['cube_half'] * 1000:.0f} mm)")
    log(f"Finger half-width along closing axis: contact "
        f"{ctx.finger_halfwidth['contact'] * 1000:.1f} mm, open "
        f"{ctx.finger_halfwidth['open'] * 1000:.1f} mm; container inner half-width "
        f"{geo['cont_inner_half'] * 1000:.1f} mm -> release "
        f"{'above the walls (cube drops ' + format(ctx.release_drop * 100, '.1f') + ' cm)' if ctx.release_above_walls else 'inside the container'}")
    log(f"Hand yaw model: yaw_home={math.degrees(ctx.yaw_home):.1f} deg, "
        f"dyaw/dq1={ctx.k1:+.2f}, dyaw/dq7={ctx.k7:+.2f}")

    coarse = grid(args.z_range[0], args.z_range[1], args.z_coarse_step)
    cells = {}          # (top, ix, iy) -> dict
    pool = None
    if args.workers > 1:
        pool = mp.get_context("spawn").Pool(args.workers, initializer=_init_worker,
                                            initargs=(vars(args),))
    else:
        _init_worker(vars(args))

    def screen_heights(heights):
        tasks = []
        for top in heights:
            for iy, y in enumerate(ys):
                for ix, x in enumerate(xs):
                    for kind in ("pick", "place"):
                        tasks.append((kind, float(top), ix, iy, float(x), float(y)))
        log(f"Screening {len(tasks)} chains at heights {list(map(float, heights))} ...")
        it = pool.imap_unordered(_screen_task, tasks, chunksize=8) if pool else map(_screen_task, tasks)
        t0 = time.time()
        for n, r in enumerate(it, 1):
            kind, top, ix, iy, ok, stage, yaw, ikr, ipe, ioe, qik = r
            c = cells.setdefault((top, ix, iy), dict(
                table_top_z=top, x=float(xs[ix]), y=float(ys[iy]), ix=ix, iy=iy))
            c[f"{kind}_ok"] = ok
            c[f"{kind}_stage"] = stage
            c[f"{kind}_yaw_deg"] = yaw
            c[f"{kind}_ik_reachable"] = ikr
            c[f"{kind}_ik_pos_err_m"] = ipe
            c[f"{kind}_ik_ori_err_rad"] = ioe
            c[f"{kind}_q_ik"] = qik
            if n % 200 == 0 or n == len(tasks):
                log(f"  {n}/{len(tasks)} chains ({time.time() - t0:.0f} s)")

    height_scores = {}

    def score_heights(heights):
        for top in heights:
            top = round(float(top), 6)
            pick = np.zeros((len(ys), len(xs)), bool)
            place = np.zeros_like(pick)
            ikm = np.zeros_like(pick)
            for (t, ix, iy), c in cells.items():
                if abs(t - top) < 1e-9:
                    pick[iy, ix] = c["pick_ok"]
                    place[iy, ix] = c["place_ok"]
                    ikm[iy, ix] = c["pick_ik_reachable"]
            blk = blocked_mask(ctx, top, xs, ys, geo)
            for (t, ix, iy), c in cells.items():
                if abs(t - top) < 1e-9:
                    c["blocked_by_robot"] = bool(blk[iy, ix])
            cand = best_table_for_height(pick, place, blk, xs, ys, geo, args.min_table_side)
            height_scores[top] = dict(
                pick=pick, place=place, blocked=blk, cand=cand,
                area=cand["area"] if cand else 0.0, n_ik=int(ikm.sum()),
                n_pick=int(pick.sum()), n_task=int((pick & place).sum()))
            log(f"  Z={top:.3f}: IK-reachable {int(ikm.sum())}, pick {int(pick.sum())}, "
                f"pick+place {int((pick & place).sum())}, best table area "
                f"{height_scores[top]['area']:.4f} m^2")

    # ---- Stage A: coarse heights ---------------------------------------
    screen_heights(coarse)
    score_heights(coarse)
    # ---- Stage B: refine around the two best coarse heights -------------
    if args.z_fine_step > 0 and args.z_fine_step < args.z_coarse_step:
        ranked = sorted(height_scores, key=lambda h: -height_scores[h]["area"])[:2]
        fine = set()
        for h in ranked:
            for k in range(1, int(round(args.z_coarse_step / args.z_fine_step))):
                for sgn in (-1, 1):
                    z = round(h + sgn * k * args.z_fine_step, 6)
                    if args.z_range[0] - 1e-9 <= z <= args.z_range[1] + 1e-9 \
                            and all(abs(z - e) > 1e-9 for e in height_scores):
                        fine.add(z)
        if fine:
            screen_heights(sorted(fine))
            score_heights(sorted(fine))
    if pool:
        pool.close(); pool.join()

    # ---- Stage C: validation with the full table ------------------------
    rng = np.random.default_rng(args.seed)
    excluded = {h: set() for h in height_scores}
    accepted = None
    log("\nValidating candidates with the full table (top + legs) and full robot geometry ...")
    for rnd in range(args.validation_rounds):
        ranked = [h for h in sorted(height_scores, key=lambda h: (-height_scores[h]["area"], h))
                  if height_scores[h]["cand"] is not None]
        if not ranked:
            break
        top = ranked[0]
        hs = height_scores[top]
        cand = hs["cand"]
        log(f"  round {rnd + 1}: Z={top:.3f}, table X {cand['table'][0]:.3f}..{cand['table'][1]:.3f}"
            f", Y {cand['table'][2]:.3f}..{cand['table'][3]:.3f}")
        rep = validate_candidate(ctx, top, cand, xs, ys, rng, args.transfer_pairs, log)
        bad_pairs = [r for r in rep["pair_results"] if not r["ok"]]
        if rep["table_home_ok"] and not rep["fail_pick"] and not rep["fail_place"] \
                and not bad_pairs and not rep.get("no_pairs"):
            accepted = (top, cand, rep)
            break
        if not rep["table_home_ok"] or rep.get("no_pairs"):
            excluded[top].add(cand["idx"])
        for (i, j) in rep["fail_pick"]:
            hs["pick"][j, i] = False
        for (i, j) in rep["fail_place"]:
            hs["place"][j, i] = False
        for r in bad_pairs:
            hs["pick"][r["pick"][1], r["pick"][0]] = False
            hs["place"][r["place"][1], r["place"][0]] = False
        hs["cand"] = best_table_for_height(hs["pick"], hs["place"], hs["blocked"], xs, ys,
                                           geo, args.min_table_side, exclude=excluded[top])
        hs["area"] = hs["cand"]["area"] if hs["cand"] else 0.0
        # record validation demotions in the cell table
        for (i, j) in rep["fail_pick"]:
            cells[(top, i, j)]["pick_validation"] = "failed:" + rep["pick"][(i, j)].stage
        for (i, j) in rep["fail_place"]:
            cells[(top, i, j)]["place_validation"] = "failed:" + rep["place"][(i, j)].stage
    if accepted is None:
        raise RuntimeError(
            "No candidate table passed full validation. Widen the sampling ranges, "
            "increase --validation-rounds, or relax --min-table-side.")
    best_top, cand, rep = accepted

    # ---- Stage D: physics replay ----------------------------------------
    phys = []
    ok_pairs = [r for r in rep["pair_results"] if r["ok"]]
    chosen, used = [], set()
    for r in ok_pairs:                      # distinct cube cells first
        if len(chosen) < args.physics_pairs and r["pick"] not in used:
            chosen.append(r); used.add(r["pick"])
    for r in ok_pairs:
        if len(chosen) < args.physics_pairs and r not in chosen:
            chosen.append(r)
    for r in chosen:
        pc, bc = r["pick"], r["place"]
        ctx.park_objects()
        ctx.set_table(*ctx.full_table, legs=True)
        res = physics_replay(ctx, best_top, rep["pick"][pc], rep["place"][bc], r["path"],
                             (xs[pc[0]], ys[pc[1]]), (xs[bc[0]], ys[bc[1]]))
        res.update(pick=pc, place=bc)
        phys.append(res)
        log(f"  physics replay cube ({xs[pc[0]]:.2f},{ys[pc[1]]:.2f}) -> container "
            f"({xs[bc[0]]:.2f},{ys[bc[1]]:.2f}): {'OK' if res['ok'] else 'FAILED'} "
            f"(lifted={res['cube_lifted']}, in_container={res['cube_in_container']}, "
            f"container moved {res['container_displacement'] * 100:.1f} cm, "
            f"max joint tracking error {res['max_tracking_error']:.3f} rad)")

    # ---- Annotate cells ------------------------------------------------
    for (t, ix, iy), c in cells.items():
        c["task_ok"] = bool(c.get("pick_ok") and c.get("place_ok"))
        c["ik_grasp_reachable"] = bool(c.get("pick_ik_reachable"))
        a = geo["cube_half"]
        c["tcp"] = (c["x"], c["y"], t + a + ctx.grasp_dz)
        in_t = abs(t - best_top) < 1e-9
        c["in_cube_region"] = in_t and (ix, iy) in rep["pick"]
        c["in_container_region"] = in_t and (ix, iy) in rep["place"]
        if c["in_cube_region"]:
            c["pick_validation"] = "L2_ok"
        if c["in_container_region"]:
            c["place_validation"] = "L2_ok"

    # ---- CSV ---------------------------------------------------------------
    csv_path = RESULTS / "franka_dexterous_workspace.csv"
    fields = ["table_top_z_m", "x_m", "y_m", "cube_center_z_m", "grasp_tcp_z_m",
              "target_orientation", "blocked_by_robot",
              "ik_grasp_reachable", "pick_ik_pos_err_m", "pick_ik_ori_err_rad",
              "pick_q_grasp_rad", "pick_yaw_deg", "pick_ok", "pick_fail_stage",
              "place_ik_reachable", "place_ik_pos_err_m", "place_ik_ori_err_rad",
              "place_q_release_rad", "place_yaw_deg", "place_ok", "place_fail_stage",
              "task_ok_screened", "pick_validation", "place_validation",
              "in_recommended_cube_region", "in_recommended_container_region"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key in sorted(cells):
            c = cells[key]
            fmtq = lambda q: "" if q is None else " ".join(f"{v:.4f}" for v in q)
            w.writerow({
                "table_top_z_m": c["table_top_z"], "x_m": c["x"], "y_m": c["y"],
                "cube_center_z_m": round(c["table_top_z"] + geo["cube_half"], 6),
                "grasp_tcp_z_m": round(c["tcp"][2], 6),
                "target_orientation": "hand +Z -> world -Z; yaw = *_yaw_deg about world Z",
                "blocked_by_robot": int(c.get("blocked_by_robot", False)),
                "ik_grasp_reachable": int(c["ik_grasp_reachable"]),
                "pick_ik_pos_err_m": c.get("pick_ik_pos_err_m"),
                "pick_ik_ori_err_rad": c.get("pick_ik_ori_err_rad"),
                "pick_q_grasp_rad": fmtq(c.get("pick_q_ik")),
                "pick_yaw_deg": c.get("pick_yaw_deg"),
                "pick_ok": int(bool(c.get("pick_ok"))),
                "pick_fail_stage": "" if c.get("pick_ok") else c.get("pick_stage"),
                "place_ik_reachable": int(bool(c.get("place_ik_reachable"))),
                "place_ik_pos_err_m": c.get("place_ik_pos_err_m"),
                "place_ik_ori_err_rad": c.get("place_ik_ori_err_rad"),
                "place_q_release_rad": fmtq(c.get("place_q_ik")),
                "place_yaw_deg": c.get("place_yaw_deg"),
                "place_ok": int(bool(c.get("place_ok"))),
                "place_fail_stage": "" if c.get("place_ok") else c.get("place_stage"),
                "task_ok_screened": int(c["task_ok"]),
                "pick_validation": c.get("pick_validation", "not_validated"),
                "place_validation": c.get("place_validation", "not_validated"),
                "in_recommended_cube_region": int(c["in_cube_region"]),
                "in_recommended_container_region": int(c["in_container_region"]),
            })
    pair_csv = RESULTS / "franka_dexterous_workspace_pairs.csv"
    with pair_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cube_x_m", "cube_y_m", "container_x_m", "container_y_m",
                    "pair_ok", "transfer_mode", "reason", "physics_replay"])
        pmap = {(p["pick"], p["place"]): p for p in phys}
        for r in rep["pair_results"]:
            pr = pmap.get((r["pick"], r["place"]))
            w.writerow([xs[r["pick"][0]], ys[r["pick"][1]], xs[r["place"][0]], ys[r["place"][1]],
                        int(r["ok"]), r["mode"], r["reason"],
                        "" if pr is None else ("ok" if pr["ok"] else "failed")])

    # ---- Plots -------------------------------------------------------------
    pairs_drawn = [(r["pick"], r["place"]) for r in rep["pair_results"] if r["ok"]]
    paths, plt = make_plots(RESULTS, xs, ys, sorted(height_scores), list(cells.values()),
                            height_scores, best_top, cand, rep, base_pos, ctx,
                            pairs_drawn, show=not args.no_show)

    # ---- Summary -----------------------------------------------------------
    tx0, tx1, ty0, ty1 = cand["table"]
    lx, ly = tx1 - tx0, ty1 - ty0
    tcx, tcy = 0.5 * (tx0 + tx1), 0.5 * (ty0 + ty1)
    th = TABLE_HALF_THICK
    lay = table_layout(tcx, tcy, best_top, lx, ly)
    n_pairs = len(rep["pair_results"])
    n_pairs_ok = sum(r["ok"] for r in rep["pair_results"])
    n_via = sum(r["mode"] == "via" for r in rep["pair_results"])
    n_phys_ok = sum(p["ok"] for p in phys)
    tot = len(cells)
    leg_lines = []
    for k, (pos, size) in enumerate(lay["legs"], 1):
        leg_lines.append(
            f'      <geom name="table_leg_{k}" type="box" size="{size[0]:.3f} {size[1]:.3f} '
            f'{size[2]:.3f}" pos="{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}" rgba="0.30 0.20 0.13 1"/>')
    cr, br = cand["cube_region"], cand["cont_region"]
    summary = f"""Franka Panda pick-and-place table recommendation
Generated by compute_franka_dexterous_workspace.py on {time.strftime('%Y-%m-%d %H:%M:%S')}
Run time: {time.time() - t_start:.0f} s

COORDINATE FRAME
  World frame of the compiled scene (same as franka_pick_place_scene.py).
  Franka base body link0 at {np.round(base_pos, 4).tolist()} m (world), +X forward, +Z up.
  All numbers below are world-frame metres; table_fr.xml uses the same frame.

RECOMMENDED TABLE (table_fr.xml conventions)
  Tabletop surface height Z : {best_top:.4f} m
  Tabletop thickness        : {2 * th:.4f} m (half {th:.3f}, as in table_fr.xml)
  Tabletop X dimension      : {lx:.4f} m   (X = {tx0:.4f} .. {tx1:.4f})
  Tabletop Y dimension      : {ly:.4f} m   (Y = {ty0:.4f} .. {ty1:.4f})
  Table centre (x, y)       : ({tcx:.4f}, {tcy:.4f}) m
  Table body pos            : "{tcx:.4f} {tcy:.4f} {best_top - th:.4f}"
  table_top geom size       : "{lx / 2:.4f} {ly / 2:.4f} {th:.4f}"
  Area                      : {lx * ly:.4f} m^2

  Ready-to-paste body (verify before replacing table_fr.xml; not applied automatically):
    <body name="table" pos="{tcx:.4f} {tcy:.4f} {best_top - th:.4f}">
      <geom name="table_top" type="box" size="{lx / 2:.4f} {ly / 2:.4f} {th:.4f}" pos="0 0 0"
            rgba="0.48 0.32 0.20 1" contype="1" conaffinity="1" friction="1.0 0.005 0.0001"/>
{chr(10).join(leg_lines)}
    </body>

FEASIBLE PLACEMENT REGIONS (world XY of object centres)
  Cube-centre region      : X {cr[0]:.3f} .. {cr[1]:.3f}, Y {cr[2]:.3f} .. {cr[3]:.3f}
     (= tabletop shrunk by cube half {geo['cube_half']:.3f} + CUBE_EDGE_MARGIN {CUBE_EDGE_MARGIN:.3f},
      the region franka_pick_place_scene.py samples cube centres from at yaw 0)
  Container-centre region : X {br[0]:.3f} .. {br[1]:.3f}, Y {br[2]:.3f} .. {br[3]:.3f}
     (= tabletop shrunk by container half {geo['cont_half']:.3f} + CONTAINER_EDGE_MARGIN {CONTAINER_EDGE_MARGIN:.3f})
  Every grid sample in the cube region passed the pick chain and every grid
  sample whose cell overlaps the container region passed the place chain
  against this exact table (validation level L2).

OPTIMIZATION
  Objective   : maximise tabletop area.
  Constraints : all sampled cube positions reachable by the scene randomizer are
                pick-feasible; all container cells are place-feasible; table does
                not touch the robot at the home pose; sides >= {args.min_table_side:.2f} m;
                a layout with cube-to-container gap >= {CUBE_CONTAINER_GAP:.2f} m exists;
                sampled pick->transfer->place pairs all pass (L3).
  Tie-break   : more feasible cells, then table centred closer to y = 0.
  Search      : screening at coarse heights {list(map(float, coarse))} (step {args.z_coarse_step} m),
                refinement step {args.z_fine_step} m around the best two; exhaustive search
                over all grid-aligned rectangles per height; best candidate re-validated
                with the full table and repaired/re-optimized up to {args.validation_rounds} rounds.
  Heights evaluated: {sorted(map(float, height_scores))}
  Area per height  : {', '.join(f"{h:.3f}:{height_scores[h]['area']:.3f}" for h in sorted(height_scores))}
                     (after validation repairs for the heights that were validated)

SAMPLING / MODEL
  XY grid      : X {xs[0]:.3f}..{xs[-1]:.3f}, Y {ys[0]:.3f}..{ys[-1]:.3f}, step {args.xy_step:.3f} m ({len(xs)} x {len(ys)})
  Grasp yaws   : {GRASP_YAWS_DEG} deg (first feasible kept)
  Orientation  : hand/site +Z (finger approach axis) along world -Z, i.e. R = Rz(yaw)*Rx(pi)
  TCP          : midpoint of left/right_finger_pad (the physical contact pads),
                 offset in 'gripper' site frame {np.round(ctx.tcp_offset_site, 4).tolist()} m
  Grasp depth  : TCP {ctx.grasp_dz * 1000:.1f} mm above cube centre; lowest finger point
                 {ctx.tip_below_tcp * 1000:.1f} mm below TCP; fingertip-table clearance >= {FINGERTIP_TABLE_CLEARANCE * 1000:.0f} mm;
                 pad/cube-face overlap {ctx.pad_overlap * 1000:.1f} mm
  Fingers      : contact at finger_joint = {ctx.q_contact:.4f} m, open {FINGER_OPEN} m
                 (open gap {ctx.open_gap * 1000:.1f} mm for a {2 * geo['cube_half'] * 1000:.0f} mm cube)
  Release      : open Panda fingers span {2 * ctx.finger_halfwidth['open'] * 1000:.1f} mm along the closing
                 axis vs a {2 * geo['cont_inner_half'] * 1000:.1f} mm container opening, so the gripper
                 {'opens with its fingertips ' + format(FINGER_WALL_CLEARANCE * 1000, '.0f') + ' mm above the wall top; the cube (already inside the walls) drops ' + format(ctx.release_drop * 100, '.1f') + ' cm onto the base' if ctx.release_above_walls else 'opens inside the container'}.
  IK           : Mink FrameTask on site '{SITE_NAME}' + PostureTask (cost {IK_POSTURE_COST}) +
                 ConfigurationLimit, solver {SOLVER}; tolerances {POS_TOL * 1000:.1f} mm / {ORI_TOL:.3f} rad
                 (measured by forward kinematics); seeds from the ready pose with joint1
                 toward the target and joint7 set for the requested yaw.
  Motions      : vertical approach {APPROACH_HEIGHT} m, lift {LIFT_HEIGHT} m, place approach {PLACE_APPROACH} m,
                 waypoints every {PATH_STEP * 100:.0f} cm (IK-tracked, joint jump <= {MAX_JOINT_JUMP} rad);
                 transfer joint-space straight line, checked every {TRANSFER_JOINT_STEP} rad
                 (fallback: one raised via-point).
  Margins      : robot {ROBOT_MARGIN * 1000:.0f} mm, held cube {CUBE_MARGIN * 1000:.0f} mm,
                 finger/cube penetration allowed {FINGER_CUBE_TOL * 1000:.0f} mm only while grasping.

FEASIBILITY COUNTS
  Sampled cells (all heights)              : {tot}
  L0 IK-reachable grasp pose               : {sum(c['ik_grasp_reachable'] for c in cells.values())}
  L1 pick chain feasible (screened)        : {sum(bool(c.get('pick_ok')) for c in cells.values())}
  L1 place chain feasible (screened)       : {sum(bool(c.get('place_ok')) for c in cells.values())}
  L1 pick AND place feasible (screened)    : {sum(c['task_ok'] for c in cells.values())}
  At selected height Z={best_top:.3f}:
    L2 cube cells validated                : {len(rep['pick'])}
    L2 container cells validated           : {len(rep['place'])}
    L3 pick->transfer->place pairs         : {n_pairs_ok}/{n_pairs} passed ({n_via} needed the via-point)
    L4 physics replays                     : {n_phys_ok}/{len(phys)} succeeded

VALIDATED vs APPROXIMATED
  Validated (geometric/kinematic, at samples): IK within tolerance and joint limits;
    signed-distance clearance of all Panda collision geoms to table top, legs, floor,
    cube, container; finger/pad contact geometry at closure; held-cube clearance;
    distal-vs-proximal self-collision; IK-tracked vertical motions.
  Physically simulated: only the {len(phys)} replayed pairs (PD + gravity-compensated
    arm tracking the plan, dynamic fingers/cube/container) - grasp stability elsewhere is NOT simulated.
  Approximations / limitations:
    - Continuous space between grid samples and between waypoints is not verified.
    - Screening (L1) uses a {2 * PATCH_HALF:.2f} m local table patch and distal robot links only;
      the recommended table is re-checked in full (L2/L3).
    - Mesh collision uses convex hulls (conservative); hand_capsule is included.
    - Self-collision only distal (link6/7, hand, fingers) vs link0-3.
    - Cube/container yaw = 0; random yaw requires a matching wrist yaw
      (verify with the scene's randomized layouts).
    - Transfer is not a general motion planner; only sampled pairs are verified.
    - Home pose is the Panda ready pose {Q_HOME.tolist()} (XML qpos0 has joint4 = 0,
      which is outside its limits and is not used).
    - The lab scene's torque/velocity controller and timing are not simulated.
"""
    sp = RESULTS / "franka_dexterous_workspace_table_optimization.txt"
    sp.write_text(summary)
    log("\n=== RECOMMENDED FRANKA TABLE ===")
    log(f"Top Z {best_top:.3f} m | size {lx:.3f} x {ly:.3f} m | centre ({tcx:.3f}, {tcy:.3f})")
    log(f"Pairs {n_pairs_ok}/{n_pairs}, physics replays {n_phys_ok}/{len(phys)}")
    log(f"CSV: {csv_path}\nPairs CSV: {pair_csv}")
    for p in paths:
        log(f"Plot: {p}")
    log(f"Summary: {sp}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
