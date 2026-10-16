#!/usr/bin/env python3
"""Batch experiments: HEAL and Franka pick-and-place with Mink, DLS and QP IK.

Runs every (robot, IK method, episode seed) combination headlessly by importing the existing,
unmodified task scripts and calling their own ``build_world`` / ``PickPlace.run`` entry points.
Each episode runs in its own subprocess (crash/timeout isolation); results are appended to disk
after every episode so an interrupted batch keeps its partial results and can be resumed.

Fair comparison: for a given robot the same episode seed produces the same cube/container layout
for all three IK methods (the layout depends only on the seed and is verified per episode against
a reference layout computed before the batch starts). By default both robots use the same seeds.

Usage (from the repository root):
  python3 Lab/lab03-ik-pick-and-place-heal-frnaka/scripts/run_experiments.py
Common options:
  --episodes 25 --base-seed 639 --robots heal franka --methods mink dls qp
  --workers 1 --episode-timeout 1200 --mujoco-gl egl --no-screenshots
  --resume <run_dir>            continue an interrupted run (skips finished episodes)
  --summarize-only <run_dir>    rebuild summaries/plots from an existing run
  --validate-only               check imports, models, seeds and layouts, then exit

Outputs go to Lab/lab03-ik-pick-and-place-heal-frnaka/results/experiments/run_<timestamp>/ :
  config.json            experiment configuration, seeds, versions, task parameters, file hashes
  layouts.csv            reference initial cube/container poses per robot and seed
  episodes.jsonl/.csv    one record per episode (all metrics; unavailable ones marked as such)
  summary.csv            aggregate metrics per robot x IK method
  failure_summary.csv    failure counts per category with example seeds/messages
  pairwise_tests.csv     paired (same-seed) exact McNemar tests between IK methods
  report.md              human-readable summary, failure analysis and mitigation notes
  plots/                 comparison plots (per robot and joint), layout/workspace plots
  screenshots/           rendered failure/success states (if an OpenGL context is available)
  failure_states/        top-down schematic of every failure (always produced)
  logs/                  full stdout/stderr of each episode
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as _dt
import hashlib
import importlib.util
import io
import json
import math
import multiprocessing as mp
import os
import platform
import random
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
LAB_DIR = SCRIPT_DIR.parent
REPO_DIR = LAB_DIR.parent.parent
RESULTS_ROOT = LAB_DIR / "results" / "experiments"

ROBOTS = ("heal", "franka")
METHODS = ("mink", "dls", "qp")
ROBOT_LABELS = {"heal": "HEAL", "franka": "Franka Panda"}
METHOD_LABELS = {"mink": "Mink", "dls": "DLS", "qp": "QP"}
# Task-script candidates (first existing file is used; override with --heal-script/--franka-script)
TASK_SCRIPT_CANDIDATES = {
    "heal": ("HealPickPlace.py", "HealPickPlace_updated2.py", "heal_pick_place.py"),
    "franka": ("FrankaPickPlace.py", "franka_pick_place.py"),
}
REQUIRED_TASK_ATTRS = ("build_world", "PickPlace", "TaskError", "IK_METHOD_LABELS",
                       "find_by_suffix", "scene", "GuardedIK")
UNAVAILABLE = "unavailable"
CLEARANCE_CAP = 0.25          # m, distances beyond this are reported as ">= cap"
POS_LIMIT_TOL = 1e-3          # rad/m beyond jnt_range counted as a position-limit violation
VEL_LIMIT_TOL = 0.10          # fractional overshoot of the task speed limit tolerated (measured)

EPISODE_FIELDS = [
    "robot", "ik_method", "seed", "episode_index", "status", "success", "placement_verified",
    "failure_category", "failure_stage", "failure_message", "exception_type",
    "init_cube_x", "init_cube_y", "init_cube_z", "init_cube_yaw_deg",
    "init_container_x", "init_container_y", "init_container_z", "init_container_yaw_deg",
    "layout_hash", "layout_matches_reference",
    "final_cube_x", "final_cube_y", "final_cube_z", "final_container_x", "final_container_y",
    "final_container_z", "placement_report",
    "ik_solve_time_total_s", "ik_solve_time_mean_ms", "ik_solve_time_p95_ms",
    "ik_solve_time_max_ms", "ik_solver_calls", "ik_iterations", "ik_exceptions",
    "ik_near_singular_calls", "ik_min_sigma", "ik_active_limit_calls",
    "setup_time_s", "episode_wall_time_s", "sim_time_s", "control_steps",
    "min_clearance_m", "min_clearance_table_m", "min_clearance_container_m",
    "min_clearance_floor_m", "min_clearance_pair", "clearance_samples", "clearance_note",
    "joint_pos_limit_violation_steps", "joint_pos_limit_max_violation", "joint_pos_limits_note",
    "commanded_speed_violation_steps", "max_commanded_joint_speed",
    "measured_speed_violation_steps", "max_measured_joint_speed", "joint_speed_limit",
    "state_at_end_tcp", "state_at_end_q", "screenshot", "log_file", "mujoco_version",
]


# ======================================================================================
# Small utilities
# ======================================================================================
def now_stamp():
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if math.isfinite(v) else str(v)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if isinstance(x, Path):
        return str(x)
    return x


def resolve_task_script(robot, override=None):
    if override:
        p = Path(override)
        p = p if p.is_absolute() else (Path.cwd() / p)
        if not p.is_file():
            raise FileNotFoundError(f"{robot} task script not found: {p}")
        return p.resolve()
    for name in TASK_SCRIPT_CANDIDATES[robot]:
        p = SCRIPT_DIR / name
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(f"No {robot} task script found in {SCRIPT_DIR}; looked for "
                            f"{', '.join(TASK_SCRIPT_CANDIDATES[robot])}")


def load_task_module(robot, path):
    """Import a task script by path without running its main() (no prompts, no viewer)."""
    for p in (str(LAB_DIR), str(SCRIPT_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
    mod_name = f"_task_{robot}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    missing = [a for a in REQUIRED_TASK_ATTRS if not hasattr(module, a)]
    if missing:
        raise AttributeError(f"{path.name} does not expose {missing}")
    for m in METHODS:
        if m not in module.IK_METHOD_LABELS:
            raise AttributeError(f"{path.name} does not support IK method {m!r}")
    return module


def quat_yaw_deg(q):
    w, x, y, z = q
    return float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))


def extract_layout(mod, model, data):
    import mujoco
    out = {}
    for key, jname in (("cube", "cube_free"), ("container", "container_free")):
        j = mod.find_by_suffix(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        a = int(model.jnt_qposadr[j])
        q = np.array(data.qpos[a:a + 7], dtype=float)
        out[key] = dict(x=float(q[0]), y=float(q[1]), z=float(q[2]), yaw_deg=quat_yaw_deg(q[3:7]))
    rounded = [round(out[k][f], 9) for k in ("cube", "container") for f in ("x", "y", "z", "yaw_deg")]
    out["hash"] = hashlib.sha1(json.dumps(rounded).encode()).hexdigest()[:16]
    return out


def table_geometry(mod, model, data):
    import mujoco
    mujoco.mj_forward(model, data)
    x0, x1, y0, y1 = mod.scene.table_bounds(model, data)
    return dict(x_min=x0, x_max=x1, y_min=y0, y_max=y1,
                top_z=float(mod.scene.table_top_height(model, data)))


def module_task_parameters(mod):
    names = ("APPROACH_HEIGHT", "GRASP_Z_CLEARANCE", "GRIPPER_OPEN_CMD", "GRIPPER_CLOSE_CMD",
             "CONTROL_DT", "VELOCITY_KD", "IK_GAIN", "IK_DAMPING", "MAX_JOINT_SPEED",
             "JOINT_HOLD_KP", "SPEED_TRANSIT", "SPEED_PRECISE", "YAW_RATE", "POS_TOL", "TILT_TOL",
             "YAW_TOL", "CUBE_HALF", "CONTAINER_INNER_HALF", "CUBE_EDGE_MARGIN",
             "CONTAINER_EDGE_MARGIN", "GRIPPER_CLEARANCE", "YAW_RANGE_DEG", "PLACE_YAW_TOL",
             "MAX_CONSECUTIVE_IK_FAILURES")
    return {n: jsonable(getattr(mod, n)) for n in names if hasattr(mod, n)}


# ======================================================================================
# Signed distance with a GJK cross-check (MuJoCo 3.11 can return spurious exact zeros)
# ======================================================================================
class Clearance:
    def __init__(self, model, data, robot_geoms, groups):
        self.m, self.d = model, data
        self.robot_geoms = list(robot_geoms)
        self.groups = groups          # name -> list of obstacle geoms

    def _points(self, g):
        import mujoco
        m, d = self.m, self.d
        t = m.geom_type[g]
        R = d.geom_xmat[g].reshape(3, 3)
        c = d.geom_xpos[g]
        sz = m.geom_size[g]
        G = mujoco.mjtGeom
        if t == G.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            return c + m.mesh_vert[a:a + n] @ R.T, 0.0
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
    def _closest(W):
        best, best_idx = None, None
        n = len(W)
        for mask in range(1, 1 << n):
            idx = [i for i in range(n) if mask >> i & 1]
            P = W[idx]
            if len(idx) == 1:
                lam = np.array([1.0])
            else:
                mu = np.linalg.lstsq((P[1:] - P[0]).T, -P[0], rcond=None)[0]
                lam = np.r_[1.0 - mu.sum(), mu]
                if np.any(lam < -1e-12):
                    continue
            x = lam @ P
            if best is None or x @ x < best @ best - 1e-15:
                best, best_idx = x, idx
        return best, best_idx

    def _gjk(self, A, B):
        v = A[0] - B[0]
        W = np.empty((0, 3))
        for _ in range(64):
            vv = float(v @ v)
            if vv < 1e-18:
                return 0.0
            w = A[np.argmin(A @ v)] - B[np.argmax(B @ v)]
            if vv - float(v @ w) <= 1e-10 * max(vv, 1e-12) + 1e-14:
                return math.sqrt(vv)
            W = np.vstack([W, w])
            v, idx = self._closest(W)
            W = W[idx]
            if len(W) == 4:
                return 0.0
        return math.sqrt(float(v @ v))

    def dist(self, a, b, cap):
        import mujoco
        dd = mujoco.mj_geomDistance(self.m, self.d, a, b, cap, None)
        if dd > 0.0:
            return dd
        # dd <= 0: confirm with GJK. Some MuJoCo builds return a spurious exact 0 or a spurious
        # negative value for separated convex pairs; GJK cannot measure penetration depth, so a
        # confirmed intersection keeps MuJoCo's (negative) value.
        G = mujoco.mjtGeom
        ta, tb = self.m.geom_type[a], self.m.geom_type[b]
        if G.mjGEOM_PLANE in (ta, tb):
            pl, other = (a, b) if ta == G.mjGEOM_PLANE else (b, a)
            P, r = self._points(other)
            if P is None:
                return dd
            n = self.d.geom_xmat[pl].reshape(3, 3)[:, 2]
            return min(cap, float(np.min((P - self.d.geom_xpos[pl]) @ n)) - r)
        A, ra = self._points(a)
        B, rb = self._points(b)
        if A is None or B is None:
            return dd
        sep = self._gjk(A, B)
        if sep <= 1e-9:
            return dd                       # intersection confirmed
        return min(cap, sep - ra - rb)

    def sample(self):
        """Min distance per obstacle group, using bounding spheres to skip far pairs."""
        import mujoco
        m, d = self.m, self.d
        out = {}
        for name, obs in self.groups.items():
            best, pair = CLEARANCE_CAP, None
            for a in self.robot_geoms:
                pa, ra = d.geom_xpos[a], m.geom_rbound[a]
                for b in obs:
                    if m.geom_type[b] != mujoco.mjtGeom.mjGEOM_PLANE and \
                            m.geom_type[a] != mujoco.mjtGeom.mjGEOM_PLANE and \
                            np.linalg.norm(pa - d.geom_xpos[b]) - ra - m.geom_rbound[b] > best:
                        continue
                    dd = self.dist(a, b, CLEARANCE_CAP)
                    if dd < best:
                        best, pair = dd, (a, b)
            out[name] = (best, pair)
        return out


# ======================================================================================
# One episode (runs inside a child process)
# ======================================================================================
def classify_failure(message, exc_type):
    m = (message or "").lower()
    rules = (
        ("consecutive ik updates", "ik_solver_failure"),
        ("invalid arm joint-velocity", "invalid_ik_output"),
        ("collided with the container", "container_collision"),
        ("grasp failed", "grasp_failed_no_pad_contact"),
        ("grasp lost during lift", "grasp_lost_during_lift"),
        ("slipped out of the gripper", "cube_slipped_during_transfer"),
        ("not placed inside the container", "placement_failed"),
        ("not parallel", "yaw_alignment_failed"),
        ("fingers never came to rest", "gripper_close_timeout"),
        ("release: timed out", "gripper_open_timeout"),
        ("open gripper: timed out", "gripper_open_timeout"),
        ("cube landing", "cube_landing_timeout"),
        ("object settle", "object_settle_timeout"),
        ("not resting on the table", "object_settle_failed"),
        ("joint move timed out", "return_to_neutral_timeout"),
        ("no cube/container layout", "layout_sampling_failed"),
        ("cannot measure tcp", "tcp_measurement_failed"),
        ("unexpected actuator order", "model_mapping_error"),
        ("not the optimized franka table", "table_mismatch"),
    )
    for key, cat in rules:
        if key in m:
            return cat
    if "timed out" in m:
        label = m.split(":", 1)[0].strip().replace(" ", "_").replace("(", "").replace(")", "")
        return f"motion_timeout_{label}" if label else "motion_timeout"
    return f"exception_{exc_type}" if exc_type else "unknown"


def render_state(model, data, path, center, width=640, height=480):
    """Offscreen screenshot; returns path or an explanation string."""
    try:
        import mujoco
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        w = min(width, int(model.vis.global_.offwidth))
        h = min(height, int(model.vis.global_.offheight))
        renderer = mujoco.Renderer(model, h, w)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = center
        cam.distance, cam.azimuth, cam.elevation = 1.4, 150.0, -35.0
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        renderer.close()
        plt.imsave(str(path), img)
        return str(path)
    except Exception as exc:  # no OpenGL context etc.
        return f"{UNAVAILABLE}: {type(exc).__name__}: {exc}"


def run_episode(spec):
    """Run one headless episode by driving the task script's own objects. Returns a dict."""
    if spec.get("mujoco_gl"):
        os.environ["MUJOCO_GL"] = spec["mujoco_gl"]
    import mujoco

    robot, method, seed = spec["robot"], spec["ik_method"], int(spec["seed"])
    rec = {f: UNAVAILABLE for f in EPISODE_FIELDS}
    rec.update(robot=robot, ik_method=method, seed=seed, episode_index=spec["episode_index"],
               status="error", success=False, placement_verified=UNAVAILABLE,
               failure_category="", failure_stage="", failure_message="", exception_type="",
               log_file=spec["log_file"], mujoco_version=mujoco.__version__, screenshot="")
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))

    t_setup = time.perf_counter()
    mod = load_task_module(robot, Path(spec["task_script"]))
    mod.IK_METHOD = method             # the task scripts read this module global
    model, data = mod.build_world(True, seed, True)
    layout = extract_layout(mod, model, data)
    rec.update(init_cube_x=layout["cube"]["x"], init_cube_y=layout["cube"]["y"],
               init_cube_z=layout["cube"]["z"], init_cube_yaw_deg=layout["cube"]["yaw_deg"],
               init_container_x=layout["container"]["x"], init_container_y=layout["container"]["y"],
               init_container_z=layout["container"]["z"],
               init_container_yaw_deg=layout["container"]["yaw_deg"], layout_hash=layout["hash"])
    ref = spec.get("reference_layout_hash")
    rec["layout_matches_reference"] = (layout["hash"] == ref) if ref else UNAVAILABLE
    tgeo = table_geometry(mod, model, data)

    runner = mod.PickPlace(model, data, viewer=None, realtime=False)
    rec["setup_time_s"] = time.perf_counter() - t_setup

    # ---------------- instrumentation (wrappers on this runner instance only) ----------------
    stats = dict(ik_times=[], ik_exc=0, near_sing=0, min_sigma=math.inf, active=0,
                 steps=0, cmd_viol=0, max_cmd=0.0, meas_viol=0, max_meas=0.0,
                 pos_viol=0, pos_max=0.0, clear=dict(), clear_pair="", clear_samples=0,
                 stage="")
    guarded = isinstance(runner.ik, mod.GuardedIK)
    inner = runner.ik.solver if guarded else runner.ik
    orig_solve = inner.compute_joint_velocities

    def timed_solve(*a, **k):
        t0 = time.perf_counter()
        try:
            out = orig_solve(*a, **k)
        except Exception:
            stats["ik_exc"] += 1
            raise
        finally:
            stats["ik_times"].append(time.perf_counter() - t0)
        last = getattr(inner, "last", None)
        if last is not None:
            stats["min_sigma"] = min(stats["min_sigma"], float(last.sigma_min))
            stats["near_sing"] += int(bool(last.near_singular))
            stats["active"] += int(bool(getattr(last, "active_limits", ())))
        return out
    inner.compute_joint_velocities = timed_solve

    speed_limit = float(getattr(mod, "MAX_JOINT_SPEED", math.nan))
    jids = runner.arm_joint_ids
    limited = np.array([bool(model.jnt_limited[j]) for j in jids])
    q_lo = np.where(limited, model.jnt_range[jids, 0], -np.inf)
    q_hi = np.where(limited, model.jnt_range[jids, 1], np.inf)

    # Robot collision geoms. Geoms on bodies that cannot move (the base, welded to the world)
    # rest on the floor by design and are excluded from the clearance metric.
    moving = {b for b in runner.robot_bodies if int(model.body_weldid[b]) != 0}
    robot_geoms = [g for g in range(model.ngeom)
                   if int(model.geom_bodyid[g]) in moving and
                   (model.geom_contype[g] or model.geom_conaffinity[g] or model.geom_group[g] == 3)]
    table_body = mod.find_by_suffix(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    groups = {
        "table": [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == table_body],
        "container": [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == runner.box_body],
    }
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor >= 0:
        groups["floor"] = [floor]
    clearance = Clearance(model, data, robot_geoms, groups)
    every = max(1, int(spec["clearance_every"]))

    orig_advance = runner._advance

    def instrumented_advance(arm_velocity):
        v = np.asarray(arm_velocity, dtype=float)
        if v.size:
            cmd = float(np.max(np.abs(v)))
            stats["max_cmd"] = max(stats["max_cmd"], cmd)
            if math.isfinite(speed_limit) and cmd > speed_limit + 1e-6:
                stats["cmd_viol"] += 1
        try:
            orig_advance(arm_velocity)
        finally:
            stats["steps"] += 1
            qd = np.abs(data.qvel[runner.arm_dofadr])
            meas = float(np.max(qd))
            stats["max_meas"] = max(stats["max_meas"], meas)
            if math.isfinite(speed_limit) and meas > speed_limit * (1.0 + VEL_LIMIT_TOL):
                stats["meas_viol"] += 1
            q = data.qpos[runner.arm_qadr]
            over = np.maximum(q - q_hi, q_lo - q)
            over = float(np.max(np.where(limited, over, -np.inf))) if limited.any() else -np.inf
            if over > POS_LIMIT_TOL:
                stats["pos_viol"] += 1
            stats["pos_max"] = max(stats["pos_max"], over if math.isfinite(over) else 0.0)
            if stats["steps"] % every == 0:
                mujoco.mj_kinematics(model, data)
                for name, (dmin, pair) in clearance.sample().items():
                    if dmin < stats["clear"].get(name, (math.inf, None))[0]:
                        stats["clear"][name] = (dmin, pair)
                stats["clear_samples"] += 1
    runner._advance = instrumented_advance

    orig_log = runner.log

    def tracking_log(msg):
        s = str(msg).strip()
        if s.startswith("STAGE"):
            stats["stage"] = s.split(":", 1)[0]
        orig_log(msg)
    runner.log = tracking_log

    # ---------------- run the unmodified task ----------------
    mujoco.set_mjcb_control(runner.controller.control_callback)
    t0 = time.perf_counter()
    try:
        runner.run()
        rec.update(status="completed", success=True)
    except mod.TaskError as exc:
        rec.update(status="failed", success=False, failure_message=str(exc),
                   exception_type="TaskError")
    except Exception as exc:
        rec.update(status="failed", success=False, exception_type=type(exc).__name__,
                   failure_message=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}")
    finally:
        mujoco.set_mjcb_control(None)
    rec["episode_wall_time_s"] = time.perf_counter() - t0
    if not rec["success"]:
        rec["failure_category"] = classify_failure(rec["failure_message"], rec["exception_type"])
        rec["failure_stage"] = stats["stage"] or "setup"

    # ---------------- final state and metrics ----------------
    try:
        ok, report = runner.verify_placement()
        rec["placement_verified"] = bool(ok)
        rec["placement_report"] = report
    except Exception as exc:
        rec["placement_report"] = f"{UNAVAILABLE}: {exc}"
    try:
        c, b = runner.cube_pos(), runner.box_pos()
        rec.update(final_cube_x=c[0], final_cube_y=c[1], final_cube_z=c[2],
                   final_container_x=b[0], final_container_y=b[1], final_container_z=b[2])
        rec["state_at_end_tcp"] = json.dumps([round(float(v), 5) for v in runner.tcp()])
        rec["state_at_end_q"] = json.dumps([round(float(v), 5) for v in runner.q_arm()])
    except Exception:
        pass
    rec["sim_time_s"] = float(data.time)
    rec["control_steps"] = stats["steps"]
    times = np.array(stats["ik_times"])
    rec["ik_solver_calls"] = int(times.size)
    if times.size:
        rec["ik_solve_time_total_s"] = float(times.sum())
        rec["ik_solve_time_mean_ms"] = float(times.mean() * 1e3)
        rec["ik_solve_time_p95_ms"] = float(np.percentile(times, 95) * 1e3)
        rec["ik_solve_time_max_ms"] = float(times.max() * 1e3)
    else:
        rec["ik_solve_time_total_s"] = 0.0
    rec["ik_iterations"] = (f"{UNAVAILABLE}: differential IK performs one "
                            f"{'QP' if method in ('mink', 'qp') else 'damped least-squares'} solve per "
                            "control step; solver-internal iterations are not exposed "
                            "(see ik_solver_calls)")
    rec["ik_exceptions"] = stats["ik_exc"]
    if method == "mink":
        rec["ik_near_singular_calls"] = f"{UNAVAILABLE}: not reported by MinkIK"
        rec["ik_min_sigma"] = f"{UNAVAILABLE}: not reported by MinkIK"
        rec["ik_active_limit_calls"] = f"{UNAVAILABLE}: not reported by MinkIK"
    else:
        rec["ik_near_singular_calls"] = stats["near_sing"]
        rec["ik_min_sigma"] = stats["min_sigma"] if math.isfinite(stats["min_sigma"]) else UNAVAILABLE
        rec["ik_active_limit_calls"] = stats["active"] if method == "qp" else \
            f"{UNAVAILABLE}: DLS has no joint-limit constraints"
    if stats["clear"]:
        names = {g: (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or
                     f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g]))}"
                     f"#geom{g}")
                 for g in range(model.ngeom)}
        for key in ("table", "container", "floor"):
            if key in stats["clear"]:
                rec[f"min_clearance_{key}_m"] = float(stats["clear"][key][0])
        overall = min(stats["clear"].items(), key=lambda kv: kv[1][0])
        rec["min_clearance_m"] = float(overall[1][0])
        pair = overall[1][1]
        rec["min_clearance_pair"] = f"{overall[0]}: {names[pair[0]]}|{names[pair[1]]}" if pair else \
            f"none within {CLEARANCE_CAP} m"
    rec["clearance_samples"] = stats["clear_samples"]
    rec["clearance_note"] = (f"signed distance between moving robot collision geoms and table/container/"
                             f"floor geoms (cube excluded: intended contact), sampled every {every} "
                             f"control step(s); values >= {CLEARANCE_CAP} m are capped")
    rec["joint_pos_limit_violation_steps"] = stats["pos_viol"]
    rec["joint_pos_limit_max_violation"] = stats["pos_max"]
    rec["joint_pos_limits_note"] = (f"{int(limited.sum())}/{len(jids)} arm joints have XML limits; "
                                    f"violation = beyond range by > {POS_LIMIT_TOL}")
    rec["commanded_speed_violation_steps"] = stats["cmd_viol"]
    rec["max_commanded_joint_speed"] = stats["max_cmd"]
    rec["measured_speed_violation_steps"] = stats["meas_viol"]
    rec["max_measured_joint_speed"] = stats["max_meas"]
    rec["joint_speed_limit"] = (f"{speed_limit} rad/s (task MAX_JOINT_SPEED; measured violation = "
                                f"> {1 + VEL_LIMIT_TOL:.2f} x limit; the XML defines no joint "
                                "velocity limits)")

    # ---------------- screenshots ----------------
    want = spec.get("screenshot_failure") if not rec["success"] else spec.get("screenshot_success")
    if want:
        center = [0.5 * (tgeo["x_min"] + tgeo["x_max"]), 0.5 * (tgeo["y_min"] + tgeo["y_max"]),
                  tgeo["top_z"]]
        rec["screenshot"] = render_state(model, data, Path(want), center)
    return rec


def _child_entry(spec, conn):
    log_path = Path(spec["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", buffering=1) as log:
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                rec = run_episode(spec)
        except BaseException as exc:  # setup failures, import errors, etc.
            msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}"
            log.write("\nRUNNER ERROR: " + msg)
            rec = {f: UNAVAILABLE for f in EPISODE_FIELDS}
            rec.update(robot=spec["robot"], ik_method=spec["ik_method"], seed=spec["seed"],
                       episode_index=spec["episode_index"], status="error", success=False,
                       failure_category=classify_failure(str(exc), type(exc).__name__),
                       failure_stage="setup", failure_message=msg,
                       exception_type=type(exc).__name__, log_file=spec["log_file"],
                       screenshot="", placement_verified=UNAVAILABLE)
        log.write(f"\nEPISODE RESULT: success={rec.get('success')} "
                  f"category={rec.get('failure_category')}\n")
    try:
        conn.send(jsonable(rec))
    finally:
        conn.close()


# ======================================================================================
# Validation (before the batch)
# ======================================================================================
def _validate_robot(robot, script, seeds, out_q):
    """Child: import module, check API, build layouts for every seed, construct each IK."""
    try:
        import mujoco
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mod = load_task_module(robot, Path(script))
            layouts, tables = {}, None
            for s in seeds:
                model, data = mod.build_world(True, int(s), True)
                layouts[int(s)] = extract_layout(mod, model, data)
                if tables is None:
                    tables = table_geometry(mod, model, data)
            # determinism: rebuild the first seeds and compare
            for s in seeds[:3]:
                model, data = mod.build_world(True, int(s), True)
                if extract_layout(mod, model, data)["hash"] != layouts[int(s)]["hash"]:
                    raise RuntimeError(f"layout for seed {s} is not deterministic")
            solver_types = {}
            model, data = mod.build_world(True, int(seeds[0]), True)
            before = extract_layout(mod, model, data)["hash"]
            for m in METHODS:
                mod.IK_METHOD = m
                runner = mod.PickPlace(model, data, viewer=None, realtime=False)
                inner = runner.ik.solver if isinstance(runner.ik, mod.GuardedIK) else runner.ik
                solver_types[m] = f"{type(inner).__module__}.{type(inner).__name__}"
                if not hasattr(inner, "compute_joint_velocities"):
                    raise RuntimeError(f"{m}: solver has no compute_joint_velocities()")
            if extract_layout(mod, model, data)["hash"] != before:
                raise RuntimeError("constructing the task changed the initial layout")
            info = dict(
                ok=True, robot=robot, script=str(script), sha256=sha256(script),
                layouts=layouts, table=tables, solver_types=solver_types,
                arm_joints=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                            for j in runner.arm_joint_ids],
                ee_site=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, runner.site_id),
                nu=int(model.nu), nq=int(model.nq), nv=int(model.nv),
                task_parameters=module_task_parameters(mod),
                scene_module=getattr(mod.scene, "__file__", UNAVAILABLE),
            )
        out_q.put(jsonable(info))
    except BaseException as exc:
        out_q.put(dict(ok=False, robot=robot, script=str(script),
                       error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}"))


def validate(robots, scripts, seeds, timeout=600):
    ctx = mp.get_context("spawn")
    results = {}
    for robot in robots:
        q = ctx.Queue()
        p = ctx.Process(target=_validate_robot, args=(robot, str(scripts[robot]), seeds, q))
        p.start()
        try:
            results[robot] = q.get(timeout=timeout)
        except Exception as exc:
            results[robot] = dict(ok=False, robot=robot, error=f"validation timed out/crashed: {exc}")
        p.join(10)
        if p.is_alive():
            p.terminate()
    return results


# ======================================================================================
# Batch execution
# ======================================================================================
class Batch:
    def __init__(self, run_dir, args, ref_layouts, scripts):
        self.run_dir = run_dir
        self.args = args
        self.ref = ref_layouts
        self.scripts = scripts
        self.jsonl = run_dir / "episodes.jsonl"
        self.csv = run_dir / "episodes.csv"
        self.done = self._load_done()
        self.shots = {}
        for r in self._records():
            key = (r["robot"], r["ik_method"])
            s = self.shots.setdefault(key, {"fail": 0, "success": 0})
            if r.get("screenshot") and not str(r["screenshot"]).startswith(UNAVAILABLE):
                s["success" if r.get("success") else "fail"] += 1

    def _records(self):
        if not self.jsonl.is_file():
            return []
        out = []
        for line in self.jsonl.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out

    def _load_done(self):
        final = {"completed", "failed", "error", "timeout", "crashed"}
        if getattr(self.args, "retry_timeouts", False):
            final -= {"timeout", "crashed"}
        return {(r["robot"], r["ik_method"], int(r["seed"])) for r in self._records()
                if r.get("status") in final}

    def _append(self, rec):
        with open(self.jsonl, "a") as f:
            f.write(json.dumps(jsonable(rec)) + "\n")
            f.flush()
            os.fsync(f.fileno())
        new = not self.csv.is_file()
        with open(self.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=EPISODE_FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            row = {k: (v.replace("\n", " | ") if isinstance(v, str) else v)
                   for k, v in jsonable(rec).items()}
            w.writerow(row)

    def make_spec(self, robot, method, seed, idx):
        a = self.args
        tag = f"{robot}_{method}_seed{seed}"
        s = self.shots.setdefault((robot, method), {"fail": 0, "success": 0})
        shot_dir = self.run_dir / "screenshots"
        spec = dict(robot=robot, ik_method=method, seed=int(seed), episode_index=idx,
                    task_script=str(self.scripts[robot]),
                    log_file=str(self.run_dir / "logs" / f"{tag}.log"),
                    clearance_every=a.clearance_every, mujoco_gl=a.mujoco_gl,
                    reference_layout_hash=self.ref[robot][str(seed)]["hash"]
                    if str(seed) in self.ref[robot] else None,
                    screenshot_failure=None, screenshot_success=None)
        if not a.no_screenshots:
            if s["fail"] < a.max_failure_screenshots:
                spec["screenshot_failure"] = str(shot_dir / f"FAIL_{tag}.png")
            if s["success"] < 1:
                spec["screenshot_success"] = str(shot_dir / f"SUCCESS_{tag}.png")
        return spec

    def _finish(self, spec, rec):
        key = (spec["robot"], spec["ik_method"])
        s = self.shots.setdefault(key, {"fail": 0, "success": 0})
        shot = rec.get("screenshot", "")
        if shot and not str(shot).startswith(UNAVAILABLE):
            s["success" if rec.get("success") else "fail"] += 1
        self._append(rec)
        self.done.add((spec["robot"], spec["ik_method"], int(spec["seed"])))

    def _failure_record(self, spec, status, message):
        rec = {f: UNAVAILABLE for f in EPISODE_FIELDS}
        rec.update(robot=spec["robot"], ik_method=spec["ik_method"], seed=spec["seed"],
                   episode_index=spec["episode_index"], status=status, success=False,
                   failure_category=status, failure_stage="unknown (process ended)",
                   failure_message=message, exception_type=status, log_file=spec["log_file"],
                   screenshot="", placement_verified=UNAVAILABLE)
        ref = self.ref[spec["robot"]].get(str(spec["seed"]))
        if ref:
            rec.update(init_cube_x=ref["cube"]["x"], init_cube_y=ref["cube"]["y"],
                       init_cube_z=ref["cube"]["z"], init_cube_yaw_deg=ref["cube"]["yaw_deg"],
                       init_container_x=ref["container"]["x"],
                       init_container_y=ref["container"]["y"],
                       init_container_z=ref["container"]["z"],
                       init_container_yaw_deg=ref["container"]["yaw_deg"],
                       layout_hash=ref["hash"], layout_matches_reference=True)
        return rec

    def run(self, jobs):
        ctx = mp.get_context("spawn")
        pending = [j for j in jobs if (j[0], j[1], int(j[2])) not in self.done]
        total, n_done = len(jobs), len(jobs) - len(pending)
        print(f"{n_done}/{total} episodes already recorded; {len(pending)} to run "
              f"with {self.args.workers} worker(s).", flush=True)
        active = []   # (process, conn, spec, t_start)
        t_batch = time.perf_counter()
        try:
            while pending or active:
                while pending and len(active) < self.args.workers:
                    robot, method, seed, idx = pending.pop(0)
                    spec = self.make_spec(robot, method, seed, idx)
                    parent, child = ctx.Pipe(duplex=False)
                    p = ctx.Process(target=_child_entry, args=(spec, child), daemon=True)
                    p.start()
                    child.close()
                    active.append((p, parent, spec, time.perf_counter()))
                time.sleep(0.2)
                for item in list(active):
                    p, conn, spec, t0 = item
                    rec = None
                    if conn.poll():
                        try:
                            rec = conn.recv()
                        except EOFError:
                            rec = None
                        p.join(30)
                        if rec is None:
                            rec = self._failure_record(spec, "crashed",
                                                       f"child exited with code {p.exitcode} "
                                                       "without a result")
                    elif not p.is_alive():
                        rec = self._failure_record(spec, "crashed",
                                                   f"child exited with code {p.exitcode} "
                                                   "without a result (see log)")
                    elif time.perf_counter() - t0 > self.args.episode_timeout:
                        p.terminate()
                        p.join(10)
                        rec = self._failure_record(
                            spec, "timeout",
                            f"episode exceeded the wall-clock limit of {self.args.episode_timeout} s")
                    if rec is None:
                        continue
                    active.remove(item)
                    conn.close()
                    self._finish(spec, rec)
                    n_done += 1
                    el = time.perf_counter() - t_batch
                    print(f"[{n_done:4d}/{total}] {spec['robot']:6s} {spec['ik_method']:4s} "
                          f"seed {spec['seed']:>10d}: "
                          f"{'SUCCESS' if rec.get('success') else 'FAIL   '} "
                          f"{rec.get('failure_category') or '':32s} "
                          f"({float(rec['episode_wall_time_s']) if isinstance(rec.get('episode_wall_time_s'), (int, float)) else float('nan'):6.1f} s) "
                          f"| elapsed {el / 60:6.1f} min", flush=True)
        except KeyboardInterrupt:
            print("\nInterrupted: terminating running episodes; finished results are kept.",
                  flush=True)
            for p, conn, spec, _ in active:
                p.terminate()
            raise


# ======================================================================================
# Analysis, summaries and plots
# ======================================================================================
def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def wilson(k, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


MITIGATIONS = {
    "ik_solver_failure": "Check solver availability/feasibility (QP bounds, joint-limit margins); "
                         "add a joint-limit or posture task, reduce IK_GAIN, or re-seed from neutral.",
    "invalid_ik_output": "Guard against NaN/shape errors in the solver output; inspect the Jacobian "
                         "near singular configurations.",
    "container_collision": "Increase the transfer height or GRIPPER_CLEARANCE, or route the "
                           "transfer through a higher via-point.",
    "grasp_failed_no_pad_contact": "Verify cube yaw alignment and grasp height; reduce approach "
                                   "speed or adjust GRASP_Z_CLEARANCE/TCP offset.",
    "grasp_lost_during_lift": "Lower SPEED_PRECISE / yaw rate during lift, increase finger "
                              "friction or gripper force.",
    "cube_slipped_during_transfer": "Lower SPEED_TRANSIT and YAW_RATE; avoid large wrist slews "
                                    "while carrying the cube.",
    "placement_failed": "Release lower/centred over the container; check drop yaw choice and "
                        "container yaw estimate.",
    "yaw_alignment_failed": "Wrist range insufficient for the requested yaw; allow the alternate "
                            "90 deg candidate or relax YAW_TOL.",
    "gripper_close_timeout": "Increase GRIPPER_CLOSE_TIMEOUT or check finger contact/damping.",
    "gripper_open_timeout": "Increase GRIPPER_OPEN_TIMEOUT; check for finger/obstacle contact.",
    "cube_landing_timeout": "Lower the release height; increase CUBE_LAND_TIMEOUT.",
    "object_settle_timeout": "Increase OBJECT_SETTLE_TIMEOUT; check initial object drop height.",
    "return_to_neutral_timeout": "Increase the joint-move timeout or check joint-limit contact.",
    "timeout": "Episode exceeded the wall-clock limit; profile the run or raise --episode-timeout.",
    "crashed": "Child process ended without a result; see the episode log (possible native crash).",
}


def mitigation_for(cat):
    if cat in MITIGATIONS:
        return MITIGATIONS[cat]
    if cat.startswith("motion_timeout"):
        return ("Target unreachable or tracking too slow with the required yaw/tilt: check "
                "workspace reach at this layout, joint limits (QP active limits), singularity "
                "diagnostics, and consider relaxing POS_TOL or reducing speed.")
    if cat.startswith("exception_"):
        return "Unexpected exception: inspect the traceback in the episode log."
    return "Inspect the episode log."


def load_records(run_dir):
    recs = []
    p = run_dir / "episodes.jsonl"
    if p.is_file():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    # keep the latest record per (robot, method, seed)
    latest = {}
    for r in recs:
        latest[(r["robot"], r["ik_method"], int(r["seed"]))] = r
    return list(latest.values())


def summarize(run_dir, robots, methods):
    recs = load_records(run_dir)
    if not recs:
        print("No episode records to summarize.")
        return
    cfg = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").is_file() else {}
    rows, fail_rows, pair_rows = [], [], []

    def col(rs, key):
        return [v for v in (_num(r.get(key)) for r in rs) if v is not None]

    for robot in robots:
        for method in methods:
            rs = [r for r in recs if r["robot"] == robot and r["ik_method"] == method]
            if not rs:
                continue
            n = len(rs)
            k = sum(bool(r.get("success")) for r in rs)
            lo, hi = wilson(k, n)
            ik_t = col(rs, "ik_solve_time_total_s")
            ik_ms = col(rs, "ik_solve_time_mean_ms")
            p95 = col(rs, "ik_solve_time_p95_ms")
            wall = col(rs, "episode_wall_time_s")
            sim = col(rs, "sim_time_s")
            clr = col(rs, "min_clearance_m")
            clr_s = [v for r in rs if r.get("success") for v in [_num(r.get("min_clearance_m"))]
                     if v is not None]
            calls = col(rs, "ik_solver_calls")
            cats = {}
            for r in rs:
                if not r.get("success"):
                    cats[r.get("failure_category") or "unknown"] = cats.get(
                        r.get("failure_category") or "unknown", 0) + 1

            def mean(v):
                return float(np.mean(v)) if v else UNAVAILABLE

            def med(v):
                return float(np.median(v)) if v else UNAVAILABLE

            def std(v):
                return float(np.std(v, ddof=1)) if len(v) > 1 else UNAVAILABLE

            rows.append(dict(
                robot=robot, ik_method=method, episodes=n, successes=k,
                success_rate=k / n, success_ci95_low=lo, success_ci95_high=hi,
                ik_time_per_episode_mean_s=mean(ik_t), ik_time_per_episode_median_s=med(ik_t),
                ik_time_per_episode_std_s=std(ik_t), ik_call_mean_ms=mean(ik_ms),
                ik_call_p95_ms_mean=mean(p95), ik_solver_calls_mean=mean(calls),
                ik_iterations=UNAVAILABLE + " (one solve per control step; see ik_solver_calls)",
                ik_exceptions_total=int(sum(col(rs, "ik_exceptions"))),
                episode_wall_time_mean_s=mean(wall), episode_wall_time_median_s=med(wall),
                sim_time_mean_s=mean(sim),
                min_clearance_mean_m=mean(clr), min_clearance_min_m=(min(clr) if clr else UNAVAILABLE),
                min_clearance_successful_mean_m=mean(clr_s),
                episodes_with_joint_pos_violation=sum(1 for v in col(rs, "joint_pos_limit_violation_steps") if v > 0),
                joint_pos_violation_steps_total=int(sum(col(rs, "joint_pos_limit_violation_steps"))),
                episodes_with_commanded_speed_violation=sum(1 for v in col(rs, "commanded_speed_violation_steps") if v > 0),
                episodes_with_measured_speed_violation=sum(1 for v in col(rs, "measured_speed_violation_steps") if v > 0),
                max_measured_joint_speed=(max(col(rs, "max_measured_joint_speed"))
                                          if col(rs, "max_measured_joint_speed") else UNAVAILABLE),
                layout_mismatches=sum(1 for r in rs if r.get("layout_matches_reference") is False),
                failures=n - k, failure_categories=json.dumps(cats, sort_keys=True),
            ))
            for cat, cnt in sorted(cats.items(), key=lambda kv: -kv[1]):
                ex = [r for r in rs if not r.get("success") and (r.get("failure_category") or "unknown") == cat]
                fail_rows.append(dict(
                    robot=robot, ik_method=method, category=cat, count=cnt,
                    fraction_of_episodes=cnt / n,
                    stages=json.dumps(sorted({str(r.get("failure_stage")) for r in ex})),
                    example_seeds=" ".join(str(r["seed"]) for r in ex[:5]),
                    example_message=str(ex[0].get("failure_message", ""))[:300].replace("\n", " | "),
                    mitigation=mitigation_for(cat)))
        # paired comparisons on common seeds
        for i, a in enumerate(methods):
            for bm in methods[i + 1:]:
                ra = {int(r["seed"]): bool(r.get("success")) for r in recs
                      if r["robot"] == robot and r["ik_method"] == a}
                rb = {int(r["seed"]): bool(r.get("success")) for r in recs
                      if r["robot"] == robot and r["ik_method"] == bm}
                common = sorted(set(ra) & set(rb))
                if not common:
                    continue
                b = sum(ra[s] and not rb[s] for s in common)
                c = sum(rb[s] and not ra[s] for s in common)
                pair_rows.append(dict(robot=robot, method_a=a, method_b=bm, paired_seeds=len(common),
                                      a_only_success=b, b_only_success=c,
                                      both_success=sum(ra[s] and rb[s] for s in common),
                                      both_fail=sum(not ra[s] and not rb[s] for s in common),
                                      mcnemar_exact_p=mcnemar_exact(b, c)))

    def write_csv(path, rws):
        if not rws:
            path.write_text("")
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rws[0].keys()))
            w.writeheader()
            for r in rws:
                w.writerow(jsonable(r))

    write_csv(run_dir / "summary.csv", rows)
    write_csv(run_dir / "failure_summary.csv", fail_rows)
    write_csv(run_dir / "pairwise_tests.csv", pair_rows)
    (run_dir / "summary.json").write_text(json.dumps(jsonable(dict(
        summary=rows, failures=fail_rows, pairwise=pair_rows)), indent=2))
    make_plots(run_dir, recs, rows, robots, methods, cfg)
    write_report(run_dir, recs, rows, fail_rows, pair_rows, robots, methods, cfg)
    print(f"Summaries written to {run_dir}")


def make_plots(run_dir, recs, rows, robots, methods, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    pdir = run_dir / "plots"
    pdir.mkdir(exist_ok=True)
    colors = {"mink": "#1f77b4", "dls": "#ff7f0e", "qp": "#2ca02c"}
    robots = [r for r in robots if any(x["robot"] == r for x in recs)]

    def get(robot, method, key, success_only=False):
        return [v for r in recs if r["robot"] == robot and r["ik_method"] == method
                and (not success_only or r.get("success"))
                for v in [_num(r.get(key))] if v is not None]

    def row(robot, method):
        for r in rows:
            if r["robot"] == robot and r["ik_method"] == method:
                return r
        return None

    def bar_success(ax, robot_list):
        width = 0.8 / len(methods)
        for j, m in enumerate(methods):
            xs, ys, err = [], [], [[], []]
            for i, robot in enumerate(robot_list):
                r = row(robot, m)
                if r is None:
                    continue
                xs.append(i + (j - (len(methods) - 1) / 2) * width)
                ys.append(100 * r["success_rate"])
                err[0].append(100 * (r["success_rate"] - r["success_ci95_low"]))
                err[1].append(100 * (r["success_ci95_high"] - r["success_rate"]))
            if xs:
                ax.bar(xs, ys, width, yerr=err, capsize=4, color=colors[m],
                              label=METHOD_LABELS[m])
                for bx, by in zip(xs, ys):
                    ax.text(bx, min(by + 2, 103), f"{by:.0f}%", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(robot_list)))
        ax.set_xticklabels([ROBOT_LABELS[r] for r in robot_list])
        ax.set_ylim(0, 110)
        ax.set_ylabel("Success rate (%)  [95% Wilson CI]")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    def box(ax, robot, key, ylabel, success_only=False, scale=1.0):
        data, labels, cols = [], [], []
        for m in methods:
            v = [x * scale for x in get(robot, m, key, success_only)]
            data.append(v if v else [np.nan])
            labels.append(f"{METHOD_LABELS[m]}\n(n={len(v)})")
            cols.append(colors[m])
        bp = ax.boxplot(data, patch_artist=True, showmeans=True)
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.45)
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)

    # ---- joint comparison figures ----
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_success(ax, robots)
    ax.set_title("Pick-and-place success rate by IK method (same seeds per robot)")
    fig.tight_layout()
    fig.savefig(pdir / "success_rate_joint.png", dpi=160)
    plt.close(fig)

    metrics = [
        ("ik_solve_time_total_s", "IK solve time per episode (s)", False, 1.0),
        ("ik_solve_time_mean_ms", "Mean IK solve time per call (ms)", False, 1.0),
        ("episode_wall_time_s", "Episode wall-clock time (s)", False, 1.0),
        ("sim_time_s", "Simulated episode time (s)", False, 1.0),
        ("min_clearance_m", "Min robot-obstacle clearance (mm)", False, 1000.0),
        ("ik_solver_calls", "IK solver calls per episode", False, 1.0),
    ]
    for key, label, succ, scale in metrics:
        fig, axes = plt.subplots(1, len(robots), figsize=(5.5 * len(robots), 4.5), squeeze=False)
        for ax, robot in zip(axes[0], robots):
            box(ax, robot, key, label, succ, scale)
            ax.set_title(ROBOT_LABELS[robot])
        fig.suptitle(label + " - all episodes")
        fig.tight_layout()
        fig.savefig(pdir / f"{key}_joint.png", dpi=160)
        plt.close(fig)

    # ---- failure categories (stacked) ----
    cats = sorted({r.get("failure_category") or "unknown" for r in recs if not r.get("success")})
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(robots) * len(methods)), 5.5))
    bottoms = None
    combos = [(rb, m) for rb in robots for m in methods]
    x = np.arange(len(combos))
    bottoms = np.zeros(len(combos))
    cmap = plt.get_cmap("tab20")
    for ci, cat in enumerate(cats):
        vals = np.array([sum(1 for r in recs if r["robot"] == rb and r["ik_method"] == m
                             and not r.get("success") and (r.get("failure_category") or "unknown") == cat)
                         for rb, m in combos], dtype=float)
        ax.bar(x, vals, bottom=bottoms, color=cmap(ci % 20), label=cat)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ROBOT_LABELS[rb]}\n{METHOD_LABELS[m]}" for rb, m in combos])
    ax.set_ylabel("Failed episodes")
    ax.set_title("Failure categories per robot and IK method")
    if cats:
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    else:
        ax.text(0.5, 0.5, "No failures recorded", transform=ax.transAxes, ha="center")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(pdir / "failure_categories_joint.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---- constraint violations ----
    fig, ax = plt.subplots(figsize=(9, 4.8))
    keys = [("joint_pos_limit_violation_steps", "joint position limit"),
            ("commanded_speed_violation_steps", "commanded speed > limit"),
            ("measured_speed_violation_steps", "measured speed > limit")]
    w = 0.8 / len(keys)
    for j, (k, lab) in enumerate(keys):
        vals = [sum(1 for r in recs if r["robot"] == rb and r["ik_method"] == m and (_num(r.get(k)) or 0) > 0)
                for rb, m in combos]
        ax.bar(x + (j - 1) * w, vals, w, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ROBOT_LABELS[rb]}\n{METHOD_LABELS[m]}" for rb, m in combos])
    ax.set_ylabel("Episodes with >= 1 violation")
    ax.set_title("Joint-limit / velocity-constraint violations")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(pdir / "constraint_violations_joint.png", dpi=160)
    plt.close(fig)

    # ---- per-robot dashboards (separate) ----
    for robot in robots:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        bar_success(axes[0, 0], [robot])
        axes[0, 0].set_title("Success rate")
        box(axes[0, 1], robot, "ik_solve_time_total_s", "IK solve time / episode (s)")
        axes[0, 1].set_title("IK (planning) time per episode")
        box(axes[0, 2], robot, "episode_wall_time_s", "Episode wall time (s)")
        axes[0, 2].set_title("Total episode execution time")
        box(axes[1, 0], robot, "min_clearance_m", "Min clearance (mm)", scale=1000.0)
        axes[1, 0].set_title("Minimum obstacle clearance")
        box(axes[1, 1], robot, "ik_solve_time_mean_ms", "IK solve per call (ms)")
        axes[1, 1].set_title("IK solve time per call")
        ax = axes[1, 2]
        seeds = sorted({int(r["seed"]) for r in recs if r["robot"] == robot})
        mat = np.full((len(methods), len(seeds)), np.nan)
        for i, m in enumerate(methods):
            for j, s in enumerate(seeds):
                for r in recs:
                    if r["robot"] == robot and r["ik_method"] == m and int(r["seed"]) == s:
                        mat[i, j] = 1.0 if r.get("success") else 0.0
        ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([METHOD_LABELS[m] for m in methods])
        ax.set_xlabel("Episode (same seed index for all methods)")
        ax.set_title("Paired outcomes (green = success)")
        fig.suptitle(f"{ROBOT_LABELS[robot]}: IK method comparison")
        fig.tight_layout()
        fig.savefig(pdir / f"{robot}_comparison.png", dpi=160)
        plt.close(fig)

    # ---- workspace / layout plots (table justification) ----
    tables = cfg.get("tables", {})
    for robot in robots:
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        t = tables.get(robot)
        if t:
            ax.add_patch(Rectangle((t["x_min"], t["y_min"]), t["x_max"] - t["x_min"],
                                   t["y_max"] - t["y_min"], fill=False, lw=2, ec="saddlebrown",
                                   label=f"table (top z = {t['top_z']:.3f} m)"))
        seeds = sorted({int(r["seed"]) for r in recs if r["robot"] == robot})
        for s in seeds:
            rs = [r for r in recs if r["robot"] == robot and int(r["seed"]) == s]
            nsucc = sum(bool(r.get("success")) for r in rs)
            c = "tab:green" if nsucc == len(rs) else ("tab:red" if nsucc == 0 else "tab:orange")
            r0 = rs[0]
            cx, cy = _num(r0.get("init_cube_x")), _num(r0.get("init_cube_y"))
            bx, by = _num(r0.get("init_container_x")), _num(r0.get("init_container_y"))
            if None in (cx, cy, bx, by):
                continue
            ax.plot(cx, cy, "s", color=c, ms=7)
            ax.plot(bx, by, "o", mfc="none", mec=c, ms=10)
            ax.plot([cx, bx], [cy, by], "-", color=c, lw=0.6, alpha=0.5)
            ax.annotate(f"{nsucc}/{len(rs)}", (cx, cy), fontsize=6, xytext=(3, 3),
                        textcoords="offset points")
        ax.plot(0, 0, "*", ms=15, color="tab:blue", label="robot base (world origin)")
        ax.plot([], [], "s", color="gray", label="cube start")
        ax.plot([], [], "o", mfc="none", mec="gray", label="container start")
        ax.plot([], [], "-", color="tab:green", label="all methods succeeded")
        ax.plot([], [], "-", color="tab:orange", label="some methods failed")
        ax.plot([], [], "-", color="tab:red", label="all methods failed")
        ax.set_aspect("equal")
        ax.set_xlabel("World X (m)")
        ax.set_ylabel("World Y (m)")
        ax.set_title(f"{ROBOT_LABELS[robot]}: episode layouts on the table\n"
                     "(label = successful IK methods for that seed)")
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(pdir / f"{robot}_layouts_workspace.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    # ---- failure-state schematics (always available) ----
    fdir = run_dir / "failure_states"
    fdir.mkdir(exist_ok=True)
    for r in recs:
        if r.get("success"):
            continue
        out = fdir / f"{r['robot']}_{r['ik_method']}_seed{r['seed']}.png"
        if out.is_file():
            continue
        fig, ax = plt.subplots(figsize=(6, 6))
        t = tables.get(r["robot"])
        if t:
            ax.add_patch(Rectangle((t["x_min"], t["y_min"]), t["x_max"] - t["x_min"],
                                   t["y_max"] - t["y_min"], fill=False, lw=2, ec="saddlebrown"))
        pts = [("init_cube", "s", "tab:red", "cube start"),
               ("init_container", "o", "tab:gray", "container start"),
               ("final_cube", "x", "k", "cube at end")]
        for pref, mk, c, lab in pts:
            px, py = _num(r.get(pref + "_x")), _num(r.get(pref + "_y"))
            if px is not None and py is not None:
                ax.plot(px, py, mk, color=c, ms=10, mfc="none" if mk == "o" else c, label=lab)
        try:
            tcp = json.loads(r.get("state_at_end_tcp"))
            ax.plot(tcp[0], tcp[1], "^", color="tab:blue", ms=10,
                    label=f"TCP at end (z={tcp[2]:.3f})")
        except Exception:
            pass
        ax.plot(0, 0, "*", ms=14, color="tab:blue")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        msg = str(r.get("failure_message", "")).split("\n")[0][:110]
        ax.set_title(f"{ROBOT_LABELS[r['robot']]} / {METHOD_LABELS[r['ik_method']]} / seed "
                     f"{r['seed']}\n{r.get('failure_category')} @ {r.get('failure_stage')}\n{msg}",
                     fontsize=7)
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)


def write_report(run_dir, recs, rows, fail_rows, pair_rows, robots, methods, cfg):
    L = []
    L.append(f"# Pick-and-place IK comparison: {run_dir.name}\n")
    L.append(f"Generated {now_stamp()}. Episodes recorded: {len(recs)}. Seeds per robot: "
             f"{len(cfg.get('seeds', []))} (identical across IK methods; "
             f"{'same' if cfg.get('same_seeds_for_both_robots', True) else 'different'} for both robots).\n")
    L.append("Success means the task script's own `run()` finished without raising: placement "
             "verified inside the container (position, resting height and wall-parallel yaw) and "
             "return to neutral completed.\n")
    L.append("IK (planning) time is the wall-clock time spent inside the solver's "
             "`compute_joint_velocities` only; episode time is the full wall-clock execution of "
             "`run()` (simulation, control and IK).\n")
    L.append("\n## Summary\n")
    L.append("| Robot | IK | n | Success | 95% CI | IK time/episode (s) | IK/call (ms) | "
             "Episode time (s) | Min clearance (mm) | Pos-limit viol. eps | Speed viol. eps |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")

    def f(v, s=1.0, d=3):
        return f"{v * s:.{d}f}" if isinstance(v, (int, float)) else str(v)
    for r in rows:
        L.append(f"| {ROBOT_LABELS[r['robot']]} | {METHOD_LABELS[r['ik_method']]} | {r['episodes']} | "
                 f"{r['successes']}/{r['episodes']} ({100 * r['success_rate']:.0f}%) | "
                 f"{100 * r['success_ci95_low']:.0f}-{100 * r['success_ci95_high']:.0f}% | "
                 f"{f(r['ik_time_per_episode_mean_s'])} | {f(r['ik_call_mean_ms'])} | "
                 f"{f(r['episode_wall_time_mean_s'], d=1)} | {f(r['min_clearance_min_m'], 1000, 1)} | "
                 f"{r['episodes_with_joint_pos_violation']} | "
                 f"{r['episodes_with_measured_speed_violation']} |")
    L.append("\nIK iterations: unavailable - all three solvers are differential IK methods that "
             "perform exactly one solve per control step and do not expose internal solver "
             "iterations; `ik_solver_calls` is reported instead.\n")
    L.append("\n## Failure analysis\n")
    if fail_rows:
        L.append("| Robot | IK | Category | Count | Stages | Example seeds | Mitigation |")
        L.append("|---|---|---|---|---|---|---|")
        for fr in fail_rows:
            L.append(f"| {ROBOT_LABELS[fr['robot']]} | {METHOD_LABELS[fr['ik_method']]} | "
                     f"{fr['category']} | {fr['count']} | {fr['stages']} | {fr['example_seeds']} | "
                     f"{fr['mitigation']} |")
    else:
        L.append("No failures were recorded.\n")
    L.append("\n## Paired comparison (same seeds, exact McNemar test)\n")
    if pair_rows:
        L.append("| Robot | A | B | Seeds | A only | B only | Both | Neither | p |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for p in pair_rows:
            L.append(f"| {ROBOT_LABELS[p['robot']]} | {METHOD_LABELS[p['method_a']]} | "
                     f"{METHOD_LABELS[p['method_b']]} | {p['paired_seeds']} | {p['a_only_success']} | "
                     f"{p['b_only_success']} | {p['both_success']} | {p['both_fail']} | "
                     f"{p['mcnemar_exact_p']:.3f} |")
    mism = [r for r in recs if r.get("layout_matches_reference") is False]
    L.append(f"\nLayout consistency: {len(mism)} episode(s) had an initial layout different from "
             "the reference for their seed" + (" (see episodes.csv)." if mism else "."))
    L.append("\n## Workspace / table\n")
    for robot, t in cfg.get("tables", {}).items():
        L.append(f"- {ROBOT_LABELS.get(robot, robot)}: tabletop X {t['x_min']:.3f}..{t['x_max']:.3f} m, "
                 f"Y {t['y_min']:.3f}..{t['y_max']:.3f} m, top z = {t['top_z']:.3f} m "
                 f"(see plots/{robot}_layouts_workspace.png).")
    L.append("\n## Files\n")
    L.append("- episodes.csv / episodes.jsonl, summary.csv, failure_summary.csv, pairwise_tests.csv, "
             "layouts.csv, config.json\n- plots/, failure_states/, screenshots/, logs/")
    (run_dir / "report.md").write_text("\n".join(L) + "\n")


# ======================================================================================
# Main
# ======================================================================================
def versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform(),
           "hostname": socket.gethostname()}
    for name in ("mujoco", "numpy", "mink", "qpsolvers", "daqp", "matplotlib"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "installed (version unknown)")
        except Exception as exc:
            out[name] = f"{UNAVAILABLE}: {type(exc).__name__}"
    try:
        out["git_commit"] = subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"],
                                           capture_output=True, text=True, timeout=10).stdout.strip() or UNAVAILABLE
        dirty = subprocess.run(["git", "-C", str(REPO_DIR), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        out["git_dirty"] = bool(dirty)
    except Exception:
        out["git_commit"] = UNAVAILABLE
    return out


def make_seeds(base_seed, n):
    seeds = [int(s) for s in np.random.SeedSequence(base_seed).generate_state(n, dtype=np.uint32)]
    seeds = [s % (2 ** 31 - 1) for s in seeds]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("Seed generation produced duplicates; choose another --base-seed")
    return seeds


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=25, help="episodes per robot x IK method (>= 25 recommended)")
    ap.add_argument("--base-seed", type=int, default=639, help="master seed for the episode seed list")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="explicit episode seeds (overrides --episodes/--base-seed)")
    ap.add_argument("--separate-robot-seeds", action="store_true",
                    help="use a different seed list for each robot (default: same seeds for both)")
    ap.add_argument("--robots", nargs="+", choices=ROBOTS, default=list(ROBOTS))
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--heal-script", default=None, help="path to the HEAL task script")
    ap.add_argument("--franka-script", default=None, help="path to the Franka task script")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel episodes (1 gives the cleanest timing comparison)")
    ap.add_argument("--episode-timeout", type=float, default=1200.0, help="wall-clock limit per episode (s)")
    ap.add_argument("--clearance-every", type=int, default=2, help="sample clearance every N control steps")
    ap.add_argument("--no-screenshots", action="store_true", help="disable offscreen rendering")
    ap.add_argument("--max-failure-screenshots", type=int, default=3, help="per robot x method")
    ap.add_argument("--mujoco-gl", default=None, help="MUJOCO_GL backend for screenshots (egl, osmesa, glfw)")
    ap.add_argument("--output-root", default=str(RESULTS_ROOT))
    ap.add_argument("--resume", default=None, help="existing run directory to continue")
    ap.add_argument("--retry-timeouts", action="store_true",
                    help="with --resume: re-run episodes recorded as timeout/crashed")
    ap.add_argument("--summarize-only", default=None, help="existing run directory to re-summarize")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--allow-failed-validation", action="store_true",
                    help="run the robots that validated even if another robot failed validation")
    args = ap.parse_args()
    if args.episodes < 1:
        ap.error("--episodes must be >= 1")
    if args.workers < 1:
        ap.error("--workers must be >= 1")
    return args


def main():
    args = parse_args()
    if args.summarize_only:
        run_dir = Path(args.summarize_only).resolve()
        cfg = json.loads((run_dir / "config.json").read_text())
        summarize(run_dir, cfg["robots"], cfg["methods"])
        return 0

    # ---------------- run directory and configuration ----------------
    if args.resume:
        run_dir = Path(args.resume).resolve()
        cfg = json.loads((run_dir / "config.json").read_text())
        robots, methods = cfg["robots"], cfg["methods"]
        seeds_by_robot = {r: [int(s) for s in cfg["seeds_by_robot"][r]] for r in robots}
        scripts = {r: Path(cfg["task_scripts"][r]["path"]) for r in robots}
        for r in robots:
            if sha256(scripts[r]) != cfg["task_scripts"][r]["sha256"]:
                print(f"WARNING: {scripts[r].name} changed since the run started.", flush=True)
        ref_layouts = cfg["reference_layouts"]
        # keep the original measurement settings; workers/timeout may be changed on resume
        for k in ("clearance_every", "no_screenshots", "max_failure_screenshots", "mujoco_gl"):
            if k in cfg.get("args", {}):
                setattr(args, k, cfg["args"][k])
    else:
        robots, methods = list(args.robots), list(args.methods)
        if args.seeds:
            base = [int(s) for s in args.seeds]
            if len(set(base)) != len(base):
                raise SystemExit("--seeds contains duplicates")
            seeds_by_robot = {r: base for r in robots}
        elif args.separate_robot_seeds:
            seeds_by_robot = {r: make_seeds(args.base_seed + 1000 * i, args.episodes)
                              for i, r in enumerate(robots)}
        else:
            base = make_seeds(args.base_seed, args.episodes)
            seeds_by_robot = {r: base for r in robots}
        scripts = {}
        for r in robots:
            scripts[r] = resolve_task_script(r, getattr(args, f"{r}_script"))
        print("Task scripts:", {r: str(p) for r, p in scripts.items()}, flush=True)

        # ---------------- validation ----------------
        print("Validating imports, models, IK solvers and seeded layouts ...", flush=True)
        val = {}
        for r in robots:
            val.update(validate([r], {r: scripts[r]}, seeds_by_robot[r]))
        bad = [r for r in robots if not val[r].get("ok")]
        for r in robots:
            v = val[r]
            if v.get("ok"):
                hashes = [v["layouts"][str(s)]["hash"] if str(s) in v["layouts"] else v["layouts"][s]["hash"]
                          for s in seeds_by_robot[r]]
                uniq = len(set(hashes))
                print(f"  {ROBOT_LABELS[r]}: OK | joints {v['arm_joints']} | site {v['ee_site']} | "
                      f"solvers {v['solver_types']} | {uniq}/{len(hashes)} distinct layouts | table "
                      f"{v['table']}", flush=True)
                if uniq != len(hashes):
                    print(f"  WARNING: {len(hashes) - uniq} seed(s) gave duplicate layouts for {r}.")
            else:
                print(f"  {ROBOT_LABELS[r]}: VALIDATION FAILED\n{v.get('error')}", flush=True)
        if bad and not args.allow_failed_validation:
            print("Aborting: fix the validation errors above (or pass --allow-failed-validation "
                  "to run only the robots that validated).", file=sys.stderr)
            return 3
        robots = [r for r in robots if r not in bad]
        if not robots:
            print("No robot passed validation.", file=sys.stderr)
            return 3
        if args.validate_only:
            print("Validation complete (--validate-only).")
            return 0

        root = Path(args.output_root)
        root.mkdir(parents=True, exist_ok=True)
        run_dir = root / f"run_{now_stamp()}"
        k = 1
        while run_dir.exists():
            run_dir = root / f"run_{now_stamp()}_{k}"
            k += 1
        run_dir.mkdir(parents=True)
        for sub in ("logs", "plots", "screenshots", "failure_states"):
            (run_dir / sub).mkdir()
        ref_layouts = {r: {str(s): val[r]["layouts"][str(s)] if str(s) in val[r]["layouts"]
                           else val[r]["layouts"][s] for s in seeds_by_robot[r]} for r in robots}
        cfg = dict(
            created=now_stamp(), run_dir=str(run_dir), robots=robots, methods=methods,
            episodes_per_combination=len(next(iter(seeds_by_robot.values()))),
            base_seed=args.base_seed, seeds=seeds_by_robot[robots[0]],
            seeds_by_robot=seeds_by_robot,
            same_seeds_for_both_robots=len({tuple(v) for v in seeds_by_robot.values()}) == 1,
            seed_policy=("episode seed -> task build_world(randomize=True, seed, randomize_yaw=True), "
                         "which seeds numpy.random.default_rng(seed); Python random and the legacy "
                         "numpy RNG are also seeded with the episode seed in each episode process. "
                         "The IK method is set only after the layout is built."),
            task_scripts={r: dict(path=str(scripts[r]), sha256=val[r]["sha256"],
                                  scene_module=val[r]["scene_module"]) for r in robots},
            solver_types={r: val[r]["solver_types"] for r in robots},
            robot_models={r: dict(arm_joints=val[r]["arm_joints"], ee_site=val[r]["ee_site"],
                                  nu=val[r]["nu"], nq=val[r]["nq"], nv=val[r]["nv"]) for r in robots},
            task_parameters={r: val[r]["task_parameters"] for r in robots},
            tables={r: val[r]["table"] for r in robots},
            reference_layouts=ref_layouts,
            metrics=dict(
                ik_time="wall time inside the IK solver's compute_joint_velocities() only",
                episode_time="wall time of PickPlace.run() (excludes scene build/TCP measurement)",
                setup_time="scene build + PickPlace construction",
                clearance=f"robot collision geoms vs table/container/floor, every "
                          f"{args.clearance_every} control step(s), capped at {CLEARANCE_CAP} m",
                joint_position_limits=f"XML jnt_range, tolerance {POS_LIMIT_TOL}",
                joint_speed_limit=f"task MAX_JOINT_SPEED; measured tolerance {VEL_LIMIT_TOL:.0%}",
                ik_iterations=UNAVAILABLE + " (one solve per control step)",
                success="task run() completed: verify_placement() true and return to neutral"),
            args=vars(args), versions=versions(),
        )
        (run_dir / "config.json").write_text(json.dumps(jsonable(cfg), indent=2))
        with open(run_dir / "layouts.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["robot", "seed", "cube_x", "cube_y", "cube_z", "cube_yaw_deg",
                        "container_x", "container_y", "container_z", "container_yaw_deg", "hash"])
            for r in robots:
                for s in seeds_by_robot[r]:
                    L = ref_layouts[r][str(s)]
                    w.writerow([r, s, L["cube"]["x"], L["cube"]["y"], L["cube"]["z"],
                                L["cube"]["yaw_deg"], L["container"]["x"], L["container"]["y"],
                                L["container"]["z"], L["container"]["yaw_deg"], L["hash"]])

    print(f"Run directory: {run_dir}", flush=True)
    jobs = []
    for r in robots:
        for idx, s in enumerate(seeds_by_robot[r]):
            for m in methods:   # interleave methods per seed so partial runs stay comparable
                jobs.append((r, m, int(s), idx))
    batch = Batch(run_dir, args, ref_layouts, scripts)
    exit_code = 0
    try:
        batch.run(jobs)
    except KeyboardInterrupt:
        exit_code = 130
    finally:
        try:
            summarize(run_dir, robots, methods)
        except Exception as exc:
            print(f"Summary generation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
    recs = load_records(run_dir)
    print(f"\nDone: {sum(bool(r.get('success')) for r in recs)}/{len(recs)} successful episodes. "
          f"Results in {run_dir}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
