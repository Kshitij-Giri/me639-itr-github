#!/usr/bin/env python3
"""Independent constrained QP differential IK for MuJoCo.

This module is self-contained: it does not import or depend on the DLS IK module.
Pass a compiled MuJoCo model/data pair, controlled scalar joint names, and the
end-effector site. Returned velocities follow the supplied joint order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mujoco
import numpy as np
import qpsolvers

DEFAULT_EE_SITE = "ee_site"

class IKFailure(RuntimeError):
    """Base class for QP solver failures."""

class IKNumericalError(IKFailure):
    """QP pose calculations or solution produced invalid numerical values."""

@dataclass(frozen=True)
class IKError:
    position_m: float
    orientation_rad: float

@dataclass(frozen=True)
class IKDiagnostics:
    position_error_m: float
    orientation_error_rad: float
    sigma_min: float
    condition_number: float
    speed_scale: float
    near_singular: bool
    active_limits: tuple = ()

def downward_rotation(yaw_degrees: float = 0.0) -> np.ndarray:
    """Rotation with tool-local +Z pointing world -Z, yawed about world Z."""
    yaw = np.deg2rad(float(yaw_degrees))
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                   [np.sin(yaw), np.cos(yaw), 0.0],
                   [0.0, 0.0, 1.0]])
    return rz @ np.diag([1.0, -1.0, -1.0])


def rotation_error_vector(rot_target: np.ndarray, rot_current: np.ndarray) -> np.ndarray:
    """World-frame rotation vector w such that exp(w) @ R_current = R_target."""
    q_d, q_c, q_c_inv, q_e = (np.zeros(4) for _ in range(4))
    mujoco.mju_mat2Quat(q_d, np.ascontiguousarray(rot_target, dtype=float).ravel())
    mujoco.mju_mat2Quat(q_c, np.ascontiguousarray(rot_current, dtype=float).ravel())
    mujoco.mju_negQuat(q_c_inv, q_c)
    mujoco.mju_mulQuat(q_e, q_d, q_c_inv)
    if q_e[0] < 0.0:
        q_e = -q_e
    vec_norm = float(np.linalg.norm(q_e[1:]))
    if vec_norm < 1e-12:
        return 2.0 * q_e[1:]
    return (2.0 * np.arctan2(vec_norm, q_e[0])) * q_e[1:] / vec_norm


class TaskKinematics:
    """Pose error and site Jacobian with respect to caller-selected scalar joints."""

    def __init__(self, model: mujoco.MjModel, joint_names: Sequence[str],
                 site_name: str = DEFAULT_EE_SITE, position_cost: float = 1.0,
                 orientation_cost: float = 1.0):
        self.model = model
        self.joint_names = tuple(joint_names)
        self.n_joints = len(self.joint_names)
        if not self.joint_names:
            raise ValueError("joint_names must contain at least one joint name")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must not contain duplicates")
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"MuJoCo site {site_name!r} was not found in the model")
        ids = []
        supported = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        for name in self.joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint {name!r} was not found in the model")
            if model.jnt_type[jid] not in supported:
                raise ValueError(f"Joint {name!r} must be a hinge or slide joint (one DoF)")
            ids.append(jid)
        self.joint_ids = np.asarray(ids, dtype=int)
        self.dof_idx = np.array([model.jnt_dofadr[j] for j in ids], dtype=int)
        self.qpos_idx = np.array([model.jnt_qposadr[j] for j in ids], dtype=int)
        inf = np.inf
        self.q_min = np.array([model.jnt_range[j, 0] if model.jnt_limited[j] else -inf for j in ids])
        self.q_max = np.array([model.jnt_range[j, 1] if model.jnt_limited[j] else inf for j in ids])
        if position_cost < 0 or orientation_cost < 0:
            raise ValueError("task costs must be non-negative")
        self.weights = np.array([position_cost] * 3 + [orientation_cost] * 3, dtype=float)
        self._scratch = mujoco.MjData(model)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    @staticmethod
    def check_target(position: Sequence[float], rotation: np.ndarray):
        pos, rot = np.asarray(position, dtype=float), np.asarray(rotation, dtype=float)
        if pos.shape != (3,):
            raise ValueError("target_position must contain exactly 3 values (x, y, z)")
        if rot.shape != (3, 3):
            raise ValueError("target_rotation must be a 3x3 rotation matrix")
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(rot))):
            raise ValueError("Target pose must contain only finite numbers")
        if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rot), 1.0, atol=1e-5):
            raise ValueError("target_rotation must be a valid rotation matrix")
        return pos, rot

    def pose(self, qpos: np.ndarray):
        """Return site position and rotation for a full generalized-position vector."""
        s = self._scratch
        s.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, s)
        mujoco.mj_comPos(self.model, s)
        return s.site_xpos[self.site_id].copy(), s.site_xmat[self.site_id].reshape(3, 3).copy()

    def error_and_jacobian(self, qpos, target_pos, target_rot):
        pos, rot = self.pose(qpos)
        err = np.concatenate([target_pos - pos, rotation_error_vector(target_rot, rot)])
        mujoco.mj_jacSite(self.model, self._scratch, self._jacp, self._jacr, self.site_id)
        jac = np.vstack([self._jacp[:, self.dof_idx], self._jacr[:, self.dof_idx]])
        return err, jac


def _speed_limits(max_joint_velocity, n):
    if max_joint_velocity is None:
        return None
    limits = np.asarray(max_joint_velocity, dtype=float)
    if limits.ndim == 0:
        limits = np.full(n, float(limits))
    if limits.shape != (n,) or np.any(limits <= 0):
        raise ValueError(f"max_joint_velocity must be a positive scalar or {n} positive values")
    return limits


class QPInfeasibleError(IKFailure):
    """The constrained QP has no feasible point or the solver returned no solution."""


class QPIK:
    SINGULAR_SIGMA = 1e-2
    """Constrained differential IK: DLS objective + joint-position and joint-velocity limits.

    Args: ``solver`` (qpsolvers backend, default "daqp"), ``dt``, ``damping`` (lambda), ``gain``,
    ``max_joint_velocity`` (rad/s, scalar or one value per selected joint; None = no velocity bound), ``limit_margin`` (rad kept
    between each selected joint and its position limits).
    """

    def __init__(self, model, data, joint_names: Sequence[str], site_name: str = DEFAULT_EE_SITE, solver: str = "daqp",
                 dt: float = 0.01, damping: float = 1e-3, gain: float = 0.2,
                 max_joint_velocity: float | Sequence[float] | None = 1.0,
                 position_cost: float = 1.0, orientation_cost: float = 1.0,
                 limit_margin: float = 0.0):
        if dt <= 0:
            raise ValueError("dt must be positive")
        if damping <= 0:
            raise ValueError("damping must be positive (keeps the Hessian positive definite)")
        if solver not in qpsolvers.available_solvers:
            raise RuntimeError(f"QP solver {solver!r} not available; installed: {qpsolvers.available_solvers}. "
                               "Install with: pip install qpsolvers daqp")
        self.model, self.data, self.solver = model, data, solver
        self.dt, self.damping, self.gain = float(dt), float(damping), float(gain)
        self.limit_margin = float(limit_margin)
        self.max_joint_velocity = _speed_limits(max_joint_velocity, len(joint_names))
        self.kin = TaskKinematics(model, joint_names, site_name, position_cost, orientation_cost)
        self.site_id, self.dof_indices = self.kin.site_id, list(self.kin.dof_idx)
        self._target = None
        self.last = None

    def _bounds(self, q_arm: np.ndarray):
        """Per-joint velocity interval [lb, ub] combining position and velocity limits."""
        vmax = np.full(self.kin.n_joints, np.inf) if self.max_joint_velocity is None else self.max_joint_velocity
        lb = np.maximum(-vmax, (self.kin.q_min + self.limit_margin - q_arm) / self.dt)
        ub = np.minimum(vmax, (self.kin.q_max - self.limit_margin - q_arm) / self.dt)
        return lb, ub

    def compute_joint_velocities(self, target_position: Sequence[float], target_rotation: np.ndarray) -> np.ndarray:
        """Joint velocities (rad/s), ordered as supplied in joint_names."""
        pos_d, rot_d = self.kin.check_target(target_position, target_rotation)
        self._target = (pos_d, rot_d)
        err, jac = self.kin.error_and_jacobian(self.data.qpos, pos_d, rot_d)
        if not (np.all(np.isfinite(err)) and np.all(np.isfinite(jac))):
            raise IKNumericalError("non-finite pose error or Jacobian (check the simulation state)")
        a = self.kin.weights[:, None] * jac
        b = self.kin.weights * (self.gain / self.dt) * err
        hess = 2.0 * (a.T @ a + self.damping * np.eye(self.kin.n_joints))
        hess = 0.5 * (hess + hess.T)                      # exact symmetry for the solver
        lin = -2.0 * (a.T @ b)

        q_arm = self.data.qpos[self.kin.qpos_idx]
        lb, ub = self._bounds(q_arm)
        bad = np.flatnonzero(lb > ub + 1e-12)
        if bad.size:
            names = ", ".join(f"{self.kin.joint_names[i]} (q={q_arm[i]:+.4f}, range "
                              f"[{self.kin.q_min[i]:+.3f}, {self.kin.q_max[i]:+.3f}])" for i in bad)
            raise QPInfeasibleError(f"empty feasible set: {names} is outside its limits by more than "
                                    "max_joint_velocity * dt")
        finite_u, finite_l = np.isfinite(ub), np.isfinite(lb)
        g_mat = np.vstack([np.eye(self.kin.n_joints)[finite_u], -np.eye(self.kin.n_joints)[finite_l]])   # G qdot <= h
        h_vec = np.concatenate([ub[finite_u], -lb[finite_l]])

        result = qpsolvers.solve_problem(qpsolvers.Problem(hess, lin, g_mat, h_vec), solver=self.solver)
        if not result.found or result.x is None:
            raise QPInfeasibleError(f"{self.solver} returned no solution (status: {result.extra})")
        qdot = np.asarray(result.x, dtype=float)
        if not np.all(np.isfinite(qdot)):
            raise IKNumericalError("QP solver returned non-finite joint velocities")
        qdot = np.clip(qdot, lb, ub)                      # remove solver tolerance overshoot

        tol = 1e-6
        active = tuple([f"{self.kin.joint_names[i]}:+" for i in range(self.kin.n_joints) if qdot[i] >= ub[i] - tol] +
                       [f"{self.kin.joint_names[i]}:-" for i in range(self.kin.n_joints) if qdot[i] <= lb[i] + tol])
        sv = np.linalg.svd(a, compute_uv=False)
        self.last = IKDiagnostics(float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:])),
                                  float(sv[-1]), float(sv[0] / max(sv[-1], 1e-300)), 1.0,
                                  bool(sv[-1] < 1e-2), active)
        return qdot

    # ---------------------------------------------------------------- helpers (Mink-compatible)
    def current_error(self) -> IKError:
        if self._target is None:
            raise RuntimeError("No target set yet; call compute_joint_velocities first")
        err, _ = self.kin.error_and_jacobian(self.data.qpos, *self._target)
        return IKError(float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:])))

    def compute_full_velocity_vector(self, target_position, target_rotation) -> np.ndarray:
        full = np.zeros(self.model.nv)
        full[self.kin.dof_idx] = self.compute_joint_velocities(target_position, target_rotation)
        return full

    def joint_velocity_dict(self, joint_velocities: Sequence[float]) -> dict[str, float]:
        values = np.asarray(joint_velocities, dtype=float)
        if values.shape != (len(self.kin.joint_names),):
            raise ValueError(f"joint_velocities must contain {len(self.kin.joint_names)} values")
        return dict(zip(self.kin.joint_names, map(float, values)))




if __name__ == "__main__":
    print("Import QPIK and provide a MuJoCo model, MjData, joint_names, and site_name.")
