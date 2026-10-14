#!/usr/bin/env python3
"""Custom Damped-Least-Squares (DLS) differential IK for the HEAL arm (no Mink).

Same interface as utils/mink_ik_heal/mink_ik_heal.py: pass the compiled MuJoCo model/data and a
desired site pose, get back six joint velocities (rad/s) ordered joint_1 ... joint_6. This module does
not load the scene, step MuJoCo or write actuator commands.

Theory (established; Nakamura & Hanafusa 1986, Wampler 1986, Buss 2004 survey)
  Differential kinematics   xdot = J(q) qdot,   J = [jacp; jacr]  (6x6 for the HEAL arm)
  Regularised objective     f(qdot) = ||W (J qdot - xdot_d)||^2 + lambda ||qdot||^2
  Stationarity              2 A^T (A qdot - b) + 2 lambda qdot = 0,   A = W J, b = W xdot_d
  =>                        (A^T A + lambda I) qdot = A^T b           (SPD for lambda > 0)
  SVD view                  qdot = sum_i  sigma_i / (sigma_i^2 + lambda) (u_i . b) v_i  (bounded by
                            1/(2 sqrt(lambda)); pseudoinverse would use 1/sigma_i -> blows up).
Implementation choices (mine, not part of the textbook method)
  * xdot_d = (gain / dt) * e, e = [p_d - p ; rotvec(R_d R^T)] (world frame, matches mj_jacSite).
  * lambda is a fixed constant ("damping"); with gain/dt scaling it equals Mink's `damping`, so the
    unconstrained Mink, DLS and QP solutions coincide (the dt^2 factors cancel).
  * Linear solve via np.linalg.solve (never an explicit inverse); lstsq fallback on LinAlgError.
  * Joint-velocity limit by uniform scaling (keeps the Cartesian direction). DLS has no joint-position
    limits - that is the point of comparing it with the QP version.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

HEAL_JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")
DEFAULT_EE_SITE = "ft_site_eff"


class IKFailure(RuntimeError):
    """Base class: the solver could not produce a usable velocity."""


class IKNumericalError(IKFailure):
    """Linear algebra failed or produced non-finite numbers."""


@dataclass(frozen=True)
class IKError:
    """Current target error magnitudes, in metres and radians (same as MinkIKHeal)."""

    position_m: float
    orientation_rad: float


@dataclass(frozen=True)
class IKDiagnostics:
    """Per-call solver information (for logging / comparing methods)."""

    position_error_m: float
    orientation_error_rad: float
    sigma_min: float            # smallest singular value of the weighted Jacobian
    condition_number: float
    speed_scale: float          # 1.0 unless the joint-speed limit scaled the solution down
    near_singular: bool
    active_limits: tuple = ()   # QP only: bounds that are active, e.g. ("joint_2:+pos",)


def downward_rotation(yaw_degrees: float = 0.0) -> np.ndarray:
    """Tool-local +Z points world -Z, rotated by yaw about world Z (same as the Mink utility)."""
    yaw = np.deg2rad(float(yaw_degrees))
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    return rz @ np.diag([1.0, -1.0, -1.0])


def rotation_error_vector(rot_target: np.ndarray, rot_current: np.ndarray) -> np.ndarray:
    """World-frame rotation vector w with exp(w) @ R_current = R_target (shortest path, |w| <= pi).

    Computed through quaternions q_e = q_d * conj(q_c): q and -q are the same rotation, so w < 0 is
    flipped to keep the short way round; angle = 2 atan2(|v|, w) is accurate for all angles.
    """
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


class HealTaskKinematics:
    """Jacobian and pose error of one MuJoCo site w.r.t. the six HEAL joints.

    Works on a private scratch MjData, so the live simulation state is never touched.
    """

    def __init__(self, model: mujoco.MjModel, site_name: str = DEFAULT_EE_SITE,
                 position_cost: float = 1.0, orientation_cost: float = 1.0):
        self.model = model
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"MuJoCo site {site_name!r} was not found in the model")
        ids = []
        for name in HEAL_JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Required HEAL joint {name!r} was not found in the model")
            if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"Expected {name!r} to be a hinge joint")
            ids.append(jid)
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
        """Site position and rotation matrix for the full generalized position vector."""
        s = self._scratch
        s.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, s)
        mujoco.mj_comPos(self.model, s)  # required by mj_jacSite (cdof, subtree COM)
        return s.site_xpos[self.site_id].copy(), s.site_xmat[self.site_id].reshape(3, 3).copy()

    def error_and_jacobian(self, qpos, target_pos, target_rot):
        """Unweighted 6-vector error [p_d - p; rotvec] and 6x6 world-frame arm Jacobian."""
        pos, rot = self.pose(qpos)
        err = np.concatenate([target_pos - pos, rotation_error_vector(target_rot, rot)])
        mujoco.mj_jacSite(self.model, self._scratch, self._jacp, self._jacr, self.site_id)
        jac = np.vstack([self._jacp[:, self.dof_idx], self._jacr[:, self.dof_idx]])
        return err, jac


def _speed_limits(max_joint_velocity, n=6):
    if max_joint_velocity is None:
        return None
    limits = np.asarray(max_joint_velocity, dtype=float)
    if limits.ndim == 0:
        limits = np.full(n, float(limits))
    if limits.shape != (n,) or np.any(limits <= 0):
        raise ValueError("max_joint_velocity must be a positive scalar or six positive values")
    return limits


class DLSIKHeal:
    """Closed-form damped-least-squares IK returning six HEAL joint velocities.

    Args mirror MinkIKHeal: ``dt`` (control period, s), ``damping`` (lambda), ``gain`` (fraction of
    the pose error to remove per control period), ``max_joint_velocity`` (rad/s, scalar or six).
    """

    SINGULAR_SIGMA = 1e-2  # sigma_min below this is reported as near_singular

    def __init__(self, model, data, site_name: str = DEFAULT_EE_SITE, dt: float = 0.01,
                 damping: float = 1e-3, gain: float = 0.2,
                 max_joint_velocity: float | Sequence[float] | None = 1.0,
                 position_cost: float = 1.0, orientation_cost: float = 1.0):
        if dt <= 0:
            raise ValueError("dt must be positive")
        if damping <= 0:
            raise ValueError("damping must be positive for DLS (lambda > 0)")
        self.model, self.data = model, data
        self.dt, self.damping, self.gain = float(dt), float(damping), float(gain)
        self.max_joint_velocity = _speed_limits(max_joint_velocity)
        self.kin = HealTaskKinematics(model, site_name, position_cost, orientation_cost)
        self.site_id, self.dof_indices = self.kin.site_id, list(self.kin.dof_idx)
        self._target = None
        self.last = None  # IKDiagnostics of the latest call

    # ---------------------------------------------------------------- core
    def _solve(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """(A^T A + lambda I) qdot = A^T b by Cholesky-backed LU solve (no explicit inverse)."""
        normal = a.T @ a + self.damping * np.eye(a.shape[1])
        try:
            qdot = np.linalg.solve(normal, a.T @ b)
        except np.linalg.LinAlgError:
            # Equivalent augmented least squares  [A; sqrt(lambda) I] qdot = [b; 0]
            aug = np.vstack([a, np.sqrt(self.damping) * np.eye(a.shape[1])])
            qdot = np.linalg.lstsq(aug, np.concatenate([b, np.zeros(a.shape[1])]), rcond=None)[0]
        if not np.all(np.isfinite(qdot)):
            raise IKNumericalError("DLS produced non-finite joint velocities")
        return qdot

    def compute_joint_velocities(self, target_position: Sequence[float], target_rotation: np.ndarray) -> np.ndarray:
        """Six joint velocities (rad/s), ordered joint_1 ... joint_6, from the current data.qpos."""
        pos_d, rot_d = self.kin.check_target(target_position, target_rotation)
        self._target = (pos_d, rot_d)
        err, jac = self.kin.error_and_jacobian(self.data.qpos, pos_d, rot_d)
        if not (np.all(np.isfinite(err)) and np.all(np.isfinite(jac))):
            raise IKNumericalError("non-finite pose error or Jacobian (check the simulation state)")
        a = self.kin.weights[:, None] * jac
        b = self.kin.weights * (self.gain / self.dt) * err
        qdot = self._solve(a, b)

        scale = 1.0
        if self.max_joint_velocity is not None:
            ratio = float(np.max(np.abs(qdot) / self.max_joint_velocity))
            if ratio > 1.0:
                scale = 1.0 / ratio
                qdot = qdot * scale
        sv = np.linalg.svd(a, compute_uv=False)
        self.last = IKDiagnostics(float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:])),
                                  float(sv[-1]), float(sv[0] / max(sv[-1], 1e-300)), scale,
                                  bool(sv[-1] < self.SINGULAR_SIGMA))
        return qdot

    # ---------------------------------------------------------------- helpers (Mink-compatible)
    def current_error(self) -> IKError:
        """Pose error of the current data.qpos relative to the last target."""
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
        if values.shape != (6,):
            raise ValueError("joint_velocities must contain six values")
        return dict(zip(HEAL_JOINT_NAMES, map(float, values)))


# ======================================================================================
# Independent self-test on known poses (shared with the QP utility).
#   python utils/dls_ik_heal/dls_ik_heal.py
# ======================================================================================
def _load_model():
    lab = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(lab / "scripts"))
    import heal_pick_place_scene as scene  # noqa: WPS433
    model = scene.construct_scene()
    return model, mujoco.MjData(model)


def common_selftest(make_ik, label: str) -> bool:
    """Jacobian, error math, convergence on known (FK-generated) poses, unreachable-target bounds."""
    model, data = _load_model()
    mujoco.mj_forward(model, data)
    kin = HealTaskKinematics(model)
    ok = True
    rng = np.random.default_rng(0)

    # 1) Jacobian vs finite differences of the forward kinematics
    q0 = data.qpos.copy()
    q0[kin.qpos_idx] = rng.uniform(-1.0, 1.0, 6)
    _, jac = kin.error_and_jacobian(q0, np.zeros(3), np.eye(3))
    p0, r0 = kin.pose(q0)
    fd = np.zeros((6, 6))
    for k in range(6):
        q1 = q0.copy(); q1[kin.qpos_idx[k]] += 1e-6
        p1, r1 = kin.pose(q1)
        fd[:, k] = np.concatenate([(p1 - p0) / 1e-6, rotation_error_vector(r1, r0) / 1e-6])
    jac_err = float(np.max(np.abs(fd - jac)))
    print(f"[{label}] Jacobian vs finite difference, max |diff| = {jac_err:.2e}")
    ok &= jac_err < 1e-5

    # 2) rotation error: known rotations, large angle, sign (q vs -q) robustness
    worst = 0.0
    for angle, axis in [(0.3, [0, 0, 1]), (1.2, [1, 0, 0]), (3.0, [0, 1, 0]), (-2.0, [1, 1, 1])]:
        axis = np.asarray(axis, float) / np.linalg.norm(axis)
        w = angle * axis
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rot_d = (np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k) @ r0
        worst = max(worst, float(np.linalg.norm(rotation_error_vector(rot_d, r0) - w)))
    print(f"[{label}] rotation-error vs known rotations (0.3..3.0 rad), max |diff| = {worst:.2e}")
    ok &= worst < 1e-9

    # 3) kinematic closed-loop convergence to poses generated by FK of random reachable configs
    ik = make_ik(model, data)
    dt, converged, finals = ik.dt, 0, []
    n_goals = 20
    for g in range(n_goals):
        q_goal = data.qpos.copy(); q_goal[kin.qpos_idx] = rng.uniform(-1.2, 1.2, 6)
        p_d, r_d = kin.pose(q_goal)
        data.qpos[:] = 0.0
        for _ in range(1500):
            v = ik.compute_joint_velocities(p_d, r_d)
            data.qpos[kin.qpos_idx] += v * dt
        e = ik.current_error()
        finals.append((e.position_m, e.orientation_rad))
        converged += e.position_m < 1e-3 and e.orientation_rad < 1e-2
    print(f"[{label}] converged to FK-generated poses: {converged}/{n_goals} "
          f"(median final err {np.median([f[0] for f in finals]) * 1e3:.4f} mm, "
          f"{np.degrees(np.median([f[1] for f in finals])):.4f} deg)")
    ok &= converged >= int(0.8 * n_goals)  # local minima / limits can legitimately defeat a few

    # 4) unreachable target (2 m away): bounded, finite output, no divergence
    data.qpos[:] = 0.0
    vmax = 0.0
    for _ in range(300):
        v = ik.compute_joint_velocities([2.0, 0.0, 0.5], downward_rotation(0.0))
        assert np.all(np.isfinite(v))
        vmax = max(vmax, float(np.max(np.abs(v))))
        data.qpos[kin.qpos_idx] += v * dt
    print(f"[{label}] unreachable target: max |qdot| = {vmax:.3f} rad/s (limit 1.0), "
          f"final distance {ik.current_error().position_m:.3f} m, near_singular={ik.last.near_singular}")
    ok &= vmax <= 1.0 + 1e-9
    data.qpos[:] = 0.0
    return bool(ok)


if __name__ == "__main__":
    passed = common_selftest(lambda m, d: DLSIKHeal(m, d, dt=0.01, damping=1e-3, gain=0.3), "DLS")
    print("DLS SELF-TEST", "PASSED" if passed else "FAILED")
    sys.exit(0 if passed else 1)
