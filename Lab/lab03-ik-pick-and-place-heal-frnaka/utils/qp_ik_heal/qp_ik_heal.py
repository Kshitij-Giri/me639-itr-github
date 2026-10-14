#!/usr/bin/env python3
"""Custom constrained (QP) differential IK for the HEAL arm (no Mink, no existing IK solver).

Same interface as utils/mink_ik_heal/mink_ik_heal.py and utils/dls_ik_heal/dls_ik_heal.py.

Theory (established: constrained/QP differential IK, e.g. Kanoun et al. 2011; DAQP solver:
Arnstrom, Bemporad, Axehill 2022)
  Same regularised tracking objective as DLS, with the unknown being the joint velocity qdot:
      f(qdot) = ||W (J qdot - xdot_d)||^2 + lambda ||qdot||^2
              = qdot^T (A^T A + lambda I) qdot - 2 (A^T b)^T qdot + const,   A = W J, b = W xdot_d
  =>  min 1/2 qdot^T H qdot + c^T qdot,   H = 2 (A^T A + lambda I),   c = -2 A^T b.
  Position-increment form: with dq = qdot * dt,  ||W(J dq - dt xdot_d)||^2 + lambda ||dq||^2 = dt^2 f,
  so the minimiser is identical (same lambda); this module solves for qdot and never mixes the two.
  Inequalities  G qdot <= h  (G = [I; -I]) encode, over one control period dt,
      joint position   q_min <= q + qdot dt <= q_max  ->  qdot <= (q_max - q)/dt,  -qdot <= (q - q_min)/dt
      joint velocity   |qdot_i| <= v_max_i
  i.e. per joint  max(-v_max, (q_min - q)/dt) <= qdot <= min(v_max, (q_max - q)/dt).
  Without active constraints the minimiser equals the DLS solution exactly.
Implementation choices (mine)
  * Solver: DAQP (dual active-set, library-free, designed for small dense QPs) through `qpsolvers`.
  * Error/Jacobian definitions are shared with the DLS module (world frame, mj_jacSite), so the three
    methods use identical pose-error, damping, dt and gain conventions.
  * Failure policy: empty feasible set (a joint farther outside its limits than v_max*dt) or a solver
    that returns no solution raises QPInfeasibleError; non-finite data raises IKNumericalError. The
    caller decides what to do (the task script holds still for a few ticks, then fails the stage).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np

_LAB_DIR = Path(__file__).resolve().parents[2]
if str(_LAB_DIR) not in sys.path:
    sys.path.insert(0, str(_LAB_DIR))

import qpsolvers  # noqa: E402

from utils.dls_ik_heal.dls_ik_heal import (  # noqa: E402,F401
    DEFAULT_EE_SITE,
    HEAL_JOINT_NAMES,
    DLSIKHeal,
    HealTaskKinematics,
    IKDiagnostics,
    IKError,
    IKFailure,
    IKNumericalError,
    _speed_limits,
    common_selftest,
    downward_rotation,
)


class QPInfeasibleError(IKFailure):
    """The constrained QP has no feasible point or the QP solver returned no solution."""


class QPIKHeal:
    """Constrained differential IK: DLS objective + joint-position and joint-velocity limits.

    Args: ``solver`` (qpsolvers backend, default "daqp"), ``dt``, ``damping`` (lambda), ``gain``,
    ``max_joint_velocity`` (rad/s, scalar or six; None = no velocity bound), ``limit_margin`` (rad kept
    between the arm and its joint-position limits).
    """

    def __init__(self, model, data, site_name: str = DEFAULT_EE_SITE, solver: str = "daqp",
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
        self.max_joint_velocity = _speed_limits(max_joint_velocity)
        self.kin = HealTaskKinematics(model, site_name, position_cost, orientation_cost)
        self.site_id, self.dof_indices = self.kin.site_id, list(self.kin.dof_idx)
        self._target = None
        self.last = None

    def _bounds(self, q_arm: np.ndarray):
        """Per-joint velocity interval [lb, ub] combining position and velocity limits."""
        vmax = np.full(6, np.inf) if self.max_joint_velocity is None else self.max_joint_velocity
        lb = np.maximum(-vmax, (self.kin.q_min + self.limit_margin - q_arm) / self.dt)
        ub = np.minimum(vmax, (self.kin.q_max - self.limit_margin - q_arm) / self.dt)
        return lb, ub

    def compute_joint_velocities(self, target_position: Sequence[float], target_rotation: np.ndarray) -> np.ndarray:
        """Six joint velocities (rad/s), ordered joint_1 ... joint_6, from the current data.qpos."""
        pos_d, rot_d = self.kin.check_target(target_position, target_rotation)
        self._target = (pos_d, rot_d)
        err, jac = self.kin.error_and_jacobian(self.data.qpos, pos_d, rot_d)
        if not (np.all(np.isfinite(err)) and np.all(np.isfinite(jac))):
            raise IKNumericalError("non-finite pose error or Jacobian (check the simulation state)")
        a = self.kin.weights[:, None] * jac
        b = self.kin.weights * (self.gain / self.dt) * err
        hess = 2.0 * (a.T @ a + self.damping * np.eye(6))
        hess = 0.5 * (hess + hess.T)                      # exact symmetry for the solver
        lin = -2.0 * (a.T @ b)

        q_arm = self.data.qpos[self.kin.qpos_idx]
        lb, ub = self._bounds(q_arm)
        bad = np.flatnonzero(lb > ub + 1e-12)
        if bad.size:
            names = ", ".join(f"{HEAL_JOINT_NAMES[i]} (q={q_arm[i]:+.4f}, range "
                              f"[{self.kin.q_min[i]:+.3f}, {self.kin.q_max[i]:+.3f}])" for i in bad)
            raise QPInfeasibleError(f"empty feasible set: {names} is outside its limits by more than "
                                    "max_joint_velocity * dt")
        finite_u, finite_l = np.isfinite(ub), np.isfinite(lb)
        g_mat = np.vstack([np.eye(6)[finite_u], -np.eye(6)[finite_l]])   # G qdot <= h
        h_vec = np.concatenate([ub[finite_u], -lb[finite_l]])

        result = qpsolvers.solve_problem(qpsolvers.Problem(hess, lin, g_mat, h_vec), solver=self.solver)
        if not result.found or result.x is None:
            raise QPInfeasibleError(f"{self.solver} returned no solution (status: {result.extra})")
        qdot = np.asarray(result.x, dtype=float)
        if not np.all(np.isfinite(qdot)):
            raise IKNumericalError("QP solver returned non-finite joint velocities")
        qdot = np.clip(qdot, lb, ub)                      # remove solver tolerance overshoot

        tol = 1e-6
        active = tuple([f"{HEAL_JOINT_NAMES[i]}:+" for i in range(6) if qdot[i] >= ub[i] - tol] +
                       [f"{HEAL_JOINT_NAMES[i]}:-" for i in range(6) if qdot[i] <= lb[i] + tol])
        sv = np.linalg.svd(a, compute_uv=False)
        self.last = IKDiagnostics(float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:])),
                                  float(sv[-1]), float(sv[0] / max(sv[-1], 1e-300)), 1.0,
                                  bool(sv[-1] < DLSIKHeal.SINGULAR_SIGMA), active)
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
        if values.shape != (6,):
            raise ValueError("joint_velocities must contain six values")
        return dict(zip(HEAL_JOINT_NAMES, map(float, values)))


# ======================================================================================
# Independent self-test (common checks + QP-specific constraint checks).
#   python utils/qp_ik_heal/qp_ik_heal.py
# ======================================================================================
def _qp_specific_tests() -> bool:
    import mujoco
    from utils.dls_ik_heal.dls_ik_heal import _load_model
    model, data = _load_model()
    mujoco.mj_forward(model, data)
    qp = QPIKHeal(model, data, dt=0.01, damping=1e-3, gain=0.3, max_joint_velocity=1.0)
    dls = DLSIKHeal(model, data, dt=0.01, damping=1e-3, gain=0.3, max_joint_velocity=1.0)
    idx, ok = qp.kin.qpos_idx, True
    tgt_r = downward_rotation(20.0)
    tgt_p = np.array([0.30, 0.10, 0.40])

    def pose_of(q6):
        q = data.qpos.copy(); q[idx] = q6
        return qp.kin.pose(q)

    # a) no active constraint (small pose error, far from limits) -> QP == DLS to solver tolerance
    q_a = np.array([0.2, -0.3, 0.4, 0.1, -0.2, 0.3])
    tgt_p, tgt_r = pose_of(q_a + 0.02)
    data.qpos[idx] = q_a
    v_qp = qp.compute_joint_velocities(tgt_p, tgt_r)
    diff = float(np.max(np.abs(v_qp - dls.compute_joint_velocities(tgt_p, tgt_r))))
    print(f"[QP] unconstrained QP vs DLS, max |diff| = {diff:.2e} rad/s (active limits: {qp.last.active_limits})")
    ok &= diff < 1e-5 and not qp.last.active_limits

    # b) position limit: joint_1 is 0.005 rad below its upper limit and the target needs it to keep
    #    increasing; the QP must cap qdot1 at (q_max - q)/dt, DLS (unconstrained) does not.
    q_b = np.array([qp.kin.q_max[0] - 0.005, -0.3, 0.4, 0.1, -0.2, 0.3])
    q_goal = q_b.copy(); q_goal[0] += 0.05            # FK wraps past +pi: asks for +0.05 rad on joint_1
    tgt_p, tgt_r = pose_of(q_goal)
    data.qpos[idx] = q_b
    v_qp = qp.compute_joint_velocities(tgt_p, tgt_r)
    allowed = (qp.kin.q_max[0] - q_b[0]) / qp.dt
    active = qp.last.active_limits
    v_dls = dls.compute_joint_velocities(tgt_p, tgt_r)
    print(f"[QP] joint_1 0.005 rad below +limit: QP qdot1 = {v_qp[0]:+.4f} (cap {allowed:.4f}), "
          f"DLS qdot1 = {v_dls[0]:+.4f}; QP active = {active}")
    ok &= v_qp[0] <= allowed + 1e-9 and v_dls[0] > allowed and "joint_1:+" in active

    # c) velocity limit as a hard constraint
    data.qpos[idx] = 0.0
    qp_fast = QPIKHeal(model, data, dt=0.01, damping=1e-3, gain=1.0, max_joint_velocity=0.2)
    v = qp_fast.compute_joint_velocities([0.5, 0.3, 0.2], tgt_r)
    print(f"[QP] velocity limit 0.2 rad/s: max |qdot| = {np.max(np.abs(v)):.4f}")
    ok &= np.max(np.abs(v)) <= 0.2 + 1e-9

    # d) infeasible: joint_2 further outside its limit than v_max*dt  -> QPInfeasibleError
    data.qpos[idx] = [0, qp.kin.q_max[1] + 0.05, 0, 0, 0, 0]
    try:
        qp.compute_joint_velocities(tgt_p, tgt_r)
        print("[QP] infeasible case: NO exception raised  <-- FAIL")
        ok = False
    except QPInfeasibleError as exc:
        print(f"[QP] infeasible case raised QPInfeasibleError: {str(exc)[:90]}...")

    # e) bad inputs
    data.qpos[idx] = 0.0
    for bad in ([np.nan, 0, 0], [0, 0]):
        try:
            qp.compute_joint_velocities(bad, tgt_r); ok = False
        except ValueError:
            pass
    print("[QP] invalid targets rejected with ValueError")
    return bool(ok)


if __name__ == "__main__":
    passed = common_selftest(lambda m, d: QPIKHeal(m, d, dt=0.01, damping=1e-3, gain=0.3), "QP")
    passed &= _qp_specific_tests()
    print("QP SELF-TEST", "PASSED" if passed else "FAILED")
    sys.exit(0 if passed else 1)
