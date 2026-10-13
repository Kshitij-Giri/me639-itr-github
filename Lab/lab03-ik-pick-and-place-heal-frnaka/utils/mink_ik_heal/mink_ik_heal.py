#!/usr/bin/env python3
"""Reusable differential-IK utility for the HEAL arm using Mink.

This module does not load the scene, step MuJoCo, or apply actuator commands.
Pass in the already-compiled MuJoCo model and data, plus a desired target pose.
It returns velocities for HEAL joints joint_1 ... joint_6 in that order. The
caller (for example, a main task script) is responsible for applying those
velocities through the robot's controller and stepping the simulation.

Important: this utility is designed to work with the composite HEAL + gripper
+ free-body scene. Mink solves against the supplied model, then this module
keeps only HEAL arm joint velocities and zeros every other DOF so the cube,
container, and gripper joints are not commanded by the arm IK result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mink
import mujoco
import numpy as np


HEAL_JOINT_NAMES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
)
DEFAULT_EE_SITE = "ft_site_eff"


@dataclass(frozen=True)
class IKError:
    """Current target error magnitudes, in metres and radians."""

    position_m: float
    orientation_rad: float


class MinkIKHeal:
    """Compute HEAL arm joint-velocity updates for a desired end-effector pose.

    Args:
        model: Compiled MuJoCo model containing HEAL and optionally other scene objects.
        data: MjData associated with ``model``.
        site_name: MuJoCo site to control. Defaults to ``ft_site_eff``.
        solver: QP solver supported by the installed Mink version (default ``daqp``).
        dt: Differential-IK timestep in seconds.
        damping: Mink solver damping coefficient.
        gain: FrameTask proportional gain.
        max_joint_velocity: Optional maximum absolute velocity in rad/s for each HEAL joint.
            May be one scalar or a six-element sequence. ``None`` disables this extra limit.
            If a limit is exceeded the whole velocity vector is scaled down uniformly, so the
            direction of the Cartesian motion is preserved (per-joint clipping would bend it).
        orientation_cost: FrameTask orientation weight, scalar or 3 values expressed in the
            *site* frame (x, y, z). Use ``(1, 1, 0)`` to leave rotation about the tool axis free.
        locked_joints: HEAL joint names to hold at their current position (zero velocity,
            enforced as an exact equality constraint inside the QP).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        site_name: str = DEFAULT_EE_SITE,
        solver: str = "daqp",
        dt: float = 0.01,
        damping: float = 1e-3,
        gain: float = 0.2,
        max_joint_velocity: float | Sequence[float] | None = 1.0,
        orientation_cost: float | Sequence[float] = 1.0,
        locked_joints: Sequence[str] = (),
    ) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        if damping < 0:
            raise ValueError("damping must be non-negative")

        self.model = model
        self.data = data
        self.site_name = site_name
        self.solver = solver
        self.dt = float(dt)
        self.damping = float(damping)
        self.gain = float(gain)

        self.site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, site_name
        )
        if self.site_id < 0:
            raise ValueError(f"MuJoCo site {site_name!r} was not found in the model")

        self.joint_ids: list[int] = []
        self.dof_indices: list[int] = []
        for name in HEAL_JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Required HEAL joint {name!r} was not found in the model")
            if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"Expected {name!r} to be a hinge joint")
            self.joint_ids.append(jid)
            self.dof_indices.append(int(model.jnt_dofadr[jid]))

        if max_joint_velocity is None:
            self.max_joint_velocity = None
        else:
            limits = np.asarray(max_joint_velocity, dtype=float)
            if limits.ndim == 0:
                limits = np.full(len(HEAL_JOINT_NAMES), float(limits))
            if limits.shape != (len(HEAL_JOINT_NAMES),):
                raise ValueError("max_joint_velocity must be a scalar or six values")
            if np.any(limits <= 0):
                raise ValueError("max_joint_velocity values must be positive")
            self.max_joint_velocity = limits

        self.configuration = mink.Configuration(model)
        self.task = mink.FrameTask(
            frame_name=site_name,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=orientation_cost,
            gain=self.gain,
        )
        # Per-axis orientation weights (site frame); used to ignore unweighted axes in errors.
        self._orientation_weight = np.broadcast_to(
            np.asarray(orientation_cost, dtype=float), (3,)
        ).copy()

        # Joints held fixed by the IK (exact equality constraint, not a soft preference).
        self.locked_joints = tuple(locked_joints)
        self.constraints = []
        if self.locked_joints:
            if not hasattr(mink, "DofFreezingTask"):
                raise RuntimeError("locked_joints needs a Mink version with DofFreezingTask "
                                   "(tested with mink 1.3.0); please upgrade mink.")
            unknown = [n for n in self.locked_joints if n not in HEAL_JOINT_NAMES]
            if unknown:
                raise ValueError(f"locked_joints contains non-HEAL joints: {unknown}")
            locked_dofs = [
                self.dof_indices[HEAL_JOINT_NAMES.index(n)] for n in self.locked_joints
            ]
            self.constraints.append(mink.DofFreezingTask(model, locked_dofs))

    @staticmethod
    def _make_target(position: Sequence[float], rotation: np.ndarray) -> mink.SE3:
        """Build a Mink SE3 target from xyz position and a 3x3 rotation matrix."""
        pos = np.asarray(position, dtype=float)
        rot = np.asarray(rotation, dtype=float)
        if pos.shape != (3,):
            raise ValueError("target_position must contain exactly 3 values (x, y, z)")
        if rot.shape != (3, 3):
            raise ValueError("target_rotation must be a 3x3 rotation matrix")
        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(rot)):
            raise ValueError("Target pose must contain only finite numbers")
        if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-5) or not np.isclose(
            np.linalg.det(rot), 1.0, atol=1e-5
        ):
            raise ValueError("target_rotation must be a valid rotation matrix")
        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = pos
        return mink.SE3.from_matrix(transform)

    def current_error(self) -> IKError:
        """Return current position and orientation error relative to the last target."""
        error = np.asarray(self.task.compute_error(self.configuration), dtype=float)
        # Axes with zero orientation weight are intentionally uncontrolled; ignore them.
        orientation_error = error[3:] * (self._orientation_weight > 0.0)
        return IKError(
            position_m=float(np.linalg.norm(error[:3])),
            orientation_rad=float(np.linalg.norm(orientation_error)),
        )

    def compute_joint_velocities(
        self,
        target_position: Sequence[float],
        target_rotation: np.ndarray,
    ) -> np.ndarray:
        """Return one HEAL joint-velocity update in rad/s, ordered joint_1 ... joint_6.

        Call this repeatedly using the latest ``data.qpos`` as the simulation runs.
        The returned six values are suitable as *velocity targets* for a controller
        that maps these six HEAL joints to the correct actuators. They are not motor
        torques and should not be assigned directly to ``data.ctrl`` for this model.
        """
        self.configuration.update(self.data.qpos.copy())
        self.task.set_target(self._make_target(target_position, target_rotation))

        velocity_all = np.asarray(
            mink.solve_ik(
                self.configuration,
                [self.task],
                self.dt,
                self.solver,
                damping=self.damping,
                # Gripper/contact joints may sit slightly outside their soft MuJoCo limits;
                # do not raise because of DOFs this utility never commands.
                safety_break=False,
                constraints=self.constraints or None,
            ),
            dtype=float,
        )
        if velocity_all.shape != (self.model.nv,):
            raise RuntimeError(
                f"Mink returned velocity shape {velocity_all.shape}; expected {(self.model.nv,)}"
            )

        # Extract only the six HEAL arm DOFs. This deliberately excludes the free
        # cube/container DOFs and the gripper's internal joints from the arm command.
        arm_velocity = velocity_all[np.asarray(self.dof_indices, dtype=int)].copy()
        if self.max_joint_velocity is not None:
            ratio = float(np.max(np.abs(arm_velocity) / self.max_joint_velocity))
            if ratio > 1.0:
                arm_velocity = arm_velocity / ratio
        return arm_velocity

    def compute_full_velocity_vector(
        self,
        target_position: Sequence[float],
        target_rotation: np.ndarray,
    ) -> np.ndarray:
        """Return an nv-length generalized-velocity vector with only HEAL DOFs nonzero.

        This is useful for kinematic integration or diagnostics. It is still not a
        MuJoCo actuator-control vector; use ``compute_joint_velocities`` when feeding
        the six arm targets to the existing velocity controller.
        """
        arm_velocity = self.compute_joint_velocities(target_position, target_rotation)
        full_velocity = np.zeros(self.model.nv, dtype=float)
        full_velocity[np.asarray(self.dof_indices, dtype=int)] = arm_velocity
        return full_velocity

    def joint_velocity_dict(self, joint_velocities: Sequence[float]) -> dict[str, float]:
        """Label a six-element result with the HEAL joint names."""
        values = np.asarray(joint_velocities, dtype=float)
        if values.shape != (len(HEAL_JOINT_NAMES),):
            raise ValueError("joint_velocities must contain six values")
        return dict(zip(HEAL_JOINT_NAMES, map(float, values)))


def downward_rotation(yaw_degrees: float = 0.0) -> np.ndarray:
    """Rotation used in the workspace analysis: tool-local +Z points world -Z."""
    yaw = np.deg2rad(float(yaw_degrees))
    rz = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    rx_pi = np.diag([1.0, -1.0, -1.0])
    return rz @ rx_pi