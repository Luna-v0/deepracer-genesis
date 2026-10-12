"""Model the DeepRacer car as a domain object over its Genesis entity.

Unknown attribute access forwards to the underlying entity via ``__getattr__``.
"""

from __future__ import annotations

import os

import genesis as gs
import torch

from ..physics.limits import ackermann_angles

_ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
_URDF = os.path.join(_ASSETS, "urdf", "deepracer", "deepracer_processed.urdf")
_WHEEL_STL = os.path.join(_ASSETS, "meshes", "deepracer", "left_rear_wheel.STL")

WHEEL_DOFS = ["left_rear_wheel_joint", "right_rear_wheel_joint",
              "left_front_wheel_joint", "right_front_wheel_joint"]
STEER_DOFS = ["left_steering_hinge_joint", "right_steering_hinge_joint"]


class Car:
    """Wrap the car URDF entity together with its steering/drive controllers.

    Owns the DOF indices, PD gains, torque limits, wheel radius, and model.

    Attributes:
        entity: The wrapped Genesis car entity loaded from the URDF.
        steering_model: Steering geometry mode, ``"ackermann"`` or ``"parallel"``.
        wheel_dofs: Local DOF indices of the four drive wheels.
        steer_dofs: Local DOF indices of the two steering hinges.
        wheel_radius: Rear-wheel radius used to map linear speed to angular speed.
    """

    def __init__(self, scene, *, merge_fixed_links: bool) -> None:
        """Add the car URDF to the (unbuilt) scene; call :meth:`configure` after build.

        Args:
            scene: The Genesis scene to add the car entity to; must not yet be
                built.
            merge_fixed_links: Whether to merge fixed links when loading the
                URDF; the ``camera_link`` is always kept regardless.
        """
        # add the URDF to the (unbuilt) scene; DOFs/gains resolved in configure()
        self.entity = scene.add_entity(
            gs.morphs.URDF(file=_URDF, pos=(0, 0, 0.05),
                           merge_fixed_links=merge_fixed_links,
                           links_to_keep=["camera_link"]))
        self.steering_model = "ackermann"

    def __getattr__(self, name):
        """Forward undefined attribute lookups to the wrapped Genesis entity.

        Args:
            name: Attribute name that ``Car`` did not resolve on itself.

        Returns:
            The corresponding attribute of the underlying entity.
        """
        # forward unknown attributes to the wrapped Genesis entity so raw entity
        # methods (get_pos, set_qpos, set_friction_ratio, n_links, idx, …) stay
        # available. Only fires for attributes Car itself does not define.
        return getattr(object.__getattribute__(self, "entity"), name)

    # ------------------------------------------------ post-build configuration
    def configure(self, car_cfg: dict, device) -> None:
        """Finish car setup after ``scene.build()``: DOF indices, gains, limits, radius.

        Args:
            car_cfg: The ``car`` config section: ``steer_kp``/``steer_kv``
                (steering PD gains), ``wheel_kv`` (drive velocity gain),
                ``wheel_max_torque`` (drive torque cap), and ``steering_model``
                (``"ackermann"`` or ``"parallel"``).
            device: Torch device on which to place the gain/limit tensors.

        Raises:
            ValueError: If ``steering_model`` is not ``"ackermann"`` or
                ``"parallel"``.
        """
        e = self.entity
        self.wheel_dofs = [e.get_joint(n).dof_idx_local for n in WHEEL_DOFS]
        self.steer_dofs = [e.get_joint(n).dof_idx_local for n in STEER_DOFS]
        e.set_dofs_kp(torch.full((2,), car_cfg["steer_kp"], device=device), self.steer_dofs)
        e.set_dofs_kv(torch.full((2,), car_cfg["steer_kv"], device=device), self.steer_dofs)
        e.set_dofs_kv(torch.full((4,), car_cfg["wheel_kv"], device=device), self.wheel_dofs)
        # cap drive torque near the traction limit; unbounded torque with a
        # P velocity controller causes wheel-slip limit cycles at high speed
        tq = car_cfg["wheel_max_torque"]
        e.set_dofs_force_range(torch.full((4,), -tq, device=device),
                               torch.full((4,), tq, device=device), self.wheel_dofs)
        # steering geometry: "ackermann" (per-wheel inner/outer split, the real
        # car) or "parallel" (both hinges at the same angle, the legacy path)
        self.steering_model = car_cfg.get("steering_model", "ackermann")
        if self.steering_model not in ("ackermann", "parallel"):
            raise ValueError("steering_model must be 'ackermann' or 'parallel', "
                             f"got {self.steering_model!r}")
        import trimesh
        self.wheel_radius = float(trimesh.load(_WHEEL_STL).extents[2]) / 2.0

    # ------------------------------------------------------- steering geometry
    def steer_targets(self, delta: torch.Tensor) -> torch.Tensor:
        """Convert a commanded center steering angle into per-hinge angles.

        Args:
            delta: Commanded center steering angle as an ``(N, 1)`` tensor.

        Returns:
            An ``(N, 2)`` tensor of ``[left, right]`` hinge angles.
        """
        if self.steering_model == "parallel":
            return delta.repeat(1, 2)
        left, right = ackermann_angles(delta)
        return torch.cat([left, right], dim=1)

    # ----------------------------------------------------------- per-step control
    def drive(self, steer_center: torch.Tensor, speed: torch.Tensor) -> None:
        """Apply one control step: front steering angle and rear-wheel speed.

        Args:
            steer_center: Commanded center steering angle as an ``(N, 1)``
                tensor.
            speed: Commanded rear-wheel linear speed as an ``(N, 1)`` tensor.
        """
        self.entity.control_dofs_position(self.steer_targets(steer_center), self.steer_dofs)
        wheel_omega = (speed / self.wheel_radius).repeat(1, 4)
        self.entity.control_dofs_velocity(wheel_omega, self.wheel_dofs)

    # ------------------------------------------------------------------- reset
    def reset_pose(self, qpos: torch.Tensor, env_ids: torch.Tensor) -> None:
        """Teleport the selected envs to a pose and zero their controllers.

        Args:
            qpos: Target joint positions for the reset envs.
            env_ids: Indices of the environments to reset.
        """
        e = self.entity
        n = len(env_ids)
        dev = qpos.device
        e.set_qpos(qpos, envs_idx=env_ids)
        e.zero_all_dofs_velocity(envs_idx=env_ids)
        e.control_dofs_position(torch.zeros(n, 2, device=dev), self.steer_dofs, envs_idx=env_ids)
        e.control_dofs_velocity(torch.zeros(n, 4, device=dev), self.wheel_dofs, envs_idx=env_ids)

    # -------------------------------------------------------------- kinematics
    def kinematics(self):
        """Read the current base-link kinematic state.

        Returns:
            A ``(pos, quat, vel, ang)`` tuple of the base link: position,
            orientation quaternion, linear velocity, and angular velocity.
        """
        e = self.entity
        return e.get_pos(), e.get_quat(), e.get_vel(), e.get_ang()
