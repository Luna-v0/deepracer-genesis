"""Scripted agents that walk the environment (no learning involved).

These drive the sim for data collection, DR previews and track sanity
checks. They read PRIVILEGED simulator state (the exact track-frame
quantities the reward uses), so they need no training — subclass
`PrivilegedAgent` and override `act()` to write your own behavior:

    class Weaver(PrivilegedAgent):
        def act(self, sim):
            steer = torch.sin(sim.progress_m)          # slalom forever
            return torch.stack([steer, torch.full_like(steer, -0.3)], dim=1)

    collect_rollout_dataset(pipeline, agent=Weaver(), ...)

Useful sim attributes (all (N,) CUDA tensors, direction-aware):
    sim.lateral        signed offset from the centerline (meters)
    sim.half_width     road half-width at the car
    sim.heading_err    yaw error vs the track tangent (0 = aligned)
    sim.dir_sign       +1/-1 driving direction (see random_direction)
    sim.v_forward      forward speed (m/s)
    sim.progress_m     arclength position along the track
"""

from __future__ import annotations

import torch


class PrivilegedAgent:
    """Base scripted driver. `act(sim) -> (N, 2) actions in [-1, 1]`
    ([steering, throttle]); `reset(env_ids)` clears per-env state after
    respawns (optional)."""

    def act(self, sim) -> torch.Tensor:
        raise NotImplementedError

    def reset(self, env_ids: torch.Tensor) -> None:
        pass


class CenterlineFollower(PrivilegedAgent):
    """Deterministic P-controller on lateral offset + heading error.

    Drives cleanly at moderate speed on every track — the baseline expert
    used by DR previews and track sanity drives.
    """

    def __init__(self, k_lateral: float = 1.1, k_heading: float = 0.9,
                 throttle: float = -0.3):
        """Configure the P-controller.

        Args:
            k_lateral: Gain on the width-normalized lateral offset.
            k_heading: Gain on sin(heading error).
            throttle: Constant throttle command in [-1, 1] (-1 = slowest).
        """
        self.k_lateral = k_lateral
        self.k_heading = k_heading
        self.throttle = throttle

    def act(self, sim) -> torch.Tensor:
        lat = sim.lateral * sim.dir_sign / sim.half_width.clamp(min=0.1)
        steer = -(self.k_lateral * lat
                  + self.k_heading * torch.sin(sim.heading_err))
        return torch.stack(
            [steer, torch.full_like(steer, self.throttle)], dim=1).clamp(-1, 1)


class NoisyExpert(CenterlineFollower):
    """CenterlineFollower + Ornstein-Uhlenbeck action noise.

    The OU process gives temporally-CORRELATED wandering (white noise just
    jitters in place): trajectories drift off the centerline and sometimes
    off the track entirely — episodes end, respawn, and the dataset gets the
    recovery/failure frames a perception model needs to see.
    """

    def __init__(self, noise: float = 0.35, theta: float = 0.05, **kwargs):
        """Configure the OU noise on top of the follower gains.

        Args:
            noise: Stationary standard deviation of the OU process.
            theta: Mean-reversion rate (smaller = longer excursions).
            **kwargs: Forwarded to CenterlineFollower (k_lateral, k_heading,
                throttle).
        """
        super().__init__(**kwargs)
        self.noise = noise
        self.theta = theta
        self._ou: torch.Tensor | None = None

    def act(self, sim) -> torch.Tensor:
        base = super().act(sim)
        if self._ou is None or self._ou.shape[0] != base.shape[0]:
            self._ou = torch.zeros_like(base)
        self._ou.mul_(1.0 - self.theta).add_(
            torch.randn_like(self._ou),
            alpha=self.noise * (2 * self.theta) ** 0.5)
        return (base + self._ou).clamp(-1.0, 1.0)

    def reset(self, env_ids: torch.Tensor) -> None:
        if self._ou is not None:
            self._ou[env_ids] = 0.0
