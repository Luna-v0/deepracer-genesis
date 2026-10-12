"""Temporal camera DR: observation latency + frame drop (Part P.2).

Unlike the per-frame effects in :mod:`~deepracer_genesis.randomization.image_aug`,
latency is *stateful* — the frame a policy sees this step depends on earlier
frames — so it lives in a small object the env advances once per step, not in the
stateless ``apply_image_aug`` pipeline.

Motivation (from the sim2real DR study): a real DeepRacer camera adds a
1-2 frame pipeline delay and occasionally repeats a stale frame, and for a 4 m/s
car that latency is likely the single largest untreated sim2real gap. The sim
renders with zero delay, so we inject it here.
"""

from __future__ import annotations

import torch


class FrameLatency:
    """Per-env camera latency and frame-drop, advanced once per control step.

    Emits the frame from ``latency`` steps ago (0 = current), then with
    probability ``drop`` repeats the previously emitted frame (a dropped or
    stale sensor read). Reset-aware: :meth:`reset` clears an env's history so a
    respawned car starts from its fresh frame with no cross-episode bleed.

    The frame history is sized lazily on the first :meth:`advance` call, so the
    image resolution need not be known at construction.

    Attributes:
        latency: Number of steps a frame is delayed before the policy sees it.
        drop: Per-env probability of repeating the previous emitted frame.
    """

    def __init__(self, num_envs: int, latency: int, drop: float, device) -> None:
        """Configure the buffer.

        Args:
            num_envs: Number of parallel envs (rows of the frame batch).
            latency: Delay in control steps (``0`` disables the delay).
            drop: Frame-drop probability in ``[0, 1]`` (``0`` disables drops).
            device: Torch device the frames live on.
        """
        self.num_envs = int(num_envs)
        self.latency = max(0, int(latency))
        self.drop = float(drop)
        self._device = device
        self._initialized = False
        self._hist: torch.Tensor | None = None   # (N, latency, C, H, W)
        self._last: torch.Tensor | None = None    # (N, C, H, W) last emitted
        # rows to re-seed (freshly reset envs) on the next advance
        self._reinit = torch.ones(self.num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Mark ``env_ids`` to be re-seeded from their next frame.

        Args:
            env_ids: Indices of envs that were just respawned.
        """
        if len(env_ids) > 0:
            self._reinit[env_ids] = True

    def advance(self, frame: torch.Tensor) -> torch.Tensor:
        """Push ``frame`` and return the (possibly delayed/dropped) obs frame.

        Args:
            frame: Current rendered frames ``(N, C, H, W)``.

        Returns:
            The frame the policy should observe this step, same shape as
            ``frame``.
        """
        if not self._initialized:
            # lazily size to the frame; seed every row with the current frame so
            # the first `latency` steps show a real image, not black
            self._initialized = True
            if self.latency > 0:
                self._hist = frame.unsqueeze(1).repeat(1, self.latency, 1, 1, 1).clone()
            self._last = frame.clone()
            self._reinit[:] = False
            return frame if self.latency == 0 else self._delay(frame)

        # re-seed freshly reset rows so no stale pre-reset frame leaks through
        if self._reinit.any():
            rows = self._reinit
            if self._hist is not None:
                self._hist[rows] = frame[rows].unsqueeze(1)
            self._last[rows] = frame[rows]
            self._reinit[:] = False

        out = self._delay(frame) if self.latency > 0 else frame
        if self.drop > 0:
            keep_prev = torch.rand(self.num_envs, device=frame.device) < self.drop
            out = torch.where(keep_prev[:, None, None, None], self._last, out)
        self._last = out.clone()
        return out

    def reseed(self, env_ids: torch.Tensor, frame: torch.Tensor) -> None:
        """Restart ``env_ids`` from ``frame`` without advancing any other env.

        A respawned env then observes ``frame`` until ``latency`` steps pass.

        Args:
            env_ids: Indices of envs that were just respawned.
            frame: Current rendered frames ``(N, C, H, W)``.
        """
        if len(env_ids) == 0:
            return
        if not self._initialized:
            self.advance(frame)        # no history yet: seeds every row
            return
        if self._hist is not None:
            self._hist[env_ids] = frame[env_ids].unsqueeze(1)
        self._last[env_ids] = frame[env_ids]
        self._reinit[env_ids] = False

    def _delay(self, frame: torch.Tensor) -> torch.Tensor:
        """Ring-buffer step: emit the oldest frame, push the newest."""
        out = self._hist[:, -1].clone()
        self._hist = torch.cat([frame.unsqueeze(1), self._hist[:, :-1]], dim=1)
        return out
