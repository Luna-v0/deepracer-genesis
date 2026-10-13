"""Provide the camera-observation DeepRacer environment.

Add a ``camera`` observation group backed by an image buffer refreshed each step.
"""

from __future__ import annotations

import torch

from .base_env import DeepRacerEnv


class VisionDeepRacerEnv(DeepRacerEnv):
    """DeepRacer environment that exposes a rendered camera observation.

    Owns the rendered image buffer and publishes it as an extra observation group.

    Attributes:
        image_buf: Full-resolution rendered camera frames for all parallel envs.
        obs_image_buf: Policy-resolution camera frames published as observations.
        frame_stride: Control steps between stacked frames (1 = consecutive).
        frame_interval_s: Seconds between stacked frames (``dt * frame_stride``).
    """

    def _init_obs_buffers(self, env_cfg: dict) -> None:
        """Preallocate the camera image buffers for all parallel envs.

        Args:
            env_cfg: Environment configuration; ``camera_res`` gives the render
                resolution as a ``(width, height)`` pair and ``frame_stack``
                the number of frames stacked along the channel axis.
        """
        w, h = env_cfg["vision"]["camera_res"]
        self.image_buf = torch.zeros(self.num_envs, 3, h, w, device=self.device)
        # policy may train below render resolution (demo videos); render() sets both
        self.obs_image_buf = self.image_buf
        # Frame stacking (deployment contract, mirrored by the car node):
        # k frames along channels, OLDEST FIRST (newest = last 3 channels);
        # fresh episodes prime by REPEATING the first frame (zeros never occur
        # on a real camera). Order and priming are part of the model card.
        self._frame_stack = int(env_cfg["vision"].get("frame_stack", 1) or 1)
        self.frame_stride = int(env_cfg["vision"].get("frame_stride", 1) or 1)
        self.frame_interval_s = self.dt * self.frame_stride
        self._ring = None
        if self._frame_stack > 1:
            self._stack_buf = torch.zeros(
                self.num_envs, 3 * self._frame_stack, h, w, device=self.device)
            self._stack_prime = torch.ones(self.num_envs, dtype=torch.bool,
                                           device=self.device)
            if self.frame_stride > 1:
                # ADR 0005: the stack holds every stride-th frame, so a ring
                # of (k-1)*stride+1 frames backs it; stride 1 needs no ring.
                depth = (self._frame_stack - 1) * self.frame_stride + 1
                self._ring = torch.zeros(self.num_envs, depth, 3, h, w,
                                         device=self.device)
        else:
            self._stack_buf = None
        # stateful temporal DR (camera latency / frame drop); None when disabled
        lat = int(self.image_aug.get("latency_steps", 0))
        drop = float(self.image_aug.get("frame_drop", 0.0))
        if lat or drop:
            from ..randomization.latency import FrameLatency
            self._frame_latency = FrameLatency(self.num_envs, lat, drop, self.device)
        else:
            self._frame_latency = None

    @property
    def camera_stack(self) -> torch.Tensor | None:
        """Return the stacked camera frames the policy observes.

        Oldest frame first, matching the deployment contract mirrored by the car.

        Returns:
            An ``(N, 3 * frame_stack, H, W)`` float tensor in ``[0, 1]``, or
            ``None`` when ``frame_stack`` is 1 and no stack is kept.
        """
        return self._stack_buf

    def _observe_camera(self, env_ids: torch.Tensor | None = None) -> None:
        """Render the current frame, then advance camera latency and the stack.

        Image-space DR applies to the observed frame; image_buf stays un-augmented.

        Args:
            env_ids: Envs just respawned, the only ones a re-render restarts;
                ``None`` advances every env by one control step.
        """
        self.image_buf, frame = self.renderer.render(self)
        if self.image_aug:
            from ..randomization.image_aug import apply_image_aug
            frame = apply_image_aug(frame, self.image_aug)
        if env_ids is None:
            # Once per control step, before the feature set reads the stack
            # (B-001). Latency first, then the push, so the stack holds the
            # (possibly delayed) frames the policy observes — as on the car.
            if self._frame_latency is not None:
                frame = self._frame_latency.advance(frame)
            if self._stack_buf is not None:
                self._push_stack(frame)
            self.obs_image_buf = frame
        else:
            # Re-render after a respawn: every other env already advanced
            # this step, so only the respawned rows take the new frame.
            if self._frame_latency is not None:
                self._frame_latency.reseed(env_ids, frame)
            obs = self.obs_image_buf.clone()
            obs[env_ids] = frame[env_ids]
            self.obs_image_buf = obs
            if self._stack_buf is not None:
                self._stack_prime[env_ids] = True
        self._prime_stack()

    def _push_stack(self, frame: torch.Tensor) -> None:
        """Advance the stack by one control step with the observed ``frame``."""
        if self._ring is None:
            self._stack_buf = torch.cat([self._stack_buf[:, 3:], frame], dim=1)
            return
        self._ring = torch.cat([self._ring[:, 1:], frame.unsqueeze(1)], dim=1)
        # every stride-th ring slot, oldest first: slot 0 is (k-1)*stride
        # steps old, the last slot is this step's frame
        self._stack_buf = self._ring[:, ::self.frame_stride].flatten(1, 2)

    def _prime_stack(self) -> None:
        """Fill respawned envs' stacks with their current frame, repeated."""
        if self._stack_buf is not None and self._stack_prime.any():
            idx = self._stack_prime
            self._stack_buf[idx] = self.obs_image_buf[idx].repeat(
                1, self._frame_stack, 1, 1)
            if self._ring is not None:
                self._ring[idx] = self.obs_image_buf[idx].unsqueeze(1).expand(
                    -1, self._ring.shape[1], -1, -1, -1)
            self._stack_prime[idx] = False

    def _reset_obs_dr(self, env_ids: torch.Tensor) -> None:
        """Drop camera-latency and frame-stack history for respawned envs."""
        if self._frame_latency is not None:
            self._frame_latency.reset(env_ids)
        if self._stack_buf is not None:
            self._stack_prime[env_ids] = True

    def _obs_groups(self) -> dict:
        """Assemble the observation groups, adding the ``camera`` frame(s).

        Returns:
            The base observation groups augmented with a ``camera`` entry:
            the current policy-resolution frame, or the k-frame channel stack
            when ``frame_stack`` > 1.
        """
        groups = super()._obs_groups()
        if self._stack_buf is not None:
            # _observe_camera already primed; this covers a reset_idx that
            # was not followed by a re-render, so zeros are never exposed.
            self._prime_stack()
            groups["camera"] = self._stack_buf
        else:
            groups["camera"] = self.obs_image_buf
        return groups
