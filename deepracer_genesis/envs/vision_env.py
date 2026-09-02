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
        if self._frame_stack > 1:
            self._stack_buf = torch.zeros(
                self.num_envs, 3 * self._frame_stack, h, w, device=self.device)
            self._stack_prime = torch.ones(self.num_envs, dtype=torch.bool,
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

    def _observe_camera(self) -> None:
        """Refresh the camera buffers from the renderer for the current state.

        Applies image-space DR to the observed frame; image_buf stays un-augmented.
        """
        self.image_buf, self.obs_image_buf = self.renderer.render(self)
        if self.image_aug:
            from ..randomization.image_aug import apply_image_aug
            self.obs_image_buf = apply_image_aug(self.obs_image_buf, self.image_aug)

    def _finalize_obs(self) -> None:
        """Advance the camera-latency buffer once for the step's final frame.

        Runs after any auto-reset re-render, so the delayed frame the policy
        sees reflects the buffer state exactly one step forward. The frame
        stack is pushed AFTER latency so the stack holds the (possibly
        delayed) frames the policy actually observes — same as the car, where
        the stack is filled from arriving (already-latent) camera messages.
        """
        if self._frame_latency is not None:
            self.obs_image_buf = self._frame_latency.advance(self.obs_image_buf)
        if self._stack_buf is not None:
            f = self.obs_image_buf
            self._stack_buf = torch.cat([self._stack_buf[:, 3:], f], dim=1)
            if self._stack_prime.any():
                idx = self._stack_prime
                self._stack_buf[idx] = f[idx].repeat(1, self._frame_stack, 1, 1)
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
            # Initial observation before the first _finalize_obs: prime the
            # stack in place (no shift) so it never exposes zero frames.
            if self._stack_prime.any():
                idx = self._stack_prime
                self._stack_buf[idx] = self.obs_image_buf[idx].repeat(
                    1, self._frame_stack, 1, 1)
                self._stack_prime[idx] = False
            groups["camera"] = self._stack_buf
        else:
            groups["camera"] = self.obs_image_buf
        return groups
