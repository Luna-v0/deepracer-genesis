"""Feature sets whose camera-recoverable channels come from a CNN or from noise.

Both build the error channels from those estimates, as the car must, not the true state.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from deepracer_genesis.envs.features import PerceptionFeatures
from deepracer_genesis.perception.model import CHANNEL_NAMES, SIGMA, load_checkpoint
from deepracer_genesis.physics.limits import MAX_SPEED, YAW_RATE_NORM

if TYPE_CHECKING:
    from deepracer_genesis.envs.deepracer_env import DeepRacerEnv

__all__ = ["CHANNEL_NAMES", "SIGMA", "CNNPerceptionFeatures",
           "NoisyPerceptionFeatures"]


def _frame_stack_of(env: DeepRacerEnv) -> int:
    """Return the env's configured frame-stack depth.

    Args:
        env: The env being built.

    Returns:
        The number of camera frames stacked along the channel axis.
    """
    return int(env.cfg["vision"].get("frame_stack", 1) or 1)


def _speed_and_yaw_rate(targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the speed (m/s) and yaw rate (rad/s) encoded in estimated targets.

    Args:
        targets: ``(N, n_targets)`` estimates in the targets' normalized units.

    Returns:
        The de-normalized speed and yaw rate the error channels are built from.
    """
    speed = targets[:, CHANNEL_NAMES.index("speed")] * MAX_SPEED
    yaw_rate = targets[:, CHANNEL_NAMES.index("yaw_rate")] * YAW_RATE_NORM
    return speed, yaw_rate


class CNNPerceptionFeatures(PerceptionFeatures):
    """PerceptionFeatures whose camera-recoverable channels come from a frozen CNN.

    Attributes:
        net: The frozen perception CNN, in eval mode with grads disabled.
        device: Device the CNN runs on.
    """

    def __init__(self, env: DeepRacerEnv, params: dict) -> None:
        """Load the frozen CNN and check it matches the env's camera config.

        Args:
            env: The env this feature set is attached to.
            params: Requires ``checkpoint`` (path to the CNN state dict);
                optional ``cnn_device`` overrides where the CNN runs.

        Raises:
            ValueError: If the env has no camera, if ``checkpoint`` is missing,
                or if the checkpoint's input width does not match
                ``3 * frame_stack``.
        """
        super().__init__(env, params)
        if not getattr(env, "vision", False):
            raise ValueError(
                "CNNPerceptionFeatures needs a camera env (CameraEnvironment); "
                "got a feature-only env, which renders no frames to read")
        checkpoint = params.get("checkpoint")
        if checkpoint is None:
            raise ValueError(
                "CNNPerceptionFeatures requires params['checkpoint'] — the path "
                "to the trained CNN. There is deliberately no default: a missing "
                "or stale checkpoint would silently change what is measured.")

        self.device = (torch.device(params["cnn_device"]) if "cnn_device" in params
                       else env.device)
        # arch-carrying payloads rebuild their own (possibly searched)
        # architecture; bare state dicts fall back to the stock PerceptionCNN
        net = load_checkpoint(Path(checkpoint), map_location=str(self.device))
        in_channels = net.arch["in_channels"]
        expected = 3 * _frame_stack_of(env)
        if in_channels != expected:
            raise ValueError(
                f"checkpoint {checkpoint} reads {in_channels} channels but this "
                f"env stacks {_frame_stack_of(env)} frames ({expected} channels). "
                "Retrain the CNN or set frame_stack to match.")

        lo, hi = self.cnn_target_slice
        n_targets = net.arch["n_targets"]
        if n_targets != hi - lo:
            raise ValueError(
                f"checkpoint {checkpoint} predicts {n_targets} channels but the "
                f"feature set's cnn_target_slice spans {hi - lo}")

        self.net = net.to(self.device)
        for p in self.net.parameters():
            p.requires_grad_(False)

    def compute(self) -> torch.Tensor:
        """Replace the camera-recoverable channels with the CNN's estimates.

        The error channels follow the CNN's speed and yaw rate, not the true ones.

        Returns:
            The feature vector with ``cnn_target_slice`` replaced by CNN output.

        Raises:
            RuntimeError: If the env exposes no camera stack.
        """
        stack = self.env.camera_stack
        if stack is None:
            raise RuntimeError(
                "camera_stack is None: the env stopped stacking frames after "
                "this feature set was built")
        targets = self._targets()
        lo, hi = self.cnn_target_slice
        with torch.inference_mode():
            y = self.net(stack.to(self.device))
        targets[:, lo:hi] = y.to(targets.device)
        return self._with_errors(targets, *_speed_and_yaw_rate(targets))


class NoisyPerceptionFeatures(PerceptionFeatures):
    """PerceptionFeatures with Gaussian noise the size of the CNN's error.

    No CNN runs and no frame is rendered, so a policy trains at feature-env
    speed; the noise is white, where the real CNN's error is time-correlated.
    """

    def _sigma(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return the per-channel noise scale, zeroed outside the chosen channels.

        Args:
            device: Device to allocate on.
            dtype: Dtype to allocate as.

        Returns:
            A ``(len(CHANNEL_NAMES),)`` tensor of standard deviations.

        Raises:
            ValueError: If ``noise_channels`` names an unknown channel.
        """
        chosen = self.params.get("noise_channels")
        if chosen is not None:
            unknown = set(chosen) - set(CHANNEL_NAMES)
            if unknown:
                raise ValueError(
                    f"unknown noise_channels {sorted(unknown)}; "
                    f"valid names are {list(CHANNEL_NAMES)}")
        scales = [s if chosen is None or n in chosen else 0.0
                  for n, s in zip(CHANNEL_NAMES, SIGMA)]
        return torch.tensor(scales, device=device, dtype=dtype)

    def compute(self) -> torch.Tensor:
        """Add scaled noise to the camera-recoverable channels.

        The error channels follow the noised speed and yaw rate, not the true ones.

        Returns:
            The feature vector with ``cnn_target_slice`` perturbed.

        Raises:
            ValueError: If the target slice is wider than the known channels.
        """
        strength = float(self.params.get("noise", 0.0))
        if not strength:
            return super().compute()
        targets = self._targets()
        lo, hi = self.cnn_target_slice
        if hi - lo > len(SIGMA):
            raise ValueError(
                f"cnn_target_slice spans {hi - lo} channels but only "
                f"{len(SIGMA)} per-channel sigmas are known; extend SIGMA "
                "before widening the slice")
        sigma = self._sigma(targets.device, targets.dtype)[:hi - lo]
        targets[:, lo:hi] = (targets[:, lo:hi]
                             + torch.randn_like(targets[:, lo:hi]) * sigma * strength)
        return self._with_errors(targets, *_speed_and_yaw_rate(targets))
