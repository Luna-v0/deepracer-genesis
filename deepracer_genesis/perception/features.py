"""Feature sets whose camera-recoverable channels come from a CNN or from noise.

Both replace the same slice of :class:`PerceptionFeatures`; the other channels
stay computed onboard, exactly as they are on the car.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from deepracer_genesis.envs.features import PerceptionFeatures
from deepracer_genesis.perception.model import PerceptionCNN

if TYPE_CHECKING:
    from deepracer_genesis.envs.deepracer_env import DeepRacerEnv

CHANNEL_NAMES = ("lateral", "heading", "speed", "yaw_rate", "beta",
                 "curv@1m", "curv@3m")

# root of each channel's validation MSE, measured on the held-out tracks
SIGMA = (0.125, 0.064, 0.060, 0.065, 0.083, 0.122, 0.224)


def _frame_stack_of(env: "DeepRacerEnv") -> int:
    """Return the env's configured frame-stack depth.

    Args:
        env: The env being built.

    Returns:
        The number of camera frames stacked along the channel axis.
    """
    return int(env.cfg["vision"].get("frame_stack", 1) or 1)


class CNNPerceptionFeatures(PerceptionFeatures):
    """PerceptionFeatures whose camera-recoverable channels come from a frozen CNN.

    Attributes:
        net: The frozen perception CNN, in eval mode with grads disabled.
        device: Device the CNN runs on.
    """

    def __init__(self, env: "DeepRacerEnv", params: dict) -> None:
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
        state = torch.load(Path(checkpoint), map_location=self.device,
                           weights_only=True)
        in_channels = state["features.0.weight"].shape[1]
        expected = 3 * _frame_stack_of(env)
        if in_channels != expected:
            raise ValueError(
                f"checkpoint {checkpoint} reads {in_channels} channels but this "
                f"env stacks {_frame_stack_of(env)} frames ({expected} channels). "
                "Retrain the CNN or set frame_stack to match.")

        lo, hi = self.cnn_target_slice
        n_targets = state["head.3.weight"].shape[0]
        if n_targets != hi - lo:
            raise ValueError(
                f"checkpoint {checkpoint} predicts {n_targets} channels but the "
                f"feature set's cnn_target_slice spans {hi - lo}")

        self.net = PerceptionCNN(in_channels=in_channels,
                                 n_targets=n_targets).to(self.device).eval()
        self.net.load_state_dict(state)
        for p in self.net.parameters():
            p.requires_grad_(False)

    def compute(self) -> torch.Tensor:
        """Overwrite the camera-recoverable channels with the CNN's estimates.

        Returns:
            The feature vector with ``cnn_target_slice`` replaced by CNN output.

        Raises:
            RuntimeError: If the env exposes no camera stack.
        """
        x = super().compute()
        stack = self.env.camera_stack
        if stack is None:
            raise RuntimeError(
                "camera_stack is None: the env stopped stacking frames after "
                "this feature set was built")
        lo, hi = self.cnn_target_slice
        with torch.inference_mode():
            y = self.net(stack.to(self.device))
        x[:, lo:hi] = y.to(x.device)
        return x


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

        Returns:
            The feature vector with ``cnn_target_slice`` perturbed.

        Raises:
            ValueError: If the target slice is wider than the known channels.
        """
        x = super().compute()
        strength = float(self.params.get("noise", 0.0))
        if not strength:
            return x
        lo, hi = self.cnn_target_slice
        if hi - lo > len(SIGMA):
            raise ValueError(
                f"cnn_target_slice spans {hi - lo} channels but only "
                f"{len(SIGMA)} per-channel sigmas are known; extend SIGMA "
                "before widening the slice")
        sigma = self._sigma(x.device, x.dtype)[:hi - lo]
        x[:, lo:hi] = x[:, lo:hi] + torch.randn_like(x[:, lo:hi]) * sigma * strength
        return x
