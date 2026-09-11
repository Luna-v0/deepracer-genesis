"""The perception CNN: a stack of camera frames in, physical scalars out.

It predicts the channels of ``PerceptionFeatures`` a camera can plausibly
recover; the rest stay computed onboard, as they are on the car.
"""

from __future__ import annotations

import torch
from torch import nn

#: the physical quantities the CNN predicts, in output order
CHANNEL_NAMES = ("lateral", "heading", "speed", "yaw_rate", "beta",
                 "curv@1m", "curv@3m")

#: root of each channel's validation MSE, measured on the held-out tracks
SIGMA = (0.125, 0.064, 0.060, 0.065, 0.083, 0.122, 0.224)

DEFAULT_INPUT_HW = (120, 160)

#: the stock trunk/head — the arch every bare (pre-arch-payload) checkpoint has
DEFAULT_ARCH = {"channels": (32, 64, 64, 64), "kernels": (5, 3, 3, 3),
                "strides": (2, 2, 2, 2), "head": 128}


def _conv_output_hw(input_hw: tuple[int, int],
                    kernels: tuple[int, ...] = DEFAULT_ARCH["kernels"],
                    strides: tuple[int, ...] = DEFAULT_ARCH["strides"],
                    ) -> tuple[int, int]:
    """Return the spatial size after the convolution stack.

    Args:
        input_hw: Frame height and width in pixels.
        kernels: Kernel size per conv layer.
        strides: Stride per conv layer.

    Returns:
        The height and width entering the dense head.

    Raises:
        ValueError: If the frame shrinks to nothing before the last layer.
    """
    h, w = input_hw
    for kernel, stride in zip(kernels, strides):
        h, w = (h - kernel) // stride + 1, (w - kernel) // stride + 1
        if h < 1 or w < 1:
            raise ValueError(
                f"input {input_hw} is too small for this convolution stack; "
                f"it collapses to {(h, w)}")
    return h, w


class PerceptionCNN(nn.Module):
    """Strided convolutions and a two-layer head over a frame stack.

    The stack is parameterized for architecture search; the defaults are the
    stock network, so pre-existing checkpoints load unchanged. Whatever the
    arch, the first conv is ``features.0`` and the output layer is ``head.3``.

    Attributes:
        features: The convolution stack.
        head: The dense head mapping to the predicted channels.
        arch: The constructor arguments, for arch-carrying checkpoints.
    """

    def __init__(self, in_channels: int = 12, n_targets: int = 7,
                 input_hw: tuple[int, int] = DEFAULT_INPUT_HW,
                 channels: tuple[int, ...] = DEFAULT_ARCH["channels"],
                 kernels: tuple[int, ...] = DEFAULT_ARCH["kernels"],
                 strides: tuple[int, ...] = DEFAULT_ARCH["strides"],
                 head: int = DEFAULT_ARCH["head"]) -> None:
        """Build the network for one frame-stack shape and architecture.

        Args:
            in_channels: Channels of the stacked input, ``3 * frame_stack``.
            n_targets: Number of physical quantities predicted.
            input_hw: Frame height and width the dense head is sized for.
            channels: Output channels per conv layer.
            kernels: Kernel size per conv layer.
            strides: Stride per conv layer.
            head: Hidden width of the two-layer dense head.

        Raises:
            ValueError: If the per-layer tuples disagree in length, or
                ``input_hw`` is too small for the convolution stack.
        """
        super().__init__()
        if not (len(channels) == len(kernels) == len(strides)):
            raise ValueError(
                f"channels/kernels/strides must have equal lengths; got "
                f"{len(channels)}/{len(kernels)}/{len(strides)}")
        self.arch = {"in_channels": in_channels, "n_targets": n_targets,
                     "input_hw": tuple(input_hw), "channels": tuple(channels),
                     "kernels": tuple(kernels), "strides": tuple(strides),
                     "head": head}
        layers: list[nn.Module] = []
        c_in = in_channels
        for c, k, s in zip(channels, kernels, strides):
            layers += [nn.Conv2d(c_in, c, kernel_size=k, stride=s), nn.ReLU()]
            c_in = c
        self.features = nn.Sequential(*layers)
        h, w = _conv_output_hw(input_hw, tuple(kernels), tuple(strides))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_in * h * w, head), nn.ReLU(),
            nn.Linear(head, n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of frame stacks to the predicted channels.

        Args:
            x: An ``(N, in_channels, H, W)`` float tensor in ``[0, 1]``.

        Returns:
            An ``(N, n_targets)`` tensor of predictions.
        """
        return self.head(self.features(x))


def save_checkpoint(net: PerceptionCNN, path) -> None:
    """Write an arch-carrying checkpoint: the architecture travels with it.

    Args:
        net: The network to checkpoint.
        path: Destination file.
    """
    torch.save({"arch": dict(net.arch), "state_dict": net.state_dict()}, path)


def load_checkpoint(source, map_location="cpu") -> PerceptionCNN:
    """Rebuild a ``PerceptionCNN`` from a checkpoint path or loaded payload.

    Accepts arch-carrying ``{"arch", "state_dict"}`` payloads and bare stock
    state dicts (probed via ``features.0.weight`` / ``head.3.weight``).

    Args:
        source: A checkpoint path, or an already-loaded payload dict.
        map_location: Device the weights are loaded onto.

    Returns:
        The rebuilt network in eval mode.
    """
    state = (source if isinstance(source, dict)
             else torch.load(source, map_location=map_location,
                             weights_only=True))
    if "state_dict" in state:
        arch = {k: tuple(v) if isinstance(v, (list, tuple)) else v
                for k, v in state["arch"].items()}
        net = PerceptionCNN(**arch)
        net.load_state_dict(state["state_dict"])
    else:
        # a bare state dict can only be the stock architecture
        net = PerceptionCNN(in_channels=state["features.0.weight"].shape[1],
                            n_targets=state["head.3.weight"].shape[0])
        net.load_state_dict(state)
    return net.eval()
