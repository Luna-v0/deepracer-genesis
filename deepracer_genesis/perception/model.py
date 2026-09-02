"""The perception CNN: a stack of camera frames in, physical scalars out.

It predicts the channels of ``PerceptionFeatures`` a camera can plausibly
recover; the rest stay computed onboard, as they are on the car.
"""

from __future__ import annotations

import torch
from torch import nn

DEFAULT_INPUT_HW = (120, 160)
_CONVS = ((5, 2), (3, 2), (3, 2), (3, 2))   # (kernel, stride) per conv layer


def _conv_output_hw(input_hw: tuple[int, int]) -> tuple[int, int]:
    """Return the spatial size after the convolution stack.

    Args:
        input_hw: Frame height and width in pixels.

    Returns:
        The height and width entering the dense head.

    Raises:
        ValueError: If the frame shrinks to nothing before the last layer.
    """
    h, w = input_hw
    for kernel, stride in _CONVS:
        h, w = (h - kernel) // stride + 1, (w - kernel) // stride + 1
        if h < 1 or w < 1:
            raise ValueError(
                f"input {input_hw} is too small for this convolution stack; "
                f"it collapses to {(h, w)}")
    return h, w


class PerceptionCNN(nn.Module):
    """Four strided convolutions and a two-layer head over a frame stack.

    Attributes:
        features: The convolution stack.
        head: The dense head mapping to the predicted channels.
    """

    def __init__(self, in_channels: int = 12, n_targets: int = 7,
                 input_hw: tuple[int, int] = DEFAULT_INPUT_HW) -> None:
        """Build the network for one frame-stack shape.

        Args:
            in_channels: Channels of the stacked input, ``3 * frame_stack``.
            n_targets: Number of physical quantities predicted.
            input_hw: Frame height and width the dense head is sized for.

        Raises:
            ValueError: If ``input_hw`` is too small for the convolution stack.
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2), nn.ReLU(),
        )
        h, w = _conv_output_hw(input_hw)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * h * w, 128), nn.ReLU(),
            nn.Linear(128, n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of frame stacks to the predicted channels.

        Args:
            x: An ``(N, in_channels, H, W)`` float tensor in ``[0, 1]``.

        Returns:
            An ``(N, n_targets)`` tensor of predictions.
        """
        return self.head(self.features(x))
