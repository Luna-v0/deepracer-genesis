"""The perception CNN: a stack of camera frames in, seven scalars out.

The seven outputs are exactly the channels of ``PerceptionFeatures`` a camera
can plausibly recover (lateral offset, heading, speed, yaw rate, slip angle and
two curvatures ahead). The other 22 channels stay computed onboard, as they are
on the real car.
"""

import torch
import torch.nn as nn


class PerceptionCNN(nn.Module):
    def __init__(self, in_channels=12, n_targets=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 8, 128), nn.ReLU(),   # 6x8 = a 120x160 frame after the four strides
            nn.Linear(128, n_targets),
        )

    def forward(self, x):
        return self.head(self.features(x))


if __name__ == "__main__":
    net = PerceptionCNN()
    x = torch.zeros(8, 12, 120, 160)

    print("shape at each stage:")
    print(f"  input         {tuple(x.shape)}")
    for layer in net.features:
        x = layer(x)
        if isinstance(layer, torch.nn.Conv2d):
            print(f"  after conv    {tuple(x.shape)}")
    y = net.head(x)
    print(f"  output        {tuple(y.shape)}")

    n = sum(p.numel() for p in net.parameters())
    print(f"\nlearnable parameters: {n:,}")
