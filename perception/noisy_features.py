"""PerceptionFeatures with noise the size of the CNN's error on the camera channels.

No CNN runs here and no image is rendered. The 7 camera channels keep their
exact simulator values and get Gaussian noise added, scaled per channel to the
root of the CNN's validation MSE; the other 22 (past actions, command deltas)
are computed onboard and stay exact.

That makes a policy trainable at feature-env speed, but the noise is
independent from step to step and across channels, where the real CNN's error
is correlated in time and worse in tight corners. Use it to rank which channels
matter -- take the absolute numbers from a run through the real CNN.
"""

import torch

from deepracer_genesis.envs.features import PerceptionFeatures

CHANNEL_NAMES = ("lateral", "heading", "speed", "yaw_rate", "beta",
                 "curv@1m", "curv@3m")

# the CNN's typical error on each channel: root of the validation MSE
SIGMA = (0.125, 0.064, 0.060, 0.065, 0.083, 0.122, 0.224)


class NoisyPerceptionFeatures(PerceptionFeatures):
    """Params:

    noise: scales SIGMA (0 = perfect perception, 1 = the CNN's).
    noise_channels: names of the channels to corrupt; None = all of them.
    """

    def _sigma(self, device, dtype) -> torch.Tensor:
        chosen = self.params.get("noise_channels")
        s = [v if chosen is None or n in chosen else 0.0
             for n, v in zip(CHANNEL_NAMES, SIGMA)]
        return torch.tensor(s, device=device, dtype=dtype)

    def compute(self) -> torch.Tensor:
        x = super().compute()
        strength = float(self.params.get("noise", 0.0))
        if strength:
            lo, hi = self.cnn_target_slice
            sigma = self._sigma(x.device, x.dtype)[:hi - lo]
            x[:, lo:hi] += torch.randn_like(x[:, lo:hi]) * sigma * strength
        return x
