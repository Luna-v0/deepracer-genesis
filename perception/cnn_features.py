"""The 7 camera channels produced by the real CNN instead of the simulator.

The env renders the image, the frozen CNN reads it, the policy receives its
estimates. The remaining 22 channels (past actions, command deltas) stay
computed onboard, as they are on the car.
"""

from pathlib import Path

import torch

from deepracer_genesis.envs.features import PerceptionFeatures

from perception.model import PerceptionCNN

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "perception.pt"


class CNNPerceptionFeatures(PerceptionFeatures):
    """Params: ``checkpoint`` (path to the .pt), else perception/perception.pt."""

    def __init__(self, env, params: dict):
        super().__init__(env, params)
        checkpoint = params.get("checkpoint") or DEFAULT_CHECKPOINT
        # mac: the camera env runs on CPU (no Madrona) but the CNN gains a lot from
        # MPS, and it is what dominates the cost of a step.
        self.dev = ("mps" if params.get("cnn_device", "mps") == "mps"
                    and torch.backends.mps.is_available() else env.device)
        self.net = PerceptionCNN().to(self.dev).eval()
        self.net.load_state_dict(torch.load(checkpoint, map_location=self.dev))
        for p in self.net.parameters():
            p.requires_grad_(False)

    def compute(self) -> torch.Tensor:
        x = super().compute()
        stack = self.env._stack_buf     # (N, 12, H, W) in [0, 1], oldest first
        if stack is not None:
            lo, hi = self.cnn_target_slice
            with torch.inference_mode():
                y = self.net(stack.to(self.dev))
            x[:, lo:hi] = y.to(x.device)
        return x
