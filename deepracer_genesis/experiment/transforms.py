"""TorchRL transforms realizing the DR stages (plan section 1.3).

Contract per torchrl 0.13.2 (cheat-sheet section 6): observation transforms
override `_apply_transform` (+ the `_reset` boilerplate so out_keys exist
after reset); action transforms override `_inv_apply_transform` and run
before base_env.step; stateful per-sub-env buffers reset via the "_reset"
mask, with `_reset_on_native_autoreset` aliased for our autoreset flag.
"""

from __future__ import annotations

import math

import torch

from torchrl.envs.transforms import Transform
from torchrl.envs.transforms.utils import _get_reset, _set_missing_tolerance

# RGB <-> YIQ (NTSC) — hue rotation happens in the IQ plane
_RGB2YIQ = torch.tensor([[0.299, 0.587, 0.114],
                         [0.596, -0.274, -0.322],
                         [0.211, -0.523, 0.312]])
_YIQ2RGB = torch.tensor([[1.0, 0.956, 0.621],
                         [1.0, -0.272, -0.647],
                         [1.0, -1.106, 1.703]])


class ImageAug(Transform):
    """Per-step image-space DR on a float [0,1] (*B, C, H, W) key.

    All parameters resample every call (i.e. every env step), independently
    per sub-env. Config keys: brightness=(lo,hi) contrast=(lo,hi) hue=max_frac
    blur=max_sigma cutout=prob noise=sigma.
    """

    def __init__(self, aug: dict, in_keys=("camera",), out_keys=None):
        out_keys = list(out_keys or in_keys)
        super().__init__(in_keys=list(in_keys), out_keys=out_keys)
        self.aug = dict(aug)

    def _u(self, lo, hi, n, device):
        return lo + (hi - lo) * torch.rand(n, 1, 1, 1, device=device)

    def _apply_transform(self, img: torch.Tensor) -> torch.Tensor:
        lead = img.shape[:-3]
        c, h, w = img.shape[-3:]
        x = img.reshape(-1, c, h, w).clone()
        n, dev = x.shape[0], x.device
        a = self.aug

        if "brightness" in a:
            x = x * self._u(*a["brightness"], n, dev)
        if "contrast" in a:
            mean = x.mean(dim=(-3, -2, -1), keepdim=True)
            x = (x - mean) * self._u(*a["contrast"], n, dev) + mean
        if a.get("hue"):
            theta = (torch.rand(n, device=dev) * 2 - 1) * a["hue"] * 2 * math.pi
            cos, sin = torch.cos(theta), torch.sin(theta)
            rot = torch.zeros(n, 3, 3, device=dev)
            rot[:, 0, 0] = 1.0
            rot[:, 1, 1] = cos; rot[:, 1, 2] = -sin
            rot[:, 2, 1] = sin; rot[:, 2, 2] = cos
            m = _YIQ2RGB.to(dev) @ rot @ _RGB2YIQ.to(dev)          # (n,3,3)
            x = torch.einsum("nij,njhw->nihw", m, x)
        if a.get("blur"):
            sigma = float(torch.rand(()).item()) * a["blur"]
            if sigma > 0.05:
                k = self._gaussian_kernel(sigma, dev)
                pad = k.shape[-1] // 2
                blurred = torch.nn.functional.conv2d(
                    x, k.expand(c, 1, -1, -1), padding=pad, groups=c)
                mask = (torch.rand(n, 1, 1, 1, device=dev) < 0.5).float()
                x = mask * blurred + (1 - mask) * x
        if a.get("cutout"):
            active = torch.rand(n, device=dev) < a["cutout"]
            if active.any():
                x = self._cutout(x, active)
        if a.get("noise"):
            x = x + torch.randn_like(x) * a["noise"]

        return x.clamp(0.0, 1.0).reshape(*lead, c, h, w)

    @staticmethod
    def _gaussian_kernel(sigma: float, device) -> torch.Tensor:
        radius = max(1, int(2 * sigma))
        xs = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
        k1 = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
        k1 = k1 / k1.sum()
        return (k1[:, None] * k1[None, :])[None, None]

    @staticmethod
    def _cutout(x: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.shape
        dev = x.device
        ph = torch.randint(h // 6, h // 3 + 1, (n,), device=dev)
        pw = torch.randint(w // 6, w // 3 + 1, (n,), device=dev)
        cy = torch.randint(0, h, (n,), device=dev)
        cx = torch.randint(0, w, (n,), device=dev)
        ys = torch.arange(h, device=dev)[None, :, None]
        xs = torch.arange(w, device=dev)[None, None, :]
        inside = ((ys >= (cy - ph // 2)[:, None, None]) & (ys < (cy + ph // 2)[:, None, None])
                  & (xs >= (cx - pw // 2)[:, None, None]) & (xs < (cx + pw // 2)[:, None, None]))
        keep = ~(inside & active[:, None, None])
        return x * keep.unsqueeze(1)

    def _reset(self, tensordict, tensordict_reset):
        with _set_missing_tolerance(self, True):
            return self._call(tensordict_reset)

    _reset_on_native_autoreset = _reset


class FrozenEncoder(Transform):
    """Run a frozen module over an obs key, write a new key, drop the raw one.

    Modeled on torchrl's _R3MNet (the canonical frozen-representation
    transform); the raw camera key is deleted from the carried tensordict and
    the spec — the vector policy downstream never sees pixels, and the
    collector stops hauling (N,3,H,W) frames it doesn't need.
    """

    def __init__(self, encoder, embed_dim: int, in_keys=("camera",),
                 out_keys=("encoded",), del_keys: bool = True):
        super().__init__(in_keys=list(in_keys), out_keys=list(out_keys))
        encoder.eval()
        encoder.requires_grad_(False)
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.del_keys = del_keys

    @torch.no_grad()
    def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        lead = obs.shape[:-3]
        out = self.encoder(obs.reshape(-1, *obs.shape[-3:]))
        return out.reshape(*lead, self.embed_dim)

    def _call(self, next_tensordict):
        next_tensordict = super()._call(next_tensordict)
        if self.del_keys:
            next_tensordict = next_tensordict.exclude(*self.in_keys)
        return next_tensordict

    forward = _call

    def _reset(self, tensordict, tensordict_reset):
        with _set_missing_tolerance(self, True):
            return self._call(tensordict_reset)

    _reset_on_native_autoreset = _reset

    def transform_observation_spec(self, observation_spec):
        from torchrl.data import Unbounded
        observation_spec = observation_spec.clone()
        ref = observation_spec[self.in_keys[0]]
        lead = ref.shape[:-3]
        if self.del_keys:
            for k in self.in_keys:
                del observation_spec[k]
        for k in self.out_keys:
            observation_spec[k] = Unbounded(shape=(*lead, self.embed_dim),
                                            device=ref.device)
        return observation_spec


class ActionNoiseDelay(Transform):
    """Actuation DR: k-step command latency, then per-channel gaussian noise.

    Runs on the inverse (action) path before base_env.step. The delay ring
    buffer holds the last k commands per sub-env; freshly reset sub-envs
    start from a zeroed buffer (neutral steering, mid throttle).
    """

    def __init__(self, n_envs: int, steer_noise=0.0, speed_noise=0.0,
                 delay_steps=0, device="cuda"):
        super().__init__(in_keys_inv=["action"], out_keys_inv=["action"])
        self.steer_noise = steer_noise
        self.speed_noise = speed_noise
        self.delay_steps = int(delay_steps)
        if self.delay_steps > 0:
            self.register_buffer(
                "buf", torch.zeros(n_envs, self.delay_steps, 2, device=device))

    def _inv_apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        out = action
        if self.delay_steps > 0:
            out = self.buf[:, -1].clone()
            self.buf.copy_(torch.cat([action.unsqueeze(1), self.buf[:, :-1]], dim=1))
        noise = torch.stack([
            torch.randn(out.shape[0], device=out.device) * self.steer_noise,
            torch.randn(out.shape[0], device=out.device) * self.speed_noise,
        ], dim=1)
        return (out + noise).clamp(-1.0, 1.0)

    def _reset(self, tensordict, tensordict_reset):
        if self.delay_steps > 0:
            mask = _get_reset("_reset", tensordict).reshape(-1)
            self.buf[mask] = 0.0
        return tensordict_reset

    _reset_on_native_autoreset = _reset
