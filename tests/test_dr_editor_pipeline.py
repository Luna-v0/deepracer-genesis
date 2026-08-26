"""dr_editor pipeline-replay tests on synthetic CPU frames (no genesis).

Pins the offline replay against the ops the env actually runs: stage coverage
and determinism of :func:`replay_stages`, GEMM-level exactness and stage ORDER
of the world-colour remap (camera-res colour before the policy-res downscale —
the two orders provably differ), the fresh-episode frame-stack contract, the
stateful latency/frame-drop semantics of :func:`replay_temporal`, and the
spec -> DR-dict extraction of :func:`dr_from_spec`.
"""

import warnings

import pytest
import torch
import torch.nn.functional as F

from deepracer_genesis.experiment import (
    PPO,
    AsymmetricCameraPolicy,
    CameraEnvironment,
    DomainRandomizationCamera,
    DomainRandomizationTrackAppearance,
)
from deepracer_genesis.randomization.visual import sample_world_color
from deepracer_genesis.tools.dr_editor.pipeline import (
    STAGES,
    apply_world_color,
    dr_from_spec,
    replay_stages,
    replay_temporal,
)
from deepracer_genesis.tools.dr_editor.rng import seeded

DEVICE = torch.device("cpu")


def _raw(n: int = 2, h: int = 12, w: int = 16, seed: int = 0) -> torch.Tensor:
    """Deterministic synthetic raw frames.

    Args:
        n: Batch (env) count.
        h: Frame height in pixels.
        w: Frame width in pixels.
        seed: Generator seed so every test sees the same frames.

    Returns:
        An ``(n, 3, h, w)`` float tensor uniform in [0, 1] on CPU.
    """
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, h, w, generator=g)


FULL_DR = {"world_color": 0.5, "pixel_noise": 0.03,
           "image_aug": {"brightness": (0.7, 1.3), "noise": 0.02}}


# ------------------------------------------- 1. stages, determinism, no-op

def test_replay_stages_returns_every_stage():
    out = replay_stages(_raw(), FULL_DR, frame_stack=2, seed=0)
    assert tuple(out) == STAGES
    for name in STAGES[:-1]:
        assert out[name].shape == (2, 3, 12, 16), name
    assert out["stack"].shape == (2, 6, 12, 16)


def test_replay_stages_deterministic_in_seed():
    raw = _raw()
    a = replay_stages(raw, FULL_DR, seed=7)
    b = replay_stages(raw, FULL_DR, seed=7)
    for name in STAGES:
        assert torch.equal(a[name], b[name]), name
    c = replay_stages(raw, FULL_DR, seed=8)
    assert not torch.equal(a["world_color"], c["world_color"])
    assert not torch.equal(a["pixel_noise"], c["pixel_noise"])


def test_raw_stage_is_the_input():
    raw = _raw()
    out = replay_stages(raw, FULL_DR, seed=0)
    assert out["raw"] is raw


def test_empty_dr_is_identity_except_stack_tiling():
    raw = _raw()
    out = replay_stages(raw, {}, frame_stack=4, seed=0)
    for name in STAGES[:-1]:
        assert torch.equal(out[name], raw), name
    assert torch.equal(out["stack"], raw.repeat(1, 4, 1, 1))


# ------------------------------- 2. world-colour exactness + stage order

def test_world_color_matches_manual_gemm():
    raw = _raw()
    n, c, h, w = raw.shape
    strength = 0.6
    with seeded(11, DEVICE):
        color = sample_world_color(n, strength, DEVICE)
    out = apply_world_color(raw, strength, color=color)

    mat, bias = color
    flat = raw.permute(0, 2, 3, 1).reshape(n, h * w, c)
    manual = ((flat @ mat.transpose(1, 2) + bias).clamp(0.0, 1.0)
              .reshape(n, h, w, c).permute(0, 3, 1, 2))
    assert torch.equal(out, manual)

    # replay_stages under the same seed samples the same palette
    staged = replay_stages(raw, {"world_color": strength}, seed=11)
    assert torch.equal(staged["world_color"], out)


def test_world_color_applied_before_policy_res_downscale():
    raw = _raw(2, 12, 16)                     # camera res 16x12 (W x H)
    strength = 0.6
    # strong colour whose clamp bites on many pixels: mean(clamp(x)) is then
    # provably != clamp(mean(x)), so the two stage orders cannot agree
    mat = (torch.eye(3) * 1.6).expand(2, 3, 3)
    bias = torch.full((2, 1, 3), -0.15)
    color = (mat, bias)

    out = replay_stages(raw, {"world_color": strength},
                        policy_res=(8, 6), color=color)
    assert out["policy_res"].shape == (2, 3, 6, 8)

    color_first = F.interpolate(apply_world_color(raw, strength, color=color),
                                size=(6, 8), mode="area")
    downscale_first = apply_world_color(
        F.interpolate(raw, size=(6, 8), mode="area"), strength, color=color)
    assert not torch.allclose(color_first, downscale_first)  # order matters
    assert torch.equal(out["policy_res"], color_first)       # replay = env order


# ---------------------------------------- 3. fresh-episode stack contract

def test_stack_repeats_the_frame_oldest_first():
    raw = _raw()
    out = replay_stages(raw, {"image_aug": {"brightness": (0.8, 0.8)}},
                        frame_stack=4, seed=3)
    stack = out["stack"]
    assert stack.shape == (2, 12, 12, 16)
    for i in range(4):                         # every 3-channel group == newest
        assert torch.equal(stack[:, 3 * i:3 * i + 3], out["image_aug"]), i


# ------------------------------------------------- 4. temporal semantics

def test_latency_delays_and_primes_with_first_frame():
    frames = [torch.full((2, 3, 4, 4), t / 10.0) for t in range(6)]
    for k in (1, 2):
        seen = replay_temporal(frames, {"image_aug": {"latency_steps": k}})
        assert len(seen) == len(frames)
        for t in range(len(frames)):
            expected = frames[t - k] if t >= k else frames[0]
            assert torch.equal(seen[t], expected), (k, t)


def test_full_frame_drop_repeats_previous_emitted_frame():
    frames = [torch.full((2, 3, 4, 4), t / 10.0) for t in range(6)]
    seen = replay_temporal(frames, {"image_aug": {"frame_drop": 1.0}}, seed=0)
    assert torch.equal(seen[0], frames[0])
    for t in range(1, len(frames)):
        assert torch.equal(seen[t], seen[t - 1]), t


def test_temporal_deterministic_in_seed():
    frames = [torch.full((4, 3, 4, 4), t / 10.0) for t in range(8)]
    dr = {"image_aug": {"latency_steps": 1, "frame_drop": 0.5}}
    a = replay_temporal(frames, dr, seed=5)
    b = replay_temporal(frames, dr, seed=5)
    for t, (fa, fb) in enumerate(zip(a, b)):
        assert torch.equal(fa, fb), t


def test_temporal_noop_without_latency_or_drop():
    frames = [torch.full((1, 3, 2, 2), t / 10.0) for t in range(3)]
    seen = replay_temporal(frames, {})
    assert all(torch.equal(s, f) for s, f in zip(seen, frames))


# ----------------------------------------------------- 5. dr_from_spec

def test_dr_from_spec_extracts_editor_dict():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = (CameraEnvironment()
                >> DomainRandomizationCamera(brightness=(0.7, 1.3),
                                             latency_steps=1)
                >> DomainRandomizationTrackAppearance(strength=0.4)
                >> AsymmetricCameraPolicy()
                >> PPO()).build()
    dr = dr_from_spec(spec)
    assert set(dr) == {"image_aug", "world_color", "pixel_noise", "env_map",
                       "camera_jitter", "physics", "action_dr"}
    assert dr["image_aug"] == {"brightness": (0.7, 1.3), "latency_steps": 1}
    assert dr["world_color"] == pytest.approx(0.4)
    assert dr["pixel_noise"] == 0.0
    assert dr["env_map"] == {}
    assert dr["camera_jitter"] == {}
    assert dr["physics"] == {}
    assert dr["action_dr"] == {"steer_noise": 0.0, "speed_noise": 0.0,
                               "delay_steps": 0}
