"""Reward functions: named, swappable, written in plain torch.

A reward function maps the env to a dict of NAMED PER-STEP TERMS — (N,)
CUDA tensors, one value per parallel car. Which terms count, and how much,
is the `reward_scales` dict (spec-level, sweepable); the weighted sum is the
step reward and every term is logged per episode (`Episode/rew_<name>`).

Write your own next to your experiment:

    from deepracer_genesis.envs.rewards import register_reward

    @register_reward("time_trial")
    def time_trial(env):
        return {
            "progress": env.d_progress,                       # meters this step
            "alive": torch.full_like(env.d_progress, -env.dt) # ticking clock
        }

    ... >> RewardShaping(fn="time_trial", scales={"progress": 10.0, "alive": 1.0})

Everything is batched torch on GPU — same speed as the built-in. Useful env
attributes (all (N,) tensors, driving-direction aware): d_progress, lateral,
half_width, heading_err, v_forward, v_lateral, yaw_rate, actions,
last_actions, dt, plus anything else on DeepRacerEnv.

NOTE the spec hashes the reward's NAME + scales, not the function body —
if you edit a registered function, rename it (or run(force=True)) so cached
runs don't shadow the change.
"""

from __future__ import annotations

import torch

REWARDS: dict[str, callable] = {}


def register_reward(name: str):
    """Decorator: make `name` usable in RewardShaping(fn=name)."""
    def deco(fn):
        if name in REWARDS:
            raise ValueError(f"reward fn {name!r} already registered")
        REWARDS[name] = fn
        return fn
    return deco


def resolve_reward(name: str):
    try:
        return REWARDS[name]
    except KeyError:
        raise ValueError(
            f"unknown reward fn {name!r}; registered: {sorted(REWARDS)} "
            "(custom rewards register via deepracer_genesis.envs.rewards"
            ".register_reward)") from None


@register_reward("deepracer")
def deepracer(env) -> dict[str, torch.Tensor]:
    """The default shaping: progress-dominated with stability terms."""
    on_track = env.lateral.abs() < (env.half_width - env.cfg["wheel_margin"])
    return {
        "progress": env.d_progress,
        "speed": env.v_forward.clamp(0.0, env.cfg["max_speed"]) * env.dt,
        "centered": torch.exp(-((env.lateral / env.half_width.clamp(min=0.1)) ** 2)) * env.dt,
        "heading": -env.heading_err.abs() * env.dt,
        "steering": -env.actions[:, 0].abs() * env.dt,
        "action_rate": -((env.actions - env.last_actions) ** 2).sum(dim=1) * env.dt,
        "off_track": (~on_track).float() * env.dt,
    }
