"""Define the MDP interface over the raw simulator: reward (R) and termination (T).

Each function reads the live env's per-step attributes and writes its per-env buffers.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from . import rules

if TYPE_CHECKING:
    from .deepracer_env import DeepRacerEnv


def map_action(env: "DeepRacerEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """Map the normalized ``[-1, 1]`` action to physical steering and speed.

    Reads the per-experiment action caps from ``env.cfg["action"]`` (whose
    defaults are sourced from :mod:`~deepracer_genesis.physics.limits`), so an
    experiment can widen or narrow the control envelope without touching the
    immutable limits. The Ackermann/parallel split of the returned center angle
    happens downstream in :meth:`~deepracer_genesis.envs.entities.Car.drive`.

    Args:
        env: The live env; reads ``actions`` ``(N, 2)`` in ``[-1, 1]`` and the
            ``max_steering_deg`` / ``min_speed`` / ``max_speed`` action caps.

    Returns:
        ``(steer, speed)``, each ``(N, 1)``: the commanded center steering angle
        in radians (positive = left) and the forward speed in m/s.
    """
    act = env.cfg["action"]
    steer = env.actions[:, 0:1] * math.radians(act["max_steering_deg"])
    speed = act["min_speed"] + (env.actions[:, 1:2] + 1) * 0.5 * (
        act["max_speed"] - act["min_speed"])
    return steer, speed


def _reward_verbatim(env: "DeepRacerEnv", r: torch.Tensor) -> None:
    """Write ``r`` into ``rew_buf`` as the whole step reward (logged as ``total``)."""
    r = r.reshape(env.rew_buf.shape)
    env.rew_buf.copy_(r)
    env.episode_sums.setdefault("total", torch.zeros_like(env.rew_buf)).add_(r)


def compute_reward(env: "DeepRacerEnv") -> None:
    """Reduce the reward fn's output into the env's reward buffer.

    A bare (N,) tensor (or a ``total`` term with no scales) is the reward
    verbatim; named terms are summed as ``scale * terms[name]``. Both paths
    accumulate ``episode_sums``; ``_``-prefixed terms are logged, never summed.

    Args:
        env: The live DeepRacer env whose ``reward_terms`` callable, per-term
            ``reward_scales``, ``rew_buf``, and ``episode_sums`` are read/written.

    Raises:
        ValueError: If a verbatim reward is mixed with ``reward_scales``, a
            scale names a ``_`` diagnostic term, or named terms have no scales.
        KeyError: If ``reward_scales`` references a term name the reward fn did
            not produce.
    """
    terms = env.reward_terms(env)          # named per-step terms (rewards.py)
    if isinstance(terms, torch.Tensor):
        if env.reward_scales:
            raise ValueError(
                "reward fn returned a bare tensor (the reward itself) but "
                f"reward_scales {sorted(env.reward_scales)} are set — they "
                "would be silently ignored; drop the scales or return named terms")
        _reward_verbatim(env, terms)
        return
    if env.reward_scales:
        bad = sorted(k for k in env.reward_scales if k.startswith("_"))
        if bad:
            raise ValueError(
                f"reward_scales references diagnostic term(s) {bad}: "
                "'_'-prefixed names are logged but never summed into the reward")
        env.rew_buf.zero_()
        for name, scale in env.reward_scales.items():
            try:
                r = terms[name] * scale
            except KeyError:
                raise KeyError(
                    f"reward_scales references term {name!r} but the reward fn "
                    f"produced {sorted(terms)}") from None
            env.rew_buf += r
            env.episode_sums[name] += r
    elif "total" in terms:
        stray = sorted(k for k in terms if k != "total" and not k.startswith("_"))
        if stray:
            raise ValueError(
                f"reward fn returned unscaled term(s) {stray} beside 'total': "
                "prefix them with '_' (diagnostic-only) or provide scales")
        _reward_verbatim(env, terms["total"])
    else:
        name = getattr(env.reward_terms, "__qualname__", env.reward_terms)
        raise ValueError(
            f"custom reward fn {name!r} returned named terms but no scales are "
            "set: pass RewardShaping(fn=..., scales={...}), or return the "
            "reward itself (a bare (N,) tensor, or a 'total' term)")
    for name, value in terms.items():
        if name.startswith("_"):
            env.episode_sums.setdefault(
                name, torch.zeros_like(env.rew_buf)).add_(value)


def check_termination(env: "DeepRacerEnv") -> None:
    """Set the termination (and, under the CMDP framing, cost) buffers for the step.

    Evaluates the off-track and flipped predicates and writes the per-env termination buffers.

    Args:
        env: The live DeepRacer env; reads ``lateral``, ``half_width``, ``up_z``,
            ``v_forward``, ``episode_length_buf``, ``max_episode_length``,
            ``emit_cost``, ``cost_fn``, and ``cfg``, and writes the termination
            (and cost) buffers listed above.
    """
    cfg = env.cfg
    off = rules.is_off_track(env.lateral, env.half_width, cfg["termination"]["off_track_margin"])
    flipped = rules.is_flipped(env.up_z)
    env.flipped_buf = flipped
    env.time_out_buf = env.episode_length_buf >= env.max_episode_length
    if env.emit_cost:
        # CMDP framing: offtrack is a COST, not a termination — declare
        # "violate at most `budget`" instead of hand-tuning a penalty.
        # Only unrecoverable states terminate (flip, or far off the road).
        hard_off = rules.is_off_track(env.lateral, env.half_width,
                                      cfg["termination"]["off_track_margin"] + 0.4)
        env.offtrack_buf = hard_off
        env.reset_buf = hard_off | flipped | env.time_out_buf
        cost = off.float() + flipped.float()
        if env.cost_fn == "offtrack_or_overspeed":
            cost += (env.v_forward > cfg["termination"].get("overspeed_limit", 3.5)).float()
        elif env.cost_fn == "crash":
            cost = flipped.float() + hard_off.float()
        env.cost_buf = cost
        env.cost_episode_sum += cost
    else:
        env.offtrack_buf = off
        env.reset_buf = off | flipped | env.time_out_buf
        # terminal penalty for genuine failures (not timeouts)
        env.rew_buf += (off | flipped).float() * cfg["termination"]["crash_penalty"]

    # A physics blow-up yields NaN/Inf (or absurdly large) state. NaN
    # comparisons are all False, so such an env passes every predicate above
    # and would poison training data until its timeout — terminate it now (its
    # reward too, so nothing non-finite reaches the learner) and let reset_idx
    # respawn it clean. Huge-but-finite states are included: they overflow to
    # inf in the loss backward pass before any NaN ever appears.
    broken = (~torch.isfinite(env.state_buf).all(dim=-1)
              | (env.state_buf.abs() > 1e6).any(dim=-1))
    if broken.any():
        env.reset_buf |= broken
        env.rew_buf = torch.where(broken, torch.zeros_like(env.rew_buf), env.rew_buf)
        first = not hasattr(env, "_nonfinite_resets")
        env._nonfinite_resets = getattr(env, "_nonfinite_resets", 0) + int(broken.sum())
        env._nonfinite_fires = getattr(env, "_nonfinite_fires", 0) + 1
        if first or env._nonfinite_fires % 200 == 0:
            print(f"[mdp] non-finite/huge state in {int(broken.sum())} env(s) -> "
                  f"forced reset (total {env._nonfinite_resets}, "
                  f"fires {env._nonfinite_fires})", flush=True)
        if first:
            _dump_body_fields(env)


def _dump_body_fields(env: "DeepRacerEnv") -> None:
    """One-shot diagnostic on the first blow-up: report whether PERSISTENT body
    fields (which no reset ever rewrites) have been corrupted in place — the
    signature of the quadrants 1.0.2 allocator/zerocopy memory-safety bugs.
    """
    e = env.car.entity
    probes = {
        "dofs_kp": lambda: e.get_dofs_kp(),
        "dofs_kv": lambda: e.get_dofs_kv(),
        "links_inertial_mass": lambda: e.get_links_inertial_mass(),
        "dofs_armature": lambda: e.get_dofs_armature(),
        "qpos": lambda: e.get_qpos(),
        "dofs_velocity": lambda: e.get_dofs_velocity(),
    }
    for name, fn in probes.items():
        try:
            t = torch.as_tensor(fn())
            n_bad = int((~torch.isfinite(t)).sum())
            big = float(t.abs().max())
            print(f"[mdp:first-blowup] {name}: "
                  f"{'CORRUPTED' if n_bad or big > 1e8 else 'ok'} "
                  f"(non-finite {n_bad}, max|x| {big:.3g})", flush=True)
        except Exception as exc:  # diagnostic only — never break the step
            print(f"[mdp:first-blowup] {name}: probe failed ({exc})", flush=True)
