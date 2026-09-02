"""Swappable reward functions, written in plain torch and passed as parameters.

Each fn maps the env to named per-step (N,) terms weighted by ``reward_scales``.

Reward + cost as combinations over the shared signal vocabulary (Part K.4): a
reward fn declares which :mod:`~deepracer_genesis.envs.signals` names it reads,
so the build-time learnability check (K.5) can verify the critic can see them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    from .base_env import DeepRacerEnv

#: a reward fn maps the env to ``{term_name: (N,) tensor}``
RewardFn = Callable[["DeepRacerEnv"], "dict[str, torch.Tensor]"]


def reads(*signals: str) -> "Callable[[RewardFn], RewardFn]":
    """Declare the signal names a reward fn reads (Part K.4).

    Attaches a ``reads`` frozenset to the fn so the K.5 learnability check can
    statically verify ``reward.reads ⊆ critic-visible signals``. Purely
    declarative metadata — it does not change what the fn computes.

    Args:
        *signals: registered signal names (see :data:`envs.signals.SIGNALS`).

    Returns:
        A decorator that sets ``fn.reads`` and returns the fn unchanged.
    """
    frozen = frozenset(signals)

    def decorate(fn: "RewardFn") -> "RewardFn":
        fn.reads = frozen  # type: ignore[attr-defined]
        return fn

    return decorate


def reward_reads(fn: "RewardFn") -> "frozenset[str]":
    """Return the signal names ``fn`` declared it reads (empty if undeclared).

    An undeclared custom reward returns ``frozenset()``, which the K.5 check
    treats as "nothing to verify" (no false-positive warnings).
    """
    return getattr(fn, "reads", frozenset())


#: signal names each cost fn reads (Part K.4; ``cost_fn`` name -> reads). The
#: env's default cost is the plain ``offtrack`` term (see mdp.check_termination).
COST_READS: "dict[str, frozenset[str]]" = {
    "offtrack": frozenset({"off_track", "up_z"}),
    "offtrack_or_overspeed": frozenset({"off_track", "up_z", "v_forward"}),
    "crash": frozenset({"off_track", "up_z"}),
}


def cost_reads(cost_fn: "str | None") -> "frozenset[str]":
    """Return the signal names the named cost fn reads (empty if none/unknown)."""
    return COST_READS.get(cost_fn or "", frozenset())


# up_z is read by the flip terminal penalty the DEFAULT (non-cost) path adds in
# mdp.check_termination — declared here so the K.5 trace covers the effective
# reward, not just the terms this fn returns (the CMDP path models it via COST_READS).
@reads("d_progress", "v_forward", "lateral", "half_width", "heading_err",
       "actions", "action_rate", "off_track", "up_z")
def deepracer(env: "DeepRacerEnv") -> dict[str, torch.Tensor]:
    """Compute the built-in DeepRacer shaping: progress-dominated with stability.

    Args:
        env: The live DeepRacerEnv, providing the driving-direction-aware (N,)
            per-car state tensors (lateral, half_width, d_progress, v_forward,
            heading_err, actions, last_actions, dt) and the ``cfg`` dict
            (wheel_margin, max_speed) read here.

    Returns:
        A mapping from term name to its (N,) per-car reward tensor: progress,
        speed, centered, heading, steering, action_rate, and off_track. The
        weighted sum under the spec's ``reward_scales`` is the step reward.

    Note:
        Declares ``reads`` = {d_progress, v_forward, lateral, half_width,
        heading_err, actions, action_rate, off_track, up_z} (Part K.4) so the
        K.5 build-time check can verify these signals are critic-visible.
        ``up_z`` is not a term this fn returns — it is read by the flip terminal
        penalty ``mdp.check_termination`` adds on the default (non-cost) path.
    """
    on_track = env.lateral.abs() < (env.half_width - env.cfg["termination"]["wheel_margin"])
    return {
        "progress": env.d_progress,
        "speed": env.v_forward.clamp(0.0, env.cfg["action"]["max_speed"]) * env.dt,
        "centered": torch.exp(-((env.lateral / env.half_width.clamp(min=0.1)) ** 2)) * env.dt,
        "heading": -env.heading_err.abs() * env.dt,
        "steering": -env.actions[:, 0].abs() * env.dt,
        "action_rate": -((env.actions - env.last_actions) ** 2).sum(dim=1) * env.dt,
        "off_track": - (~on_track).float() * env.dt,
    }
