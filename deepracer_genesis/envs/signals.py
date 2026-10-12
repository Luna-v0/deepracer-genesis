"""The signal bus: a lazy, typed, tagged namespace of world quantities (Part K.1).

One shared pool of independently-computed signals that every consumer — the
feature vector, the reward, and (later) the cost — reads from. Each signal is a
small function with metadata; nothing is a god object and nothing is eagerly
computed:

- **Lazy + per-step cache**: a signal is computed on first read and cached until
  the next ``_post_physics`` calls :meth:`SignalBus.invalidate`. A run that never
  reads ``curvature_ahead`` never pays for it.
- **Signals are raw; the driving-frame transform lives in the consumer.** Each
  signal returns exactly the value the env computed in ``_post_physics`` — no
  recomputation, no parallel copy — and its ``frame`` metadata states honestly
  whether that value is already ``dir_sign``-corrected. ``heading_err`` and
  ``d_progress`` genuinely *are* driving-frame (the env applies ``dir_sign`` when
  it computes them, see ``base_env._post_physics``), so their ``frame`` is
  ``"driving"``. ``lateral`` is the *raw* signed offset from the centerline —
  the env stores it un-corrected — so its ``frame`` is ``"raw"`` and any
  consumer that wants the driving-frame value multiplies by ``dir_sign`` itself
  (as the ``lateral`` feature block does). Marking ``lateral`` as anything but
  ``"raw"`` would be a lie: the ``frame`` field is per-signal truth, not a
  blanket claim that every signal is driving-frame.

This module is the K.1 foundation. K.2 (features as signal selections), K.3
(config-routed actor/critic keys), and K.4 (reward+cost as signal combinations)
consume this bus.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from . import rules

if TYPE_CHECKING:
    from .base_env import DeepRacerEnv


@dataclass(frozen=True)
class Signal:
    """A named world quantity plus the metadata consumers reason about.

    Attributes:
        name: Unique signal name.
        compute: ``env -> (N,)`` or ``(N, d)`` tensor for the current step.
        pixel_observable: Whether a CNN could infer it from a single frame
            (used by the K.5 learnability check).
        cost: ``"cheap"`` (kinematics/track-frame read) vs ``"heavy"``
            (lookahead/curvature geometry) — informs lazy evaluation value.
        frame: ``"driving"`` (already dir_sign-corrected) vs ``"raw"``.
    """

    name: str
    compute: Callable[[DeepRacerEnv], torch.Tensor]
    pixel_observable: bool
    cost: Literal["cheap", "heavy"] = "cheap"
    frame: Literal["raw", "driving"] = "driving"


class SignalBus:
    """Per-step lazy cache over a registry of :class:`Signal` definitions.

    Holds a reference to the live env; reads compute-on-first-use and cache
    until :meth:`invalidate`. It does not own state — it reads the same env
    attributes ``_post_physics`` and termination read (single source of truth).

    Attributes:
        registry: The name -> Signal table this bus resolves against.
    """

    def __init__(self, env: DeepRacerEnv,
                 registry: dict[str, Signal] | None = None) -> None:
        self._env = env
        self.registry: dict[str, Signal] = registry if registry is not None else SIGNALS
        self._cache: dict[str, torch.Tensor] = {}

    def invalidate(self) -> None:
        """Drop the per-step cache (call once per ``_post_physics``)."""
        self._cache.clear()

    def __getitem__(self, name: str) -> torch.Tensor:
        """Return the signal value, computing and caching on first read.

        Raises:
            KeyError: If ``name`` is not a registered signal.
        """
        cached = self._cache.get(name)
        if cached is None:
            try:
                sig = self.registry[name]
            except KeyError:
                raise KeyError(
                    f"unknown signal {name!r}; registered: {sorted(self.registry)}"
                ) from None
            cached = sig.compute(self._env)
            self._cache[name] = cached
        return cached

    def get(self, name: str) -> torch.Tensor:
        """Alias for ``bus[name]``."""
        return self[name]

    def meta(self, name: str) -> Signal:
        """Return the :class:`Signal` metadata for ``name``."""
        return self.registry[name]

    def names(self) -> tuple[str, ...]:
        """All registered signal names."""
        return tuple(self.registry)


# ---------------------------------------------------------------------------
# Signal definitions — thin, driving-frame-correct reads of the env state the
# env already computed in _post_physics (single source; no recomputation).

def _off_track(env: DeepRacerEnv) -> torch.Tensor:
    margin = env.cfg["termination"].get("wheel_margin", 0.0)
    return rules.is_off_track(env.lateral, env.half_width, margin).float()


def _d_actions(env: DeepRacerEnv) -> torch.Tensor:
    return ((env.actions - env.last_actions) ** 2).sum(dim=1)


def _curvature_ahead(env: DeepRacerEnv) -> torch.Tensor:
    # heavy: lookahead geometry. Prefer a value the env already cached this
    # step; otherwise compute a single-step-ahead curvature at the car's wp.
    cached = getattr(env, "curvature_ahead", None)
    if cached is not None:
        return cached
    return env.track.curvature[env.track._ev, env.wp_idx]


# frame is per-signal truth (see module docstring): "driving" only for signals
# the env already dir_sign-corrected (heading_err, d_progress, curvature_ahead);
# everything else is "raw" — the consumer applies dir_sign if it wants the
# driving frame (the lateral feature block does exactly this).
SIGNALS: dict[str, Signal] = {
    # cheap kinematics (velocities aren't inferable from one frame; frame-invariant)
    "v_forward": Signal("v_forward", lambda e: e.v_forward, pixel_observable=False, frame="raw"),
    "v_lateral": Signal("v_lateral", lambda e: e.v_lateral, pixel_observable=False, frame="raw"),
    "yaw_rate": Signal("yaw_rate", lambda e: e.yaw_rate, pixel_observable=False, frame="raw"),
    "up_z": Signal("up_z", lambda e: e.up_z, pixel_observable=False, frame="raw"),
    # track-frame geometry (visible in a single frame)
    #   lateral: RAW signed offset — env does NOT dir_sign-correct it; the
    #   feature layer multiplies by dir_sign itself.
    "lateral": Signal("lateral", lambda e: e.lateral, pixel_observable=True, frame="raw"),
    "half_width": Signal("half_width", lambda e: e.half_width, pixel_observable=True, frame="raw"),
    #   heading_err: env applies dir_sign in _post_physics -> genuinely driving.
    "heading_err": Signal("heading_err", lambda e: e.heading_err, pixel_observable=True,
                          frame="driving"),
    #   off_track uses |lateral|, so it is frame-invariant; label raw (uncorrected input).
    "off_track": Signal("off_track", _off_track, pixel_observable=True, frame="raw"),
    # progress is NOT recoverable from a single frame (the CNN-can't-see-progress case);
    # env dir_sign-corrects d_progress -> driving.
    "d_progress": Signal("d_progress", lambda e: e.d_progress, pixel_observable=False,
                         frame="driving"),
    # commands
    "actions": Signal("actions", lambda e: e.actions, pixel_observable=False, frame="raw"),
    "action_rate": Signal("action_rate", _d_actions, pixel_observable=False, frame="raw"),
    # heavy lookahead geometry (only paid for when read); fronts a dir_sign-aware value.
    "curvature_ahead": Signal("curvature_ahead", _curvature_ahead,
                              pixel_observable=True, cost="heavy", frame="driving"),
}


# ---------------------------------------------------------------------------
# K.5 build-time learnability check (pure; fed by K.2/K.3 once they land).

def check_learnability(reward_reads: set[str], cost_reads: set[str],
                       actor_signals: set[str], critic_signals: set[str],
                       registry: dict[str, Signal] | None = None) -> list[str]:
    """Statically flag reward/cost terms that cannot be learned.

    A reward/cost is learnable only if its inputs are recoverable from what the
    learner conditions on. The invariant is
    ``reward.reads ∪ cost.reads ⊆ critic-visible signals`` (value learning gets
    full signal); separately, a term leaning on a ``pixel_observable=False``
    signal that the actor is pixel-only and does not carry is a soft warning
    (the policy can't act on it even if the critic can score it).

    Args:
        reward_reads: Signal names the reward function reads.
        cost_reads: Signal names the cost function reads (empty in plain RL).
        actor_signals: Signals the actor observes (``{"*"}`` means "all").
        critic_signals: Signals the critic observes (``{"*"}`` means "all").
        registry: Signal table for metadata; defaults to :data:`SIGNALS`.

    Returns:
        A list of human-readable problem strings (empty if everything is
        learnable). Callers decide whether to warn or raise.
    """
    reg = registry if registry is not None else SIGNALS
    problems: list[str] = []
    all_names = set(reg)
    critic = all_names if "*" in critic_signals else set(critic_signals)
    actor = all_names if "*" in actor_signals else set(actor_signals)

    needed = set(reward_reads) | set(cost_reads)
    missing_from_critic = needed - critic
    if missing_from_critic:
        problems.append(
            "unlearnable: reward/cost read signal(s) "
            f"{sorted(missing_from_critic)} not visible to the critic")

    # actor can only act on non-pixel-observable signals it directly carries
    for name in sorted(needed & critic):
        sig = reg.get(name)
        if sig is None:
            continue
        if not sig.pixel_observable and name not in actor:
            problems.append(
                f"warning: reward/cost leans on non-pixel-observable signal "
                f"{name!r} that the actor does not carry (critic-only); the "
                "policy cannot act on it directly")
    return problems
