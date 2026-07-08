"""The Algorithm contract: how to plug your own training algorithm in.

HOW TO ADD A CUSTOM ALGORITHM (SAC, Dreamer-style world models, ...):

1. Write a class satisfying the `Algorithm` protocol below. The Trainer owns
   the outer loop (collector -> train -> log -> checkpoint -> eval); your
   class owns everything algorithm-specific:
     - `setup(builder)`: build networks/losses/optimizers from the Builder
       (which gives you `actor()`, `critic(out_key=...)`, `gae(...)`,
       `optimizer(...)`, obs-key dims, the sim, and the spec).
     - `collect_policy`: the (exploratory) policy module the Collector runs.
     - `eval_actor`: the module used for deterministic evaluation.
     - `train_on_batch(data)`: consume ONE collector yield of shape [N, T]
       (root obs/action + ("next", reward/done/terminated/truncated/obs)).
       Off-policy algorithms are free to stash it in their own replay buffer
       and take gradient steps at their own cadence; on-policy ones run their
       epochs/minibatches here. Return a dict of scalars to log.
     - `observe_env_logs(logs)`: the sim's episode logs each iteration
       (PPO-Lagrangian reads "Episode/cost" here to drive the PID lambda).
     - `checkpoint()`: state_dicts to persist alongside the Trainer payload.
2. Register it: `@register_algorithm("my_kind")`.
3. Select it from the DSL with the terminal stage `Algo(kind="my_kind",
   params={...})` — or add a dedicated Stage subclass for nicer authoring.

The declaration layer never imports this module; unknown kinds fail at
Builder time with the list of registered names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:  # only for annotations; keep import-time light
    from tensordict import TensorDictBase

    from .builder import Builder

ALGORITHMS: dict[str, type] = {}


def register_algorithm(kind: str):
    """Class decorator: make `kind` selectable from AlgorithmSpec.kind.

    Args:
        kind: Registry key, referenced from the DSL via
            `Algo(kind="my_kind", ...)`.

    Returns:
        The decorator; it registers and returns the class unchanged.

    Raises:
        ValueError: If `kind` is already registered (raised at decoration
            time).
    """
    def deco(cls: type) -> type:
        if kind in ALGORITHMS:
            raise ValueError(f"algorithm kind {kind!r} already registered")
        ALGORITHMS[kind] = cls
        return cls
    return deco


def make_algorithm(builder: "Builder") -> "Algorithm":
    """Resolve AlgorithmSpec.kind against the registry and set it up.

    Args:
        builder: The Builder whose spec selects the algorithm kind; passed
            through to the instance's setup().

    Returns:
        A ready (setup-complete) Algorithm instance.

    Raises:
        ValueError: If the kind is not registered; the message lists the
            registered names.
    """
    kind = builder.spec.algorithm.kind
    try:
        cls = ALGORITHMS[kind]
    except KeyError:
        raise ValueError(
            f"unknown algorithm kind {kind!r}; registered: {sorted(ALGORITHMS)} "
            "(custom algorithms register via "
            "deepracer_genesis.algorithms.register_algorithm)") from None
    algo = cls()
    algo.setup(builder)
    return algo


@runtime_checkable
class Algorithm(Protocol):
    """What the Trainer drives; see the module docstring for the guide."""

    def setup(self, builder: "Builder") -> None:
        """Build networks, losses and optimizers from the Builder.

        Args:
            builder: Gives you `actor()`, `critic(out_key=...)`, `gae(...)`,
                `optimizer(...)`, obs-key dims, the sim, and the spec.
        """

    @property
    def collect_policy(self):
        """Policy module the Collector runs (exploratory)."""

    @property
    def eval_actor(self):
        """Actor used for deterministic evaluation rollouts."""

    def train_on_batch(self, data: "TensorDictBase") -> dict[str, float]:
        """Consume one collector yield and take training steps.

        Args:
            data: One [N, T] collector batch (root obs/action + ("next",
                reward/done/terminated/truncated/obs)). Off-policy
                algorithms are free to stash it in their own replay buffer
                and take gradient steps at their own cadence; on-policy ones
                run their epochs/minibatches here.

        Returns:
            Dict of scalars to log.
        """

    def observe_env_logs(self, logs: dict[str, Any]) -> None:
        """Receive the sim's episode logs once per iteration (optional use).

        Args:
            logs: The sim's episode-log dict (PPO-Lagrangian reads
                "Episode/cost" here to drive the PID lambda).
        """

    def checkpoint(self) -> dict[str, Any]:
        """Extra state to persist in the run checkpoint.

        Returns:
            Mapping of name -> state_dict, saved alongside the Trainer
            payload.
        """


