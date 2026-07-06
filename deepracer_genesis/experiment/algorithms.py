"""The Algorithm contract + the two shipped implementations.

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
    """Class decorator: make `kind` selectable from AlgorithmSpec.kind."""
    def deco(cls: type) -> type:
        if kind in ALGORITHMS:
            raise ValueError(f"algorithm kind {kind!r} already registered")
        ALGORITHMS[kind] = cls
        return cls
    return deco


def make_algorithm(builder: "Builder") -> "Algorithm":
    """Resolve AlgorithmSpec.kind against the registry and set it up."""
    kind = builder.spec.algorithm.kind
    try:
        cls = ALGORITHMS[kind]
    except KeyError:
        raise ValueError(
            f"unknown algorithm kind {kind!r}; registered: {sorted(ALGORITHMS)} "
            "(custom algorithms register via "
            "deepracer_genesis.experiment.algorithms.register_algorithm)") from None
    algo = cls()
    algo.setup(builder)
    return algo


@runtime_checkable
class Algorithm(Protocol):
    """What the Trainer drives; see the module docstring for the guide."""

    def setup(self, builder: "Builder") -> None:
        """Build networks, losses and optimizers from the Builder."""

    @property
    def collect_policy(self):
        """Policy module the Collector runs (exploratory)."""

    @property
    def eval_actor(self):
        """Actor used for deterministic evaluation rollouts."""

    def train_on_batch(self, data: "TensorDictBase") -> dict[str, float]:
        """Consume one [N, T] collector yield; return scalars to log."""

    def observe_env_logs(self, logs: dict[str, Any]) -> None:
        """Receive the sim's episode logs once per iteration (optional use)."""

    def checkpoint(self) -> dict[str, Any]:
        """Extra state_dicts persisted in the run checkpoint."""


@register_algorithm("ppo")
class PPO:
    """Clipped PPO over the canonical torchrl loop (cheat-sheet section 5)."""

    def setup(self, builder: "Builder") -> None:
        self.spec = builder.spec
        self.ppo_cfg: dict = builder.spec.algorithm.ppo
        self.actor = builder.actor()
        self.critic = builder.critic()
        self.gae = builder.gae(self.critic)
        self.loss_module = builder.loss(self.actor, self.critic)
        self.optim = builder.optimizer(self.loss_module)
        self.buffer = builder.buffer()

    @property
    def collect_policy(self):
        return self.actor

    @property
    def eval_actor(self):
        return self.actor

    # -- hooks a subclass may override ---------------------------------
    def _prepare_advantage(self, data: "TensorDictBase") -> "TensorDictBase":
        """Fill the "advantage"/"value_target" keys (called under no_grad)."""
        return self.gae(data)

    def _minibatch_loss(self, batch: "TensorDictBase") -> torch.Tensor:
        loss_td = self.loss_module(batch)
        self._last_loss_td = loss_td
        return (loss_td["loss_objective"] + loss_td["loss_critic"]
                + loss_td["loss_entropy"])

    def _clip_gradients(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(),
                                       self.ppo_cfg["max_grad_norm"])

    # -- protocol -------------------------------------------------------
    def train_on_batch(self, data: "TensorDictBase") -> dict[str, float]:
        for _ in range(self.ppo_cfg["epochs"]):
            with torch.no_grad():
                data = self._prepare_advantage(data)     # on [N, T], every epoch
            # ("next", camera) only feeds GAE's next-value pass — dropping it
            # before the buffer halves the dominant tensor's footprint
            slim = data
            if ("next", "camera") in data.keys(True):
                slim = data.exclude(("next", "camera"))
            self.buffer.extend(slim.reshape(-1))
            for batch in self.buffer:
                loss = self._minibatch_loss(batch)
                loss.backward()
                self._clip_gradients()
                self.optim.step()
                self.optim.zero_grad()

        logs = {}
        for k in ("loss_objective", "loss_critic", "loss_entropy",
                  "clip_fraction", "kl_approx"):
            if k in self._last_loss_td.keys():
                logs[f"Loss/{k}"] = float(self._last_loss_td[k].detach())
        return logs

    def observe_env_logs(self, logs: dict[str, Any]) -> None:
        pass

    def checkpoint(self) -> dict[str, Any]:
        return {"actor": self.actor.state_dict(),
                "critic": self.critic.state_dict()}


@register_algorithm("ppo_lagrangian")
class PPOLagrangian(PPO):
    """PPO + cost critic + PID-controlled lambda (plan section 4).

    Delta over PPO: a second GAE over the ("next", "cost") stream writes
    cost_advantage/cost_value_target; the surrogate uses the combined
    advantage (A_r - lambda * A_c) / (1 + lambda); the cost critic trains
    with a smooth-L1 value loss; lambda is a PID loop on the violation of
    the mean episode cost against the budget, updated once per iteration.
    """

    def setup(self, builder: "Builder") -> None:
        super().setup(builder)
        from ..algorithms import PIDLagrangian

        lag: dict = builder.spec.algorithm.lagrangian
        self.cost_critic = builder.critic(out_key="cost_value")
        self.gae_cost = builder.gae_cost(self.cost_critic)
        kp, ki, kd = lag.get("pid", (0.05, 0.0005, 0.1))
        self.pid = PIDLagrangian(lag["budget"], kp, ki, kd,
                                 lambda_init=lag.get("lambda_init", 0.0))
        self.optim = builder.optimizer(self.loss_module, self.cost_critic)
        self.j_cost = 0.0

    def observe_env_logs(self, logs: dict[str, Any]) -> None:
        ep_cost = logs.get("Episode/cost")
        if ep_cost is not None:
            self.j_cost = 0.7 * self.j_cost + 0.3 * float(ep_cost)
        self.pid.update(self.j_cost)

    def _prepare_advantage(self, data: "TensorDictBase") -> "TensorDictBase":
        data = self.gae(data)
        data = self.gae_cost(data)
        lam = self.pid.value
        data["advantage"] = ((data["advantage"] - lam * data["cost_advantage"])
                             / (1.0 + lam))
        return data

    def _minibatch_loss(self, batch: "TensorDictBase") -> torch.Tensor:
        loss = super()._minibatch_loss(batch)
        cost_pred = self.cost_critic(batch)["cost_value"]
        return loss + torch.nn.functional.smooth_l1_loss(
            cost_pred, batch["cost_value_target"])

    def _clip_gradients(self) -> None:
        super()._clip_gradients()
        torch.nn.utils.clip_grad_norm_(self.cost_critic.parameters(),
                                       self.ppo_cfg["max_grad_norm"])

    def train_on_batch(self, data: "TensorDictBase") -> dict[str, float]:
        logs = super().train_on_batch(data)
        logs["Safety/lambda"] = self.pid.value
        logs["Safety/j_cost"] = self.j_cost
        return logs

    def checkpoint(self) -> dict[str, Any]:
        out = super().checkpoint()
        out["cost_critic"] = self.cost_critic.state_dict()
        out["pid"] = self.pid.state_dict()
        return out
