"""PPO-Lagrangian pieces (plan section 4).

The lambda controller is Stooke et al.'s PID-Lagrangian (ICML 2020): a PID
loop on the constraint violation (J_cost - budget) instead of naive dual
ascent, which oscillates. The combined surrogate uses
A = (A_reward - lambda * A_cost) / (1 + lambda) — the 1/(1+lambda) scaling
keeps the effective advantage magnitude stable as lambda grows.

The PPO delta lives in PPOLagrangian below: a second (cost) critic + second
GAE writing cost_advantage, the combined advantage written into the
"advantage" key ClipPPOLoss reads, and a separate smooth-L1 value loss for
the cost critic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .ppo import PPO
from .protocol import register_algorithm

if TYPE_CHECKING:
    from tensordict import TensorDictBase

    from ..experiment.builder import Builder


class PIDLagrangian:
    """One scalar lambda >= 0, PID-updated once per PPO iteration."""

    def __init__(self, budget: float, kp: float = 0.05, ki: float = 0.0005,
                 kd: float = 0.1, lambda_init: float = 0.0):
        self.budget = float(budget)
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = max(0.0, lambda_init / ki) if ki > 0 else 0.0
        self.prev_error = 0.0
        self.value = max(0.0, lambda_init)

    def update(self, j_cost: float) -> float:
        """Feed the current mean episode cost estimate; returns lambda."""
        error = float(j_cost) - self.budget
        self.integral = max(0.0, self.integral + error)
        derivative = max(0.0, error - self.prev_error)
        self.prev_error = error
        self.value = max(0.0, self.kp * error + self.ki * self.integral
                         + self.kd * derivative)
        return self.value

    def state_dict(self) -> dict:
        return {"integral": self.integral, "prev_error": self.prev_error,
                "value": self.value}

    def load_state_dict(self, state: dict):
        self.integral = state["integral"]
        self.prev_error = state["prev_error"]
        self.value = state["value"]


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
