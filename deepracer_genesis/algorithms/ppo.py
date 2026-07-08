"""Clipped PPO on the canonical torchrl loop (verified against torchrl 0.13.2)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .protocol import register_algorithm

if TYPE_CHECKING:
    from tensordict import TensorDictBase

    from ..experiment.builder import Builder


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


