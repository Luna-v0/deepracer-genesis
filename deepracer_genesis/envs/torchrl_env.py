"""TorchRL EnvBase wrapper over the GPU-batched DeepRacerEnv sim.

Contract verified against torchrl 0.13.2 (see /tmp/torchrl_cheatsheet.md):
- the sim auto-resets done sub-envs inside its own step(), so we use the
  native autoreset flag (`_torchrl_native_autoreset = True`) — the collector
  then never issues synthetic resets; ("next", obs) is NaN-filled at done
  rows and GAE's NaN-sanitizer bootstraps truncated rows with V(obs_t).
- terminated = crash/offtrack (bootstrap killed), truncated = timeout
  (bootstrap kept) — the reason the sim's time_out_buf is surfaced separately.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from torchrl.data import Bounded, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase


class TorchRLDeepRacerEnv(EnvBase):
    def __init__(self, sim, emit_cost: bool = False):
        n = sim.num_envs
        device = sim.device
        super().__init__(device=device, batch_size=[n])
        self.sim = sim
        self.emit_cost = emit_cost

        obs = {"state": Unbounded(shape=(n, sim.num_state_obs),
                                  dtype=torch.float32, device=device)}
        if sim.vision:
            w, h = sim.cfg["camera_res"]
            obs["camera"] = Unbounded(shape=(n, 3, h, w),
                                      dtype=torch.float32, device=device)
        self.observation_spec = Composite(**obs, shape=(n,), device=device)
        self.action_spec = Bounded(-1.0, 1.0, shape=(n, 2),
                                   dtype=torch.float32, device=device)
        reward = {"reward": Unbounded(shape=(n, 1), dtype=torch.float32, device=device)}
        if emit_cost:
            # cost rides in the reward spec: reward-like keys are not NaN-filled
            # by the autoreset machinery, so the cost-GAE sees clean streams
            reward["cost"] = Unbounded(shape=(n, 1), dtype=torch.float32, device=device)
        self.full_reward_spec = Composite(**reward, shape=(n,), device=device)
        self.full_done_spec = Composite(
            done=Categorical(2, shape=(n, 1), dtype=torch.bool, device=device),
            terminated=Categorical(2, shape=(n, 1), dtype=torch.bool, device=device),
            truncated=Categorical(2, shape=(n, 1), dtype=torch.bool, device=device),
            shape=(n,), device=device)

        self._torchrl_native_autoreset = True

    # ------------------------------------------------------------------
    def _obs_leaves(self, obs_td) -> dict:
        leaves = {"state": obs_td["state"]}
        if self.sim.vision:
            leaves["camera"] = obs_td["camera"]
        return leaves

    def _step(self, tensordict):
        obs_td, rew, dones, _extras = self.sim.step(tensordict["action"])
        info = self.sim.step_info
        n1 = (*self.batch_size, 1)
        terminated = (info["offtrack"] | info["flipped"]).reshape(n1)
        truncated = (info["time_out"] & ~terminated.reshape(-1)).reshape(n1)
        out = {
            **self._obs_leaves(obs_td),            # post-reset obs on done rows
            "reward": rew.reshape(n1),
            "terminated": terminated,
            "truncated": truncated,
            "done": dones.reshape(n1),
        }
        if self.emit_cost:
            out["cost"] = self.sim.cost_buf.reshape(n1)
        return TensorDict(out, batch_size=self.batch_size, device=self.device)

    def _reset(self, tensordict, **kwargs):
        mask = tensordict.get("_reset", None) if tensordict is not None else None
        if mask is None:
            ids = torch.arange(self.sim.num_envs, device=self.device)
        else:
            ids = mask.reshape(-1).nonzero(as_tuple=True)[0]
        if len(ids):
            self.sim.reset_idx(ids)
            self.sim._post_physics(ids)
        obs_td = self.sim.get_observations()
        z = torch.zeros(*self.batch_size, 1, dtype=torch.bool, device=self.device)
        return TensorDict(
            {**self._obs_leaves(obs_td),
             "done": z, "terminated": z.clone(), "truncated": z.clone()},
            batch_size=self.batch_size, device=self.device)

    def _set_seed(self, seed):
        if seed is not None:
            torch.manual_seed(seed)   # sim spawn noise uses the global RNG
