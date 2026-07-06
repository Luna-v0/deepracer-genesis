"""Evaluation: episodic metric aggregation + EvalRecord (plan section 5.2).

The aggregation core is torchrl-agnostic — it consumes plain (T, N) tensors —
so it is unit-testable without a simulator and reusable by any rollout loop.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class EvalRecord:
    """One run's provenance + measurements; the Reporter's input unit."""

    spec_id: str
    spec: dict                      # one-way dump of the ExperimentSpec
    seed: int
    ablation_group: Optional[str]
    variant: Optional[str]
    metrics: dict = field(default_factory=dict)
    train: dict = field(default_factory=dict)   # steps_per_s, wall_clock_s, ...
    created_at: str = ""

    def save(self, run_dir: str) -> str:
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, "eval_record.json")
        payload = dataclasses.asdict(self)
        payload["created_at"] = payload["created_at"] or time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path

    @staticmethod
    def load(path: str) -> "EvalRecord":
        with open(path) as f:
            return EvalRecord(**json.load(f))


def evaluate_policy(sim, actor, steps: Optional[int] = None,
                    obs_transform=None, cost_budget: Optional[float] = None) -> dict:
    """Deterministic eval rollout driving the RAW sim (plan section 5.2).

    Bypasses the collector/autoreset machinery entirely: per-step episode
    info comes straight from sim.step_info, so terminal-step stats are exact.
    `obs_transform` (e.g. the frozen encoder) is applied to observations
    before the policy when the trained policy expects derived keys.
    """
    from torchrl.envs.utils import ExplorationType, set_exploration_type

    steps = steps or sim.max_episode_length + 300
    n = sim.num_envs
    device = sim.device
    sim.reset_idx(torch.arange(n, device=device))
    sim._post_physics(torch.arange(n, device=device))

    use_cost = cost_budget is not None and hasattr(sim, "cost_buf")
    streams = {k: [] for k in ("reward", "done", "progress_delta", "offtrack")}
    costs = [] if use_cost else None

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for _ in range(steps):
            td = sim.get_observations().clone()
            if obs_transform is not None:
                td = obs_transform(td)
            td = actor(td)
            _, rew, dones, _ = sim.step(td["action"])
            info = sim.step_info
            streams["reward"].append(rew.clone())
            streams["done"].append(dones.clone())
            streams["progress_delta"].append(info["progress_delta"])
            streams["offtrack"].append(info["offtrack"] | info["flipped"])
            if use_cost:
                costs.append(sim.cost_buf.clone())

    stacked = {k: torch.stack(v) for k, v in streams.items()}
    return aggregate_episodes(
        control_dt=sim.dt,
        track_length=sim.track.total_len_env,
        cost=torch.stack(costs) if use_cost else None,
        cost_budget=cost_budget,
        **stacked)


def aggregate_episodes(
    reward: torch.Tensor,            # (T, N) per-step rewards
    done: torch.Tensor,              # (T, N) bool: episode ended after this step
    progress_delta: torch.Tensor,    # (T, N) meters gained this step
    offtrack: torch.Tensor,          # (T, N) bool: offtrack termination event
    control_dt: float,
    track_length,                    # scalar or (N,): meters per lap (per env)
    cost: Optional[torch.Tensor] = None,    # (T, N) per-step cost
    cost_budget: Optional[float] = None,
) -> dict:
    """Fold per-step streams into per-episode stats, then into scalar metrics.

    Only COMPLETED episodes (a done inside the window) are counted, so the
    partial trailing episode of each env never biases the stats. Laps derive
    from cumulative progress / track_length — robust to random spawn points
    (a car spawning just before the finish line does not get a free "lap").
    """
    T, N = reward.shape
    device = reward.device
    track_length = torch.as_tensor(track_length, device=device,
                                   dtype=torch.float32).expand(N)
    ep_return = torch.zeros(N, device=device)
    ep_progress = torch.zeros(N, device=device)
    ep_len = torch.zeros(N, device=device)
    ep_cost = torch.zeros(N, device=device)

    returns, progresses, lengths, lap_counts, offs, costs = ([] for _ in range(6))

    for t in range(T):
        ep_return += reward[t]
        ep_progress += progress_delta[t]
        ep_len += 1
        if cost is not None:
            ep_cost += cost[t]
        d = done[t]
        if d.any():
            idx = d.nonzero(as_tuple=True)[0]
            returns.append(ep_return[idx])
            progresses.append(ep_progress[idx])
            lengths.append(ep_len[idx])
            lap_counts.append(ep_progress[idx] / track_length[idx])
            offs.append(offtrack[t][idx].float())
            if cost is not None:
                costs.append(ep_cost[idx])
            ep_return[idx] = 0
            ep_progress[idx] = 0
            ep_len[idx] = 0
            if cost is not None:
                ep_cost[idx] = 0

    if not returns:            # no episode finished inside the window
        return {"episodes": 0}

    returns = torch.cat(returns)
    progresses = torch.cat(progresses)
    lengths = torch.cat(lengths)
    lap_counts = torch.cat(lap_counts)
    offs = torch.cat(offs)

    completed = lap_counts >= 1.0
    times = lengths * control_dt
    metrics = {
        "episodes": int(returns.numel()),
        "mean_return": float(returns.mean()),
        "mean_progress_m": float(progresses.mean()),
        "mean_episode_s": float(times.mean()),
        "completion_rate": float(completed.float().mean()),
        "mean_laps": float(lap_counts.mean()),
        "offtrack_rate": float(offs.mean()),
        "lap_time_s": (float((times[completed] / lap_counts[completed]).mean())
                       if completed.any() else float("nan")),
        "mean_speed_mps": float((progresses / times).mean()),
    }
    if cost is not None:
        ep_costs = torch.cat(costs)
        metrics["mean_cost"] = float(ep_costs.mean())
        if cost_budget is not None:
            metrics["cost_violation_rate"] = float((ep_costs > cost_budget).float().mean())
            metrics["budget_satisfied"] = bool(ep_costs.mean() <= cost_budget)
    return metrics
