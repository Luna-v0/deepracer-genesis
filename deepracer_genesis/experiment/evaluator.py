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


def aggregate_episodes(
    reward: torch.Tensor,            # (T, N) per-step rewards
    done: torch.Tensor,              # (T, N) bool: episode ended after this step
    progress_delta: torch.Tensor,    # (T, N) meters gained this step
    laps: torch.Tensor,              # (T, N) cumulative laps within the episode
    offtrack: torch.Tensor,          # (T, N) bool: offtrack termination event
    control_dt: float,
    cost: Optional[torch.Tensor] = None,    # (T, N) per-step cost
    cost_budget: Optional[float] = None,
) -> dict:
    """Fold per-step streams into per-episode stats, then into scalar metrics.

    Only COMPLETED episodes (a done inside the window) are counted, so the
    partial trailing episode of each env never biases the stats.
    """
    T, N = reward.shape
    device = reward.device
    ep_return = torch.zeros(N, device=device)
    ep_progress = torch.zeros(N, device=device)
    ep_len = torch.zeros(N, device=device)
    ep_cost = torch.zeros(N, device=device)

    returns, progresses, lengths, lap_counts, lap_times, offs, costs = ([] for _ in range(7))

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
            lap_counts.append(laps[t][idx])
            offs.append(offtrack[t][idx].float())
            if cost is not None:
                costs.append(ep_cost[idx])
            done_laps = laps[t][idx]
            done_len = ep_len[idx]
            completed = done_laps >= 1
            if completed.any():
                lap_times.append(
                    (done_len[completed] * control_dt) / done_laps[completed])
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

    metrics = {
        "episodes": int(returns.numel()),
        "mean_return": float(returns.mean()),
        "mean_progress_m": float(progresses.mean()),
        "mean_episode_s": float(lengths.mean() * control_dt),
        "completion_rate": float((lap_counts >= 1).float().mean()),
        "mean_laps": float(lap_counts.mean()),
        "offtrack_rate": float(offs.mean()),
        "lap_time_s": float(torch.cat(lap_times).mean()) if lap_times else float("nan"),
        "mean_speed_mps": float((progresses / (lengths * control_dt)).mean()),
    }
    if cost is not None:
        ep_costs = torch.cat(costs)
        metrics["mean_cost"] = float(ep_costs.mean())
        if cost_budget is not None:
            metrics["cost_violation_rate"] = float((ep_costs > cost_budget).float().mean())
            metrics["budget_satisfied"] = bool(ep_costs.mean() <= cost_budget)
    return metrics
