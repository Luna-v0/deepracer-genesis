"""Episodic metric aggregation and EvalRecord over torchrl-agnostic (T, N) tensors."""

from __future__ import annotations

import dataclasses
import json
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

import torch

if TYPE_CHECKING:
    from ..envs.deepracer_env import DeepRacerEnv


@dataclass
class EvalRecord:
    """One run's provenance and measurements; the Reporter's input unit.

    Attributes:
        spec_id: Identifier of the experiment spec that produced this run.
        spec: One-way dump of the ExperimentSpec.
        seed: Random seed used for the run.
        ablation_group: Ablation group label, if part of an ablation study.
        variant: Variant name within the ablation group, if any.
        metrics: Final scalar evaluation metrics.
        train: Training-side stats (steps_per_s, wall_clock_s, ...).
        eval_history: Periodic evals as [{frames, **metrics}] entries.
        created_at: Timestamp string set when the record is saved.
    """

    spec_id: str
    spec: dict                      # one-way dump of the ExperimentSpec
    seed: int
    ablation_group: Optional[str]
    variant: Optional[str]
    metrics: dict = field(default_factory=dict)
    train: dict = field(default_factory=dict)   # steps_per_s, wall_clock_s, ...
    eval_history: list = field(default_factory=list)  # periodic evals: [{frames, **metrics}]
    holdout: dict = field(default_factory=dict)  # out-of-loop per-track eval: {track: {metrics}}
    created_at: str = ""

    def save(self, run_dir: str) -> str:
        """Write this record as eval_record.json under `run_dir`.

        Args:
            run_dir: Run directory (created if missing).

        Returns:
            Path of the written JSON file.
        """
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, "eval_record.json")
        payload = dataclasses.asdict(self)
        payload["created_at"] = payload["created_at"] or time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path

    @staticmethod
    def load(path: str) -> "EvalRecord":
        """Load a record previously written by save().

        Args:
            path: Path to an eval_record.json file.

        Returns:
            The reconstructed EvalRecord.
        """
        with open(path) as f:
            return EvalRecord(**json.load(f))


def evaluate_policy(sim: "DeepRacerEnv", actor, steps: Optional[int] = None,
                    obs_transform: Callable | None = None,
                    cost_budget: Optional[float] = None,
                    telemetry: Optional[str] = None) -> dict:
    """Run a deterministic eval rollout on the raw sim, reading exact
    terminal stats from sim.step_info (no collector/autoreset).

    Args:
        sim: The raw Genesis sim (DeepRacerEnv), not the TorchRL wrapper.
        actor: TensorDict policy module producing an "action" key; run with
            deterministic (mean) actions.
        steps: Rollout length; defaults to sim.max_episode_length + 300.
        obs_transform: Optional callable applied to observations before the
            policy (e.g. the frozen encoder) when the trained policy expects
            derived keys.
        cost_budget: Per-episode cost budget; enables the cost metrics when
            the sim exposes a cost_buf.
        telemetry: Optional ``.parquet`` path — records per-step trajectory
            telemetry for the whole rollout (``analysis.telemetry``) and
            flushes it there; recording failures are printed, never raised
            (an eval must not die over its bookkeeping).

    Returns:
        Scalar metrics dict from aggregate_episodes().
    """
    steps = steps or sim.max_episode_length + 300
    n = sim.num_envs
    device = sim.device
    sim.reset_idx(torch.arange(n, device=device))
    sim._post_physics(torch.arange(n, device=device))

    recorder = None
    if telemetry is not None:
        try:
            from ..analysis.telemetry import TelemetryRecorder
            recorder = TelemetryRecorder(sim)
        except Exception as e:                    # noqa: BLE001
            print(f"[telemetry] recorder unavailable ({e}); eval continues")

    use_cost = cost_budget is not None and hasattr(sim, "cost_buf")
    streams = {k: [] for k in ("reward", "done", "progress_delta", "offtrack")}
    costs = [] if use_cost else None

    with torch.no_grad():
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
            if recorder is not None:
                recorder.step(rew, dones, info["offtrack"] | info["flipped"],
                              info["progress_delta"])
            if use_cost:
                costs.append(sim.cost_buf.clone())
    if recorder is not None:
        try:
            print(f"[telemetry] {recorder.flush(telemetry)}")
        except Exception as e:                    # noqa: BLE001
            print(f"[telemetry] flush failed ({e}); eval metrics unaffected")

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
    """Fold per-step streams into scalar metrics, counting only completed
    episodes and deriving laps from cumulative progress / track_length.

    Args:
        reward: (T, N) per-step rewards.
        done: (T, N) bool; True where the episode ended after this step.
        progress_delta: (T, N) meters gained this step.
        offtrack: (T, N) bool; offtrack termination event.
        control_dt: Control timestep in seconds.
        track_length: Scalar or (N,): meters per lap (per env).
        cost: Optional (T, N) per-step cost.
        cost_budget: Per-episode budget; adds the violation metrics when
            `cost` is given.

    Returns:
        Metrics dict: episodes, mean_return, mean_progress_m, mean_episode_s,
        completion_rate, mean_laps, offtrack_rate, lap_time_s, mean_speed_mps
        (+ mean_cost, cost_violation_rate, budget_satisfied when cost is
        given). Just {"episodes": 0} when no episode finished inside the
        window.
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


# ---------------------------------------------------------------------------
# Out-of-loop per-track holdout evaluation (Part N.3)

def build_single_track_sim(spec, track: str, num_envs: int, view: str = "none"):
    """Build a fresh single-track sim for ``track`` under NOMINAL conditions.

    A fresh sim per track sidesteps the one-scene-per-process / camera
    superimpose constraint and yields clean per-track metrics. Camera mode
    should call this in its own process (one per track).

    Every DR slice is stripped before building: holdout eval measures the
    policy under nominal conditions, not one randomization draw (matching
    ``visualize.rollout_video``). Before this, eval incoherently kept physics
    DR while dropping image/action DR — holdout numbers from runs predating
    the change are not directly comparable.

    Args:
        spec: The experiment spec (its sim_cfg is reused, track overridden,
            DR stripped).
        track: The single track name to evaluate on.
        num_envs: Parallel envs for the eval rollout.
        view: view renderer for this eval sim (``"none"`` default). ``"gui"``
            opens the interactive viewer so you can watch the rollout; it is
            auto-paced to real time when the spec did not already set a positive
            ``realtime_factor`` (else the short rollout flies by too fast to see).

    Returns:
        A built DeepRacerEnv on the spec's backend for exactly ``track``.
    """
    from dataclasses import replace

    from .._gs import ensure_init
    from ..envs import DeepRacerEnv
    from .builder import Builder
    from .spec import ActionDRSpec, ObsDRSpec

    spec = replace(spec, obs_dr=ObsDRSpec(), action_dr=ActionDRSpec())
    cfg = Builder(spec).sim_cfg()
    cfg["sim"]["track"] = track
    cfg["sim"]["view"] = view
    if view == "gui" and cfg["sim"].get("realtime_factor", 0) <= 0:
        cfg["sim"]["realtime_factor"] = 1.0    # pace the watch to real time
    ensure_init(cfg["sim"].get("backend", "gpu"))
    return DeepRacerEnv(num_envs=num_envs, env_cfg=cfg)


def evaluate_on_tracks(actor, tracks, *, sim_factory, obs_transform=None,
                       cost_budget=None, telemetry_dir=None) -> dict:
    """Evaluate ``actor`` on each track INDEPENDENTLY; return {track: metrics}.

    Args:
        actor: Deterministic policy module (see :func:`evaluate_policy`).
        tracks: The array of track names to evaluate on (e.g. the real/printed
            holdout tracks).
        sim_factory: ``track -> sim`` builder (e.g. a partial over
            :func:`build_single_track_sim`); called once per track so each gets
            a fresh scene.
        obs_transform: Optional obs transform applied before the policy.
        cost_budget: Per-episode cost budget (enables cost metrics).
        telemetry_dir: Optional directory — records per-step trajectory
            telemetry per track as ``holdout_<track>.parquet``.

    Returns:
        ``{track: metrics}`` — per-track metrics from :func:`evaluate_policy`,
        the clean per-track breakdown the aggregate path cannot produce.
    """
    out = {}
    for track in tracks:
        sim = sim_factory(track)
        path = (os.path.join(telemetry_dir, f"holdout_{track}.parquet")
                if telemetry_dir else None)
        out[track] = evaluate_policy(sim, actor, obs_transform=obs_transform,
                                     cost_budget=cost_budget, telemetry=path)
    return out
