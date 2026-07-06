"""Trainer: the algorithm-agnostic outer loop, emitting an EvalRecord.

The Trainer owns collection, logging, checkpointing, periodic + final
evaluation and the identity cache; everything algorithm-specific lives
behind the Algorithm protocol (see experiment/algorithms.py). Training-time
episode stats come from the SIM's own logs (the autoreset machinery
NaN-fills ("next", obs) at done rows, so collector data is unreliable for
episode metrics); evaluation drives the raw sim deterministically.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Optional

import torch

from .evaluator import EvalRecord, evaluate_policy


class Trainer:
    def __init__(self, builder, root: str = "runs") -> None:
        self.b = builder
        self.root = root

    # ------------------------------------------------------------------
    def fit(self, force: bool = False) -> EvalRecord:
        """Train the spec to completion (or return the cached record)."""
        spec = self.b.spec
        run_dir = spec.run_dir(self.root)
        record_path = os.path.join(run_dir, "eval_record.json")
        if os.path.exists(record_path) and not force:
            print(f"[trainer] cache hit: {run_dir}")
            return EvalRecord.load(record_path)

        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "spec.json"), "w") as f:
            json.dump(spec.to_dict(), f, indent=2)   # run record, never loaded

        torch.manual_seed(spec.seed)

        from .algorithms import make_algorithm
        env = self.b.env()
        algo = make_algorithm(self.b)
        collector = self.b.collector(env, algo.collect_policy)

        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(run_dir)

        sim = self.b.sim()
        obs_transform = self._eval_obs_transform()
        budget = (spec.algorithm.lagrangian.get("budget")
                  if spec.algorithm.kind == "ppo_lagrangian" else None)

        horizon = spec.algorithm.ppo.get("horizon", 24)
        iterations = max(1, spec.total_env_steps // (spec.env.num_envs * horizon))
        next_eval = spec.eval_every_steps or None
        eval_history: list[dict] = []
        t0 = time.perf_counter()
        frames = 0

        for i, data in enumerate(collector):
            frames += data.numel()
            algo.observe_env_logs(sim.extras.get("log", {}))
            logs = algo.train_on_batch(data)

            sps = frames / (time.perf_counter() - t0)
            for k, v in sim.extras.get("log", {}).items():
                writer.add_scalar(k, float(v), frames)
            for k, v in logs.items():
                writer.add_scalar(k, v, frames)
            writer.add_scalar("Train/steps_per_s", sps, frames)

            if i % 10 == 0 or i == iterations - 1:
                ep = sim.extras.get("log", {})
                rew = float(ep.get("Episode/rew_progress", float("nan")))
                print(f"[trainer] iter {i+1}/{iterations} frames {frames} "
                      f"sps {sps:.0f} rew_progress {rew:.2f}", flush=True)
            if (i + 1) % 25 == 0:
                self._save(run_dir, "last.pt", algo)

            if next_eval is not None and frames >= next_eval:
                # NOTE: mid-training eval resets the sim; the collector's next
                # step acts on a one-step-stale carrier obs — negligible for
                # on-policy data at sane eval cadences
                metrics = evaluate_policy(sim, algo.eval_actor,
                                          cost_budget=budget,
                                          obs_transform=obs_transform)
                eval_history.append({"frames": frames, **metrics})
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and v == v:
                        writer.add_scalar(f"eval/{k}", v, frames)
                print(f"[trainer] eval @ {frames}: "
                      f"completion {metrics.get('completion_rate', 0):.2f}",
                      flush=True)
                next_eval += spec.eval_every_steps

        collector.shutdown()
        wall = time.perf_counter() - t0
        ckpt = self._save(run_dir, "best.pt", algo)

        metrics = evaluate_policy(sim, algo.eval_actor, cost_budget=budget,
                                  obs_transform=obs_transform)
        writer.add_hparams({"spec_id": spec.id()},
                           {f"eval/{k}": v for k, v in metrics.items()
                            if isinstance(v, (int, float)) and v == v})
        writer.close()

        record = EvalRecord(
            spec_id=spec.id(), spec=spec.to_dict(), seed=spec.seed,
            ablation_group=spec.ablation_group, variant=spec.variant,
            metrics=metrics,
            eval_history=eval_history,
            train={"wall_clock_s": round(wall, 1),
                   "steps_per_s": round(frames / wall, 1),
                   "total_env_steps": frames,
                   "checkpoint": ckpt},
        )
        record.save(run_dir)
        print(f"[trainer] done: {run_dir}\n{json.dumps(metrics, indent=2)}")
        return record

    # ------------------------------------------------------------------
    def _eval_obs_transform(self) -> Optional[Callable]:
        """Frozen-encoder application for eval rollouts, when the spec has one."""
        spec = self.b.spec
        if spec.encoder.kind != "frozen_cnn":
            return None
        encoder, _ = self.b.encoder_module()
        out_key = spec.encoder.out_key

        def obs_transform(td, _enc=encoder, _k=out_key):
            td.set(_k, _enc(td["camera"]))
            return td

        return obs_transform

    def _save(self, run_dir: str, name: str, algo) -> str:
        """Persist algorithm state + (for camera policies) the actor trunk."""
        path = os.path.join(run_dir, name)
        payload = dict(algo.checkpoint())
        payload["spec"] = self.b.spec.to_dict()
        if getattr(self.b, "_actor_cnn", None) is not None:
            # camera policies also export the trunk so Phase-5 transfer can
            # rebuild a frozen encoder without touching actor internals
            payload["actor_cnn"] = self.b._actor_cnn.state_dict()
            payload["actor_mlp"] = self.b._actor_mlp.state_dict()
            payload["cnn_cfg"] = dict(self.b.spec.policy.cnn)
            payload["mlp_cfg"] = dict(self.b.spec.policy.mlp)
        torch.save(payload, path)
        return path
