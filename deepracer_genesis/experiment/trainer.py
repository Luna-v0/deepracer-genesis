"""Trainer: the PPO loop over Builder-made objects, emitting an EvalRecord.

Canonical torchrl 0.13.2 loop (cheat-sheet §5): GAE on the [N, T] collector
output every epoch, flatten into a SamplerWithoutReplacement buffer for
minibatches. Training-time episode stats come from the SIM's own logs (the
autoreset machinery NaN-fills ("next", obs) at done rows, so collector data
is unreliable for episode metrics); final metrics come from evaluate_policy
driving the raw sim.
"""

from __future__ import annotations

import json
import os
import time

import torch

from .evaluator import EvalRecord, evaluate_policy


class Trainer:
    def __init__(self, builder, root: str = "runs"):
        self.b = builder
        self.root = root

    def fit(self, force: bool = False) -> EvalRecord:
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

        env = self.b.env()
        actor = self.b.actor()
        critic = self.b.critic()
        gae = self.b.gae(critic)
        loss_module = self.b.loss(actor, critic)
        buffer = self.b.buffer()
        collector = self.b.collector(env, actor)

        lagrangian = spec.algorithm.kind == "ppo_lagrangian"
        if lagrangian:
            from ..algorithms import PIDLagrangian
            lag = spec.algorithm.lagrangian
            cost_critic = self.b.critic(out_key="cost_value")
            gae_cost = self.b.gae_cost(cost_critic)
            kp, ki, kd = lag.get("pid", (0.05, 0.0005, 0.1))
            pid = PIDLagrangian(lag["budget"], kp, ki, kd,
                                lambda_init=lag.get("lambda_init", 0.0))
            optim = self.b.optimizer(loss_module, cost_critic)
            j_cost = 0.0
        else:
            optim = self.b.optimizer(loss_module)

        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(run_dir)

        ppo = spec.algorithm.ppo
        sim = self.b.sim()
        t0 = time.perf_counter()
        frames = 0
        iterations = max(1, spec.total_env_steps // (spec.env.num_envs * ppo["horizon"]))

        for i, data in enumerate(collector):
            frames += data.numel()
            if lagrangian:
                # J_cost estimate: EMA of the sim's mean episode cost at resets
                ep_cost = sim.extras.get("log", {}).get("Episode/cost")
                if ep_cost is not None:
                    j_cost = 0.7 * j_cost + 0.3 * float(ep_cost)
                lam = pid.update(j_cost)
            for _ in range(ppo["epochs"]):
                with torch.no_grad():
                    data = gae(data)                    # on [N, T], every epoch
                    if lagrangian:
                        data = gae_cost(data)
                        # combined surrogate: (A_r - lam*A_c) / (1+lam), fed to
                        # ClipPPOLoss through the "advantage" key it reads
                        data["advantage"] = ((data["advantage"]
                                              - lam * data["cost_advantage"])
                                             / (1.0 + lam))
                buffer.extend(data.reshape(-1))
                for batch in buffer:
                    loss_td = loss_module(batch)
                    loss = (loss_td["loss_objective"] + loss_td["loss_critic"]
                            + loss_td["loss_entropy"])
                    if lagrangian:
                        cost_pred = cost_critic(batch)["cost_value"]
                        loss = loss + torch.nn.functional.smooth_l1_loss(
                            cost_pred, batch["cost_value_target"])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(loss_module.parameters(),
                                                   ppo["max_grad_norm"])
                    if lagrangian:
                        torch.nn.utils.clip_grad_norm_(cost_critic.parameters(),
                                                       ppo["max_grad_norm"])
                    optim.step()
                    optim.zero_grad()

            sps = frames / (time.perf_counter() - t0)
            for k, v in sim.extras.get("log", {}).items():
                writer.add_scalar(k, float(v), frames)
            writer.add_scalar("Train/steps_per_s", sps, frames)
            if lagrangian:
                writer.add_scalar("Safety/lambda", pid.value, frames)
                writer.add_scalar("Safety/j_cost", j_cost, frames)
            for k in ("loss_objective", "loss_critic", "loss_entropy",
                      "clip_fraction", "kl_approx"):
                if k in loss_td.keys():
                    writer.add_scalar(f"Loss/{k}", float(loss_td[k].detach()), frames)
            if i % 10 == 0 or i == iterations - 1:
                ep = sim.extras.get("log", {})
                rew = float(ep.get("Episode/rew_progress", float("nan")))
                print(f"[trainer] iter {i+1}/{iterations} frames {frames} "
                      f"sps {sps:.0f} rew_progress {rew:.2f}", flush=True)
            if (i + 1) % 25 == 0:
                self._save(run_dir, "last.pt", actor, critic)

        collector.shutdown()
        wall = time.perf_counter() - t0
        ckpt = self._save(run_dir, "best.pt", actor, critic)

        budget = (spec.algorithm.lagrangian.get("budget")
                  if spec.algorithm.kind == "ppo_lagrangian" else None)
        obs_transform = None
        if spec.encoder.kind == "frozen_cnn":
            encoder, _ = self.b.encoder_module()
            out_key = spec.encoder.out_key

            def obs_transform(td, _enc=encoder, _k=out_key):
                td.set(_k, _enc(td["camera"]))
                return td

        metrics = evaluate_policy(sim, actor, cost_budget=budget,
                                  obs_transform=obs_transform)
        writer.add_hparams({"spec_id": spec.id()},
                           {f"eval/{k}": v for k, v in metrics.items()
                            if isinstance(v, (int, float)) and v == v})
        writer.close()

        record = EvalRecord(
            spec_id=spec.id(), spec=spec.to_dict(), seed=spec.seed,
            ablation_group=spec.ablation_group, variant=spec.variant,
            metrics=metrics,
            train={"wall_clock_s": round(wall, 1),
                   "steps_per_s": round(frames / wall, 1),
                   "total_env_steps": frames,
                   "checkpoint": ckpt},
        )
        record.save(run_dir)
        print(f"[trainer] done: {run_dir}\n{json.dumps(metrics, indent=2)}")
        return record

    def _save(self, run_dir, name, actor, critic) -> str:
        path = os.path.join(run_dir, name)
        payload = {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "spec": self.b.spec.to_dict(),
        }
        if getattr(self.b, "_actor_cnn", None) is not None:
            # camera policies also export the trunk so Phase-5 transfer can
            # rebuild a frozen encoder without touching actor internals
            payload["actor_cnn"] = self.b._actor_cnn.state_dict()
            payload["actor_mlp"] = self.b._actor_mlp.state_dict()
            payload["cnn_cfg"] = dict(self.b.spec.policy.cnn)
            payload["mlp_cfg"] = dict(self.b.spec.policy.mlp)
        torch.save(payload, path)
        return path
