"""rsl-rl execution backend for the experiment API (migration off TorchRL).

Preallocated rollout storage avoids the crash; see MIGRATION_TORCHRL_TO_RSLRL.md.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import torch

from .evaluator import EvalRecord, evaluate_policy

if TYPE_CHECKING:
    from .spec import ExperimentSpec

# our PPO-stage hyperparameter names -> rsl-rl algorithm cfg names
_PPO_KEY_MAP = {
    "clip": "clip_param",
    "epochs": "num_learning_epochs",
    "minibatches": "num_mini_batches",
    "gamma": "gamma",
    "gae_lambda": "lam",
    "lr": "learning_rate",
    "entropy_coef": "entropy_coef",
    "max_grad_norm": "max_grad_norm",
    "schedule": "schedule",
    "desired_kl": "desired_kl",
}


def rsl_supported(spec: "ExperimentSpec") -> bool:
    """True when spec is in the migrated (rsl-rl) scope.

    Excludes cost/Lagrangian, frozen-CNN, and discrete actions (their phases).

    Args:
        spec: The validated experiment spec.

    Returns:
        Whether ``run()`` should dispatch this spec to the rsl-rl backend.
    """
    # feature/camera PPO, symmetric or asymmetric (obs_groups is native), physics
    # + action/image DR (env-side); everything else stays on the TorchRL Trainer.
    e, p = spec.env, spec.policy
    return (
        e.modality in ("feature", "camera")
        and not e.emits_cost
        and spec.encoder.kind == "none"
        and p.actions is None                       # continuous only
    )


def _dr_extra_cfg(spec: "ExperimentSpec") -> dict:
    """Env-side action/image DR pulled from the spec into the sim cfg.

    Args:
        spec: The validated experiment spec.

    Returns:
        A cfg fragment with ``action_dr`` and/or ``image_aug`` (empty if none).
    """
    extra: dict = {}
    ad = spec.action_dr
    if ad.delay_steps or ad.steer_noise or ad.speed_noise:
        extra["action_dr"] = {"steer_noise": ad.steer_noise,
                              "speed_noise": ad.speed_noise,
                              "delay_steps": ad.delay_steps}
    if spec.obs_dr.image_aug:
        extra["image_aug"] = dict(spec.obs_dr.image_aug)
    return extra


def spec_to_train_cfg(spec: "ExperimentSpec") -> dict:
    """Translate an ExperimentSpec into an rsl-rl OnPolicyRunner train cfg.

    Args:
        spec: The validated experiment spec.

    Returns:
        The rsl-rl train config (obs_groups, actor/critic nets, PPO algorithm).
    """
    from ..configs.cfgs import get_train_cfg

    # the model class follows the POLICY, not the env: a camera env whose policy
    # reads only vectors (frozen-encoder transfer) needs the MLP model.
    keys = set(spec.policy.actor_keys) | set(spec.policy.critic_keys)
    vision = spec.env.modality == "camera" and "camera" in keys
    cfg = get_train_cfg(vision=vision)
    cfg["obs_groups"] = {"actor": list(spec.policy.actor_keys),
                         "critic": list(spec.policy.critic_keys)}
    hidden = spec.policy.mlp.get("hidden")
    if hidden:
        cfg["actor"]["hidden_dims"] = list(hidden)
        cfg["critic"]["hidden_dims"] = list(hidden)
    if vision and spec.policy.cnn:                    # map the spec's CNN trunk
        c = spec.policy.cnn
        cnn_cfg = {"output_channels": list(c["channels"]),
                   "kernel_size": list(c["kernels"]),
                   "stride": list(c["strides"]),
                   "activation": c.get("activation", "relu"), "flatten": True}
        cfg["actor"]["cnn_cfg"] = cnn_cfg
        if "cnn_cfg" in cfg["critic"]:
            cfg["critic"]["cnn_cfg"] = dict(cnn_cfg)
    if spec.policy.distribution:
        # std_range/learn_std/init_std etc.; std_range arrives as a list from
        # spec round-trips — rsl-rl accepts any sequence.
        cfg["actor"]["distribution_cfg"].update(spec.policy.distribution)
    ppo = spec.algorithm.ppo
    for ours, theirs in _PPO_KEY_MAP.items():
        if ours in ppo:
            cfg["algorithm"][theirs] = ppo[ours]
    if "horizon" in ppo:
        cfg["num_steps_per_env"] = ppo["horizon"]
    # Custom algorithm (Algo(cls=...)): rsl-rl's OnPolicyRunner resolves
    # cfg["algorithm"]["class_name"] via resolve_callable, which accepts a class
    # OBJECT directly — so the custom class plugs into the SAME runner PPO uses
    # (it must speak algorithms.protocol.RslAlgorithm; spec.validate checks that).
    if spec.algorithm.cls is not None:
        cfg["algorithm"]["class_name"] = spec.algorithm.cls
    return cfg


class _RslActor:
    """Adapt an rsl-rl inference policy to the evaluator's ``actor(td)`` call.

    Attributes:
        policy: rsl-rl inference policy mapping an obs TensorDict to actions.
    """

    def __init__(self, policy):
        self.policy = policy

    def __call__(self, td):
        td.set("action", self.policy(td))
        return td


def _eval(sim, policy):
    """Run the eval rollout under inference_mode (rsl-rl taints sim buffers).

    Args:
        sim: The DeepRacer sim to roll out.
        policy: The rsl-rl inference policy.

    Returns:
        The aggregated eval metrics.
    """
    # rsl-rl collection runs under torch.inference_mode(), marking the sim's
    # mutable buffers as inference tensors; eval must too, to update them in place.
    with torch.inference_mode():
        return evaluate_policy(sim, _RslActor(policy))


def run_rsl(spec: "ExperimentSpec", root: str = "runs", on_eval=None) -> EvalRecord:
    """Train spec via OnPolicyRunner, returning an EvalRecord.

    Runs learn() in eval-cadence chunks so periodic eval + on_eval survive.

    Args:
        spec: The validated experiment spec.
        root: Runs directory under which the run dir is created.
        on_eval: Callback ``on_eval(frames, metrics)`` after each periodic eval;
            raise inside it to stop the run (HPO pruning).

    Returns:
        The trained run's EvalRecord.
    """
    from rsl_rl.runners import OnPolicyRunner

    from .builder import Builder

    assert spec.env is not None and spec.algorithm is not None
    # reuse all sim_cfg logic (incl. physics DR); inject env-side action/image DR
    sim = Builder(spec).sim(extra_cfg=_dr_extra_cfg(spec) or None)
    device = str(sim.device)
    run_dir = spec.run_dir(root)
    os.makedirs(run_dir, exist_ok=True)

    train_cfg = spec_to_train_cfg(spec)
    horizon = train_cfg["num_steps_per_env"]
    per_iter = spec.env.num_envs * horizon
    total_iters = max(1, spec.total_env_steps // per_iter)
    eval_iters = (max(1, spec.eval_every_steps // per_iter)
                  if spec.eval_every_steps else 0)

    runner = OnPolicyRunner(sim, train_cfg, run_dir, device=device)
    if spec.resume:
        # weights only: load() otherwise restores the optimizer and with it
        # its lr, silently overwriting the one this spec asked for.
        runner.load(spec.resume, load_cfg={"actor": True, "critic": True,
                                           "optimizer": False, "iteration": False,
                                           "rnd": False})
        lr = train_cfg["algorithm"]["learning_rate"]
        print(f"[rsl] resumed weights from {spec.resume} (lr {lr:g}, "
              f"schedule {train_cfg['algorithm']['schedule']})")

    eval_history: list[dict] = []
    t0 = time.perf_counter()
    done_iters = 0
    first = True
    while done_iters < total_iters:
        chunk = min(eval_iters or total_iters, total_iters - done_iters)
        runner.learn(num_learning_iterations=chunk, init_at_random_ep_len=first)
        first = False
        done_iters += chunk
        frames = done_iters * per_iter
        if eval_iters:
            policy = runner.get_inference_policy(device=device)
            metrics = _eval(sim, policy)
            eval_history.append({"frames": frames, **metrics})
            if on_eval is not None:
                on_eval(frames, metrics)     # may raise to prune (HPO)

    wall = time.perf_counter() - t0
    policy = runner.get_inference_policy(device=device)
    metrics = _eval(sim, policy)
    ckpt = os.path.join(run_dir, "model.pt")
    try:
        runner.save(ckpt)
    except Exception:            # noqa: BLE001 - never lose the run over a save
        ckpt = ""

    # Part N.3: out-of-loop per-track holdout eval (opt-in: real_tracks set).
    # Guarded so a finished run is never lost if a scene can't rebuild in-process.
    holdout = {}
    if spec.eval.real_tracks:
        try:
            from .evaluator import build_single_track_sim, evaluate_on_tracks
            with torch.inference_mode():
                view = "gui" if spec.eval.gui else "none"
                holdout = evaluate_on_tracks(
                    _RslActor(policy), spec.eval.real_tracks,
                    sim_factory=lambda t: build_single_track_sim(
                        spec, t, spec.eval.eval_num_envs, view=view))
        except Exception as e:   # noqa: BLE001
            print(f"[rsl] holdout eval skipped ({type(e).__name__}: {e})")

    record = EvalRecord(
        spec_id=spec.id(), spec=spec.to_dict(), seed=spec.seed,
        ablation_group=spec.ablation_group, variant=spec.variant,
        metrics=metrics, eval_history=eval_history, holdout=holdout,
        train={"wall_clock_s": round(wall, 1),
               "total_env_steps": done_iters * per_iter,
               "steps_per_s": round(done_iters * per_iter / max(wall, 1e-9), 1),
               "checkpoint": ckpt, "backend": "rsl_rl"},
    )
    record.save(run_dir)

    # Part N.4: eval charts (opt-in via EvalConfig.charts; matplotlib optional).
    if spec.eval.charts:
        try:
            from .charts import render_charts
            render_charts(record, run_dir)
        except ImportError:
            pass
    print(f"[rsl] done: {run_dir}  completion={metrics.get('completion_rate', 0):.2f}")
    return record
