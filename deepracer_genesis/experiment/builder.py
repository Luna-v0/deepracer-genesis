"""Builder: ExperimentSpec -> live Genesis sim + TorchRL objects (plan §3).

The only layer that imports the heavy stuff. All APIs verified against the
installed torchrl 0.13.2 (see /tmp/torchrl_cheatsheet.md); notable version
facts baked in: Collector (SyncDataCollector was removed), entropy_coeff
spelling, TanhNormal(low/high), action_log_prob key, GAE runs on the [N, T]
collector output directly.
"""

from __future__ import annotations

import torch
from torch import nn

import genesis as gs
from tensordict.nn import NormalParamExtractor, TensorDictModule
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import TransformedEnv
from torchrl.envs.utils import ExplorationType
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE

from ..configs.cfgs import get_env_cfg
from ..envs import DeepRacerEnv
from ..envs.torchrl_env import TorchRLDeepRacerEnv
from .spec import ExperimentSpec

_ACT = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}

_NEUTRAL_PHYSICS = {
    "friction_range": (1.0, 1.0),
    "mass_shift_kg": 0.0,
    "com_shift_m": 0.0,
    "steer_kp_scale": (1.0, 1.0),
    "wheel_kv_scale": (1.0, 1.0),
    "armature_range": (0.0, 0.0),
}


def _ensure_genesis():
    try:
        gs.init(backend=gs.cuda, logging_level="warning")
    except Exception as e:  # second init in one process
        if "initialized" not in str(e).lower():
            raise


class Builder:
    def __init__(self, spec: ExperimentSpec):
        spec.validate()
        self.spec = spec
        self._sim = None

    # ------------------------------------------------------------- sim
    def sim_cfg(self) -> dict:
        env = self.spec.env
        if env.render == "nyx":
            raise NotImplementedError(
                "render='nyx' needs the nyx-vision branch merged (Phase 2)")
        if env.features:
            raise NotImplementedError(f"extra features not implemented: {env.features}")

        obs_dr = self.spec.obs_dr
        randomize = bool(obs_dr.physics or obs_dr.camera_jitter)
        track = list(env.tracks) if len(env.tracks) > 1 else env.tracks[0]
        cfg = get_env_cfg(vision=(env.modality == "camera"), track=track,
                          randomize=randomize)
        cfg["camera_res"] = tuple(env.resolution)
        cfg["camera_fov"] = env.fov
        cfg["lookahead_k"] = env.lookahead_k
        if randomize:
            rand = dict(_NEUTRAL_PHYSICS)
            rand.update(obs_dr.physics)
            rand["camera_pitch_jitter_deg"] = obs_dr.camera_jitter.get("pitch_deg", 0.0)
            rand["camera_pos_jitter_m"] = obs_dr.camera_jitter.get("pos_m", 0.0)
            cfg["rand"] = rand
        if env.emits_cost:
            cfg["emit_cost"] = True
            cfg["cost_fn"] = env.cost_fn
        return cfg

    def sim(self) -> DeepRacerEnv:
        if self._sim is None:
            _ensure_genesis()
            self._sim = DeepRacerEnv(num_envs=self.spec.env.num_envs,
                                     env_cfg=self.sim_cfg())
        return self._sim

    # ------------------------------------------------------ torchrl env
    def transforms(self) -> list:
        """Obs/action transforms per spec — filled by Phases 3 and 5."""
        return []

    def env(self):
        base = TorchRLDeepRacerEnv(self.sim(), emit_cost=self.spec.env.emits_cost)
        transforms = self.transforms()
        if not transforms:
            return base
        env = TransformedEnv(base)
        for t in transforms:
            env.append_transform(t)
        return env

    # ----------------------------------------------------------- models
    def _mlp(self, in_features, out_features):
        p = self.spec.policy.mlp
        return MLP(in_features=in_features, out_features=out_features,
                   num_cells=list(p.get("hidden", (256, 128, 64))),
                   activation_class=_ACT[p.get("activation", "elu")],
                   device=self.sim().device)

    def _key_dims(self) -> dict:
        sim = self.sim()
        dims = {"state": sim.num_state_obs}
        if self.spec.encoder.kind == "frozen_cnn":
            dims[self.spec.encoder.out_key] = self.spec.encoder.output_dim
        return dims

    def actor(self) -> ProbabilisticActor:
        spec = self.spec
        if spec.policy.cnn is not None:
            raise NotImplementedError("camera policies land in Phase 2")
        keys = list(spec.policy.actor_keys)
        dims = self._key_dims()
        net = nn.Sequential(
            self._mlp(sum(dims[k] for k in keys), 2 * 2),
            NormalParamExtractor(),
        )
        return ProbabilisticActor(
            TensorDictModule(net, in_keys=keys, out_keys=["loc", "scale"]),
            in_keys=["loc", "scale"], out_keys=["action"],
            distribution_class=TanhNormal,
            distribution_kwargs={"low": -1.0, "high": 1.0},
            return_log_prob=True,
            default_interaction_type=ExplorationType.RANDOM,
        )

    def critic(self, out_key: str = "state_value") -> ValueOperator:
        spec = self.spec
        if spec.policy.cnn is not None:
            raise NotImplementedError("camera policies land in Phase 2")
        keys = list(spec.policy.critic_keys)
        dims = self._key_dims()
        return ValueOperator(self._mlp(sum(dims[k] for k in keys), 1),
                             in_keys=keys, out_keys=[out_key])

    # -------------------------------------------------------- optimizing
    def gae(self, critic):
        ppo = self.spec.algorithm.ppo
        return GAE(gamma=ppo["gamma"], lmbda=ppo["gae_lambda"],
                   value_network=critic, device=self.sim().device)

    def loss(self, actor, critic):
        ppo = self.spec.algorithm.ppo
        return ClipPPOLoss(actor, critic,
                           clip_epsilon=ppo["clip"],
                           entropy_coeff=ppo["entropy_coef"],
                           critic_coeff=1.0,
                           loss_critic_type="smooth_l1",
                           normalize_advantage=True)

    def collector(self, env, actor):
        ppo = self.spec.algorithm.ppo
        n = self.spec.env.num_envs
        return Collector(env, actor,
                         frames_per_batch=n * ppo["horizon"],
                         total_frames=self.spec.total_env_steps,
                         device=self.sim().device)

    def buffer(self):
        ppo = self.spec.algorithm.ppo
        n = self.spec.env.num_envs
        frames = n * ppo["horizon"]
        return TensorDictReplayBuffer(
            storage=LazyTensorStorage(frames, device=self.sim().device),
            sampler=SamplerWithoutReplacement(),
            batch_size=max(1, frames // ppo["minibatches"]))

    def optimizer(self, loss_module):
        return torch.optim.Adam(loss_module.parameters(),
                                lr=self.spec.algorithm.ppo["lr"])
