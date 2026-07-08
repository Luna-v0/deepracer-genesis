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
from tensordict.nn import NormalParamExtractor, TensorDictModule, TensorDictSequential
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import Transform  # noqa: F401 (annotations)
from torchrl.envs.utils import ExplorationType
from torchrl.modules import MLP, ConvNet, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE

from ..configs.cfgs import get_env_cfg
from ..envs import DeepRacerEnv
from ..envs.torchrl_env import TorchRLDeepRacerEnv
from .spec import ExperimentSpec
from .transforms import ActionNoiseDelay, ImageAug

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
        """Translate the EnvSpec (+DR spec) into the sim's config dict."""
        env = self.spec.env
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
        cfg["random_start"] = env.random_start
        cfg["random_direction"] = env.random_direction
        cfg["reward_fn"] = env.reward_fn
        cfg["reward_scale_overrides"] = dict(env.reward_scales)
        if env.render == "nyx":
            cfg["vision_renderer"] = "nyx"
        if randomize:
            rand = dict(_NEUTRAL_PHYSICS)
            rand.update(obs_dr.physics)
            rand["camera_pitch_jitter_deg"] = obs_dr.camera_jitter.get("pitch_deg", 0.0)
            rand["camera_pos_jitter_m"] = obs_dr.camera_jitter.get("pos_m", 0.0)
            cfg["rand"] = rand
        if obs_dr.appearance:
            cfg["appearance"] = dict(obs_dr.appearance)
        if self.spec.policy is not None and self.spec.policy.actions:
            cfg["action_table"] = [list(a) for a in self.spec.policy.actions]
        if env.emits_cost:
            cfg["emit_cost"] = True
            cfg["cost_fn"] = env.cost_fn
        return cfg

    def sim(self, extra_cfg: dict | None = None) -> DeepRacerEnv:
        """Build (once) and return the Genesis sim. `extra_cfg` merges into
        the spec-derived config on first construction (e.g. spectator camera
        for visualization rollouts)."""
        if self._sim is None:
            _ensure_genesis()
            cfg = self.sim_cfg()
            if extra_cfg:
                cfg.update(extra_cfg)
            self._sim = DeepRacerEnv(num_envs=self.spec.env.num_envs, env_cfg=cfg)
        return self._sim

    # ------------------------------------------------------ torchrl env
    def transforms(self) -> list["Transform"]:
        """Obs/action transforms per spec (order: aug -> encoder -> action DR)."""
        ts = []
        if self.spec.obs_dr.image_aug:
            ts.append(ImageAug(self.spec.obs_dr.image_aug))
        if self.spec.encoder.kind == "frozen_cnn":
            ts.append(self.encoder_transform())
        ad = self.spec.action_dr
        if ad.delay_steps or ad.steer_noise or ad.speed_noise:
            ts.append(ActionNoiseDelay(self.spec.env.num_envs,
                                       steer_noise=ad.steer_noise,
                                       speed_noise=ad.speed_noise,
                                       delay_steps=ad.delay_steps,
                                       device=self.sim().device))
        return ts

    def env(self) -> "TorchRLDeepRacerEnv | TransformedEnv":
        """The TorchRL training env: wrapper (+ TransformedEnv when the spec
        carries transforms). Collection-side only; evaluation drives the raw
        sim (see evaluator.evaluate_policy)."""
        base = TorchRLDeepRacerEnv(self.sim(), emit_cost=self.spec.env.emits_cost)
        transforms = self.transforms()
        if not transforms:
            return base
        env = TransformedEnv(base)
        for t in transforms:
            env.append_transform(t)
        return env

    # ----------------------------------------------------------- models
    def _mlp(self, in_features: int, out_features: int) -> MLP:
        """Fused MLP head per the spec's mlp config (concatenates multiple
        positional inputs — the multi-key mechanism)."""
        p = self.spec.policy.mlp
        return MLP(in_features=in_features, out_features=out_features,
                   num_cells=list(p.get("hidden", (256, 128, 64))),
                   activation_class=_ACT[p.get("activation", "elu")],
                   device=self.sim().device)

    def _key_dims(self) -> dict[str, int]:
        """Flat width of every vector observation key a policy may read."""
        sim = self.sim()
        dims = {"state": sim.num_state_obs}
        if self.spec.encoder.kind == "frozen_cnn":
            dims[self.spec.encoder.out_key] = self.spec.encoder.output_dim
        return dims

    def _cnn(self) -> ConvNet:
        """Camera trunk per the spec's cnn config."""
        c = self.spec.policy.cnn
        return ConvNet(in_features=3,
                       num_cells=list(c["channels"]),
                       kernel_sizes=list(c["kernels"]),
                       strides=list(c["strides"]),
                       activation_class=_ACT[c.get("activation", "relu")],
                       device=self.sim().device)

    def _cnn_flat_dim(self, cnn: ConvNet) -> int:
        """Flattened feature width of `cnn` at the spec's resolution."""
        w, h = self.spec.env.resolution
        with torch.no_grad():
            return cnn(torch.zeros(1, 3, h, w, device=self.sim().device)).shape[-1]

    def _head(self, keys, dims, cam_feat_key):
        """Trunk for one network: optional CNN on 'camera' feeding a fused MLP.

        Returns (modules, head_in_keys): `modules` start the TensorDictSequential;
        vector keys (+ the CNN feature key) concat inside the MLP head.
        """
        modules = []
        head_keys = []
        in_dim = 0
        for k in keys:
            if k == "camera":
                cnn = self._cnn()
                modules.append(TensorDictModule(cnn, in_keys=["camera"],
                                                out_keys=[cam_feat_key]))
                head_keys.append(cam_feat_key)
                in_dim += self._cnn_flat_dim(cnn)
            else:
                head_keys.append(k)
                in_dim += dims[k]
        return modules, head_keys, in_dim

    def actor(self) -> ProbabilisticActor:
        """Actor over spec.policy.actor_keys ((camera via CNN trunk) + vector
        keys fused in the MLP head). Continuous => TanhNormal over
        [steer, speed]; policy.actions set => Categorical over that list
        (indices; the sim looks up the (steer, speed) pair)."""
        spec = self.spec
        if spec.policy.actions is not None:
            return self._discrete_actor()
        keys = list(spec.policy.actor_keys)
        dims = self._key_dims()
        # NormalParamExtractor is its own tensordict stage: only MLP.forward
        # concatenates multiple positional inputs, nn.Sequential does not
        if spec.policy.cnn is not None and "camera" in keys:
            modules, head_keys, in_dim = self._head(keys, dims, "actor_cam_feat")
            mlp = self._mlp(in_dim, 2 * 2)
            modules.append(TensorDictModule(mlp, in_keys=head_keys,
                                            out_keys=["_pi_params"]))
            # kept for checkpointing — Phase-5 transfer rebuilds the encoder
            self._actor_cnn, self._actor_mlp = modules[0].module, mlp
        else:
            mlp = self._mlp(sum(dims[k] for k in keys), 2 * 2)
            self._actor_cnn = self._actor_mlp = None
            modules = [TensorDictModule(mlp, in_keys=keys, out_keys=["_pi_params"])]
        modules.append(TensorDictModule(NormalParamExtractor(),
                                        in_keys=["_pi_params"],
                                        out_keys=["loc", "scale"]))
        param_module = TensorDictSequential(*modules)
        return ProbabilisticActor(
            param_module,
            in_keys=["loc", "scale"], out_keys=["action"],
            distribution_class=TanhNormal,
            distribution_kwargs={"low": -1.0, "high": 1.0},
            return_log_prob=True,
            default_interaction_type=ExplorationType.RANDOM,
        )

    def _discrete_actor(self) -> ProbabilisticActor:
        spec = self.spec
        keys = list(spec.policy.actor_keys)
        dims = self._key_dims()
        n_actions = len(spec.policy.actions)
        if spec.policy.cnn is not None and "camera" in keys:
            modules, head_keys, in_dim = self._head(keys, dims, "actor_cam_feat")
            mlp = self._mlp(in_dim, n_actions)
            modules.append(TensorDictModule(mlp, in_keys=head_keys,
                                            out_keys=["logits"]))
            self._actor_cnn, self._actor_mlp = modules[0].module, mlp
        else:
            mlp = self._mlp(sum(dims[k] for k in keys), n_actions)
            self._actor_cnn = self._actor_mlp = None
            modules = [TensorDictModule(mlp, in_keys=keys, out_keys=["logits"])]
        return ProbabilisticActor(
            TensorDictSequential(*modules),
            in_keys=["logits"], out_keys=["action"],
            distribution_class=torch.distributions.Categorical,
            return_log_prob=True,
            default_interaction_type=ExplorationType.RANDOM,
        )

    def critic(self, out_key: str = "state_value") -> "ValueOperator | TensorDictSequential":
        """Value head over spec.policy.critic_keys. `out_key` lets the
        Lagrangian build a second (cost) critic with its own CNN trunk."""
        spec = self.spec
        keys = list(spec.policy.critic_keys)
        dims = self._key_dims()
        if spec.policy.cnn is not None and "camera" in keys:
            feat_key = f"critic_cam_feat_{out_key}"      # own CNN per value head
            modules, head_keys, in_dim = self._head(keys, dims, feat_key)
            head = ValueOperator(self._mlp(in_dim, 1),
                                 in_keys=head_keys, out_keys=[out_key])
            return TensorDictSequential(*modules, head)
        return ValueOperator(self._mlp(sum(dims[k] for k in keys), 1),
                             in_keys=keys, out_keys=[out_key])

    # -------------------------------------------------- frozen encoder
    def encoder_module(self) -> tuple[nn.Module, int]:
        """Rebuild the checkpointed camera actor's trunk as a frozen encoder.

        Output = activations of the actor-MLP hidden layer whose width equals
        spec.encoder.output_dim (e.g. 256 with the default (256,128,64) MLP);
        the CNN and the MLP prefix up to that layer are loaded and frozen.
        """
        enc = self.spec.encoder
        ckpt = torch.load(enc.checkpoint, map_location=self.sim().device,
                          weights_only=False)
        if "actor_cnn" not in ckpt:
            raise ValueError(
                f"{enc.checkpoint} is not a camera-policy checkpoint "
                "(no 'actor_cnn'); train a camera experiment first")
        cnn_cfg, mlp_cfg = ckpt["cnn_cfg"], ckpt["mlp_cfg"]
        cnn = ConvNet(in_features=3, num_cells=list(cnn_cfg["channels"]),
                      kernel_sizes=list(cnn_cfg["kernels"]),
                      strides=list(cnn_cfg["strides"]),
                      activation_class=_ACT[cnn_cfg.get("activation", "relu")],
                      device=self.sim().device)
        cnn.load_state_dict(ckpt["actor_cnn"])
        flat = self._cnn_flat_dim(cnn)
        mlp = MLP(in_features=flat, out_features=2 * 2,
                  num_cells=list(mlp_cfg.get("hidden", (256, 128, 64))),
                  activation_class=_ACT[mlp_cfg.get("activation", "elu")],
                  device=self.sim().device)
        mlp.load_state_dict(ckpt["actor_mlp"])
        # slice the MLP after the linear+activation pair of width output_dim
        layers, width = [], None
        for mod in mlp:
            layers.append(mod)
            if isinstance(mod, nn.Linear):
                width = mod.out_features
            elif width == enc.output_dim:
                break
        else:
            dims = [m.out_features for m in mlp if isinstance(m, nn.Linear)]
            raise ValueError(
                f"encoder output_dim={enc.output_dim} matches no hidden layer "
                f"of the checkpointed actor MLP (available: {dims[:-1]})")
        encoder = nn.Sequential(cnn, *layers).eval()
        encoder.requires_grad_(False)
        return encoder, enc.output_dim

    def encoder_transform(self):
        from .transforms import FrozenEncoder
        encoder, dim = self.encoder_module()
        return FrozenEncoder(encoder, dim, in_keys=("camera",),
                             out_keys=(self.spec.encoder.out_key,))

    # -------------------------------------------------------- optimizing
    def gae(self, critic):
        ppo = self.spec.algorithm.ppo
        return GAE(gamma=ppo["gamma"], lmbda=ppo["gae_lambda"],
                   value_network=critic, device=self.sim().device)

    def gae_cost(self, cost_critic):
        """Second GAE over the cost stream (cheat-sheet: set_keys fields have
        no _key suffix; reads ("next","cost"))."""
        lag = self.spec.algorithm.lagrangian
        ppo = self.spec.algorithm.ppo
        g = GAE(gamma=ppo["gamma"], lmbda=lag.get("cost_gae_lambda", 0.95),
                value_network=cost_critic, device=self.sim().device)
        g.set_keys(advantage="cost_advantage", value_target="cost_value_target",
                   value="cost_value", reward="cost")
        return g

    def loss(self, actor: ProbabilisticActor, critic) -> ClipPPOLoss:
        """Clipped PPO loss wired per spec.algorithm.ppo."""
        ppo = self.spec.algorithm.ppo
        return ClipPPOLoss(actor, critic,
                           clip_epsilon=ppo["clip"],
                           entropy_coeff=ppo["entropy_coef"],
                           critic_coeff=1.0,
                           loss_critic_type="smooth_l1",
                           normalize_advantage=True)

    def collector(self, env, actor) -> Collector:
        """On-policy collector: frames_per_batch = num_envs * horizon."""
        ppo = self.spec.algorithm.ppo
        n = self.spec.env.num_envs
        return Collector(env, actor,
                         frames_per_batch=n * ppo["horizon"],
                         total_frames=self.spec.total_env_steps,
                         device=self.sim().device,
                         auto_register_policy_transforms=True)

    def buffer(self) -> TensorDictReplayBuffer:
        """One-rollout minibatch buffer (SamplerWithoutReplacement)."""
        ppo = self.spec.algorithm.ppo
        n = self.spec.env.num_envs
        frames = n * ppo["horizon"]
        return TensorDictReplayBuffer(
            storage=LazyTensorStorage(frames, device=self.sim().device),
            sampler=SamplerWithoutReplacement(),
            batch_size=max(1, frames // ppo["minibatches"]))

    def optimizer(self, loss_module, *extra_modules) -> torch.optim.Adam:
        """Adam over the loss module's params (+ any extra modules, e.g.
        the Lagrangian cost critic)."""
        params = list(loss_module.parameters())
        for m in extra_modules:
            params += list(m.parameters())
        return torch.optim.Adam(params, lr=self.spec.algorithm.ppo["lr"])
