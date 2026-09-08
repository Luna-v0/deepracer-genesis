"""Default env + rsl-rl-lib 5.x train configs for DeepRacer-Genesis."""

from dataclasses import asdict, dataclass, field

from ..physics.limits import MAX_SPEED, MAX_STEERING_DEG, MIN_SPEED
from .schema import EnvConfig


def get_env_cfg(vision=False, track="reinvent_base", randomize=False,
                topdown=False, backend="gpu", view="none") -> EnvConfig:
    """Build the default nested env config, one typed section per subsystem."""
    return {
        "sim": {
            "dt": 0.01,
            "decimation": 2,          # control at 50 Hz
            "episode_length_s": 30.0,
            "track": track,
            "backend": backend,       # "gpu" | "cpu" (Part M)
            "view": view,             # "none" | "gui" | "spectator" | "topdown"
            # viewer pacing when view="gui": Nx real time; <=0 = uncapped
            "realtime_factor": 1.0,
            # Part O: metres between track tiles in camera multi-track mode.
            # Must exceed the camera's reach so no foreign tile enters frame;
            # only applied when vision + >1 track (ignored otherwise).
            "track_grid_spacing": 100.0,
        },
        # action mapping (original DeepRacer Box([-30, 0.1], [30, 4.0]))
        "action": {
            "max_steering_deg": MAX_STEERING_DEG,
            "min_speed": MIN_SPEED,
            "max_speed": MAX_SPEED,
            "action_table": None,     # discrete (steer, speed) pairs, or None
        },
        "car": {
            "steer_kp": 25.0,
            "steer_kv": 5.0,          # heavy damping: low values cause front-wheel shimmy
            "wheel_kv": 5.0,
            "wheel_max_torque": 3.0,
            # "ackermann" (per-wheel inner/outer) | "parallel" (legacy, both equal)
            "steering_model": "ackermann",
        },
        "spawn": {
            "random_start": True,
            "random_direction": False,   # coin-flip CW/CCW driving direction per episode
            "spawn_lateral_noise": 0.15,
            "spawn_yaw_noise": 0.3,
            "spawn_height": 0.03,
        },
        "termination": {
            "off_track_margin": 0.10,   # m beyond road edge before terminating
            "wheel_margin": 0.08,       # ~half car width, for all_wheels_on_track
            "crash_penalty": -10.0,
            "overspeed_limit": 3.5,     # m/s, for the offtrack_or_overspeed cost
        },
        "obs": {
            "lookahead_k": 10,
            "lookahead_stride": 3,
            "lookahead_scale": 3.0,
            "obs_noise": 0.0,
            "feature_set": None,        # a FeatureSet subclass, or None -> ClassicFeatures
            "feature_params": {},
            "obs_routing": None,        # Part K.3 per-signal actor/critic routing (None = single "state")
        },
        "vision": {
            "vision": vision,
            "camera_res": (160, 120),   # DeepRacer-native observation resolution
            "camera_fov": 90,
            "camera_pitch_deg": 10.0,
            "policy_res": None,         # downscale target for the policy (None = camera_res)
            "frame_stack": 4,           # frames stacked along channels (1 = single frame);
                                        # 4 is the contract default: ego-velocity is
                                        # unobservable from one frame (AWS/dr-gym precedent)
            "topdown_camera": topdown,  # per-env batch camera (validation checks)
            "spectator": False,         # high-res rasterizer cam, all cars in one view
            "spectator_res": (1280, 960),
            "madrona_rg_swap": False,   # see env: only alpha-cutout textures are swapped
            "vision_renderer": "batch",  # "batch" (Madrona GPU) | "nyx" (path tracer GPU) | "rasterizer" (per-env CPU)
            "nyx_mode": "Forward",      # "Forward" | "FastPathTracer" | "RefPathTracer"
            "nyx_spp": 4,
            "nyx_light_intensity": 3.0,
            "light_intensity": 6.0,
            "pixel_noise": 0.0,
            "background_color": (0.55, 0.72, 0.9),
            "field_color": (0.30, 0.48, 0.32),
            "appearance": {},           # {"world_color": strength} for the color remap
            "env_map": {},              # Part P.1 Nyx per-env sky DR (tint/multiplier ranges)
        },
        "reward": {
            "reward": None,             # a reward callable, or None -> deepracer default
            "reward_scales": {
                "progress": 10.0,
                "speed": 0.5,
                "centered": 0.5,
                "heading": 0.5,
                "steering": 0.3,
                "action_rate": 0.05,
                "off_track": 2.0,
            },
            "reward_scale_overrides": {},
            "reward_params": {},        # constants exposed as env.reward_params
            "emit_cost": False,
            "cost_fn": None,
        },
        # per the Genesis DR guide: friction/mass/COM per link, kp/kv/armature
        # per dof, all per-env (needs batch_dofs_info/batch_links_info)
        "rand": {
            "randomize": randomize,
            "friction_range": (0.6, 1.4),
            "mass_shift_kg": 0.2,
            "com_shift_m": 0.01,
            "steer_kp_scale": (0.8, 1.2),
            "wheel_kv_scale": (0.8, 1.2),
            "armature_range": (0.0, 0.01),
            "camera_pitch_jitter_deg": 2.0,
            "camera_pos_jitter_m": 0.005,
            # feature-mode geometry DR: per-episode scale on the rulebook track
            # half-width (sim2real: printed tracks measured wider). Off = (1,1).
            "track_width_scale": (1.0, 1.0),
        },
    }


@dataclass(frozen=True)
class AlgoConfig:
    """PPO hyperparameters for the rsl-rl runner (build-time).

    Attributes:
        class_name: Algorithm class the runner instantiates.
        num_learning_epochs: Optimization passes over each rollout batch.
        num_mini_batches: Minibatches the rollout is split into per epoch.
        clip_param: PPO surrogate-objective clipping range.
        gamma: Reward discount factor.
        lam: GAE bias-variance trade-off coefficient.
        value_loss_coef: Weight on the critic value loss.
        entropy_coef: Weight on the entropy bonus.
        learning_rate: Optimizer step size.
        max_grad_norm: Gradient-norm clipping threshold.
        schedule: Learning-rate schedule strategy.
        desired_kl: Target KL divergence for the adaptive schedule.
    """

    class_name: str = "PPO"
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    clip_param: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    learning_rate: float = 3.0e-4
    max_grad_norm: float = 1.0
    schedule: str = "adaptive"
    desired_kl: float = 0.01


@dataclass(frozen=True)
class TrainConfig:
    """rsl-rl OnPolicyRunner config; ``asdict()`` yields the dict it consumes.

    Attributes:
        obs_groups: Observation groups routed to actor and critic.
        actor: Actor network specification.
        critic: Critic network specification.
        algorithm: PPO hyperparameters.
        num_steps_per_env: Rollout length collected per environment.
        save_interval: Iterations between checkpoint saves.
        logger: Logging backend to use.
    """

    obs_groups: dict
    actor: dict
    critic: dict
    algorithm: AlgoConfig = field(default_factory=AlgoConfig)
    num_steps_per_env: int = 24
    save_interval: int = 100
    logger: str = "tensorboard"


def get_train_cfg(vision=False) -> dict:
    if vision:
        obs_groups = {"actor": ["camera"], "critic": ["state", "camera"]}
        actor = {
            "class_name": "CNNModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": False,
            "cnn_cfg": {
                "output_channels": [16, 32, 64],
                "kernel_size": [8, 4, 3],
                "stride": [4, 2, 1],
                "activation": "relu",
                "flatten": True,
            },
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0},
        }
        critic = {
            "class_name": "CNNModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": False,
            "cnn_cfg": {
                "output_channels": [16, 32, 64],
                "kernel_size": [8, 4, 3],
                "stride": [4, 2, 1],
                "activation": "relu",
                "flatten": True,
            },
        }
        share_cnn = True
    else:
        obs_groups = {"actor": ["state"], "critic": ["state"]}
        actor = {
            "class_name": "MLPModel",
            "hidden_dims": [256, 128, 64],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0},
        }
        critic = {
            "class_name": "MLPModel",
            "hidden_dims": [256, 128, 64],
            "activation": "elu",
            "obs_normalization": True,
        }
        share_cnn = False

    cfg = asdict(TrainConfig(obs_groups=obs_groups, actor=actor, critic=critic))
    if share_cnn:
        cfg["algorithm"]["share_cnn_encoders"] = True
    return cfg
