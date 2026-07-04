"""Default env + rsl-rl-lib 5.x train configs for DeepRacer-Genesis."""


def get_env_cfg(vision=False, track="reinvent_base", randomize=False, topdown=False):
    return {
        # sim
        "dt": 0.01,
        "decimation": 2,          # control at 50 Hz
        "episode_length_s": 30.0,
        "track": track,
        # action mapping (original DeepRacer Box([-30, 0.1], [30, 4.0]))
        "max_steering_deg": 30.0,
        "min_speed": 0.1,
        "max_speed": 4.0,
        # actuation
        "steer_kp": 25.0,
        "steer_kv": 5.0,     # heavy damping needed: low values cause front-wheel shimmy
        "wheel_kv": 5.0,
        "wheel_max_torque": 3.0,
        # spawn / termination
        "random_start": True,
        "spawn_lateral_noise": 0.15,
        "spawn_yaw_noise": 0.3,
        "spawn_height": 0.03,
        "off_track_margin": 0.10,   # m beyond road edge before terminating
        "wheel_margin": 0.08,       # ~half track width of the car, for all_wheels_on_track
        "crash_penalty": -10.0,
        # observations
        "lookahead_k": 10,
        "lookahead_stride": 3,
        "lookahead_scale": 3.0,
        "obs_noise": 0.0,
        # vision
        "vision": vision,
        "camera_res": (160, 120),      # DeepRacer-native observation resolution
        "camera_fov": 90,
        "camera_pitch_deg": 10.0,
        "topdown_camera": topdown,     # per-env batch camera (validation checks)
        "spectator": False,            # high-res rasterizer cam, all cars in one view
        "spectator_res": (1280, 960),
        "madrona_rg_swap": False,      # see env: only alpha-cutout textures are swapped
        "vision_renderer": "batch",    # "batch" (Madrona) | "raster" (per-env EGL cams)
        "pixel_noise": 0.0,
        "light_intensity": 6.0,
        # rewards
        "reward_scales": {
            "progress": 10.0,
            "speed": 0.5,
            "centered": 0.5,
            "heading": 0.5,
            "steering": 0.3,
            "action_rate": 0.05,
            "off_track": 2.0,
        },
        # domain randomization
        "randomize": randomize,
        # per the Genesis DR guide: friction/mass/COM per link, kp/kv/armature
        # per dof, all per-env (needs batch_dofs_info/batch_links_info)
        "rand": {
            "friction_range": (0.6, 1.4),
            "mass_shift_kg": 0.2,
            "com_shift_m": 0.01,
            "steer_kp_scale": (0.8, 1.2),
            "wheel_kv_scale": (0.8, 1.2),
            "armature_range": (0.0, 0.01),
            "camera_pitch_jitter_deg": 2.0,
            "camera_pos_jitter_m": 0.005,
        },
    }


def get_train_cfg(vision=False, visual_only=False):
    if vision:
        # visual_only: the critic sees pixels only too (no privileged state)
        obs_groups = ({"actor": ["camera"], "critic": ["camera"]} if visual_only
                      else {"actor": ["camera"], "critic": ["state", "camera"]})
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

    cfg = {
        "num_steps_per_env": 24,
        "save_interval": 100,
        "obs_groups": obs_groups,
        "logger": "tensorboard",
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.01,
            "learning_rate": 3.0e-4,
            "max_grad_norm": 1.0,
            "schedule": "adaptive",
            "desired_kl": 0.01,
        },
        "actor": actor,
        "critic": critic,
    }
    if share_cnn:
        cfg["algorithm"]["share_cnn_encoders"] = True
    return cfg
