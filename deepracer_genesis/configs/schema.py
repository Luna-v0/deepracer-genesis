"""Typed sections for the runtime env config: one ``TypedDict`` per subsystem
so each gets only its slice, not one blob that does everything.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, TypedDict

if TYPE_CHECKING:
    from ..envs.rewards import RewardFn


class SimConfig(TypedDict):
    """Simulator + episode timing, consumed by the env/scene build."""

    dt: float
    decimation: int
    episode_length_s: float
    track: object   # str name, or list/tuple of names for multi-track
    backend: Literal["gpu", "cpu"]   # Genesis compute backend (Part M)
    view: Literal["none", "gui", "spectator", "topdown"]   # ViewRenderer (Part M)


class ActionConfig(TypedDict):
    """Action-space mapping from normalized actions to physical commands."""

    max_steering_deg: float
    min_speed: float
    max_speed: float
    action_table: Optional[list]   # discrete (steer, speed) pairs, or None


class CarConfig(TypedDict):
    """Car actuation gains + steering geometry, consumed by ``Car.configure``."""

    steer_kp: float
    steer_kv: float
    wheel_kv: float
    wheel_max_torque: float
    steering_model: Literal["ackermann", "parallel"]


class SpawnConfig(TypedDict):
    """Per-episode spawn placement + direction, consumed by ``reset_idx``."""

    random_start: bool
    random_direction: bool
    spawn_lateral_noise: float
    spawn_yaw_noise: float
    spawn_height: float


class TerminationConfig(TypedDict):
    """Off-track / crash termination thresholds, consumed by ``mdp``/``rules``."""

    off_track_margin: float
    wheel_margin: float
    crash_penalty: float
    overspeed_limit: float
    max_laps: Optional[int]   # truncate the episode after N laps (None = endless)


class ObsConfig(TypedDict):
    """State-vector / feature-set observation settings."""

    lookahead_k: int
    lookahead_spacing_m: float   # meters between arclength lookahead samples (P1)
    lookahead_scale: float
    obs_noise: float
    feature_set: Optional[type]   # a FeatureSet subclass, or None -> ClassicFeatures
    feature_params: dict
    obs_routing: Optional[dict]   # Part K.3 actor/critic routing (None = single "state")


class VisionConfig(TypedDict):
    """Camera + renderer settings, consumed by the ``Renderer`` strategy."""

    vision: bool
    camera_res: tuple
    camera_fov: float
    camera_pitch_deg: float
    policy_res: Optional[tuple]
    frame_stack: int
    topdown_camera: bool
    spectator: bool
    spectator_res: tuple
    madrona_rg_swap: bool
    vision_renderer: Literal["batch", "nyx", "rasterizer"]
    nyx_mode: str
    nyx_spp: int
    nyx_light_intensity: float
    light_intensity: float
    pixel_noise: float
    background_color: tuple
    field_color: tuple
    appearance: dict
    env_map: dict          # Part P.1 Nyx per-env sky DR ({"tint": (lo,hi), "multiplier": (lo,hi)})


class RewardConfig(TypedDict):
    """Reward callable + term scales and the (optional) cost stream."""

    reward: "Optional[RewardFn]"   # None -> the built-in `deepracer` default
    reward_scales: dict
    reward_scale_overrides: dict
    reward_params: dict            # constants exposed as env.reward_params
    emit_cost: bool
    cost_fn: Optional[str]


class RandConfig(TypedDict):
    """Domain-randomization ranges + the master ``randomize`` switch."""

    randomize: bool
    friction_range: tuple
    mass_shift_kg: float
    com_shift_m: float
    steer_kp_scale: tuple
    wheel_kv_scale: tuple
    armature_range: tuple
    camera_pitch_jitter_deg: float
    camera_pos_jitter_m: float


class EnvConfig(TypedDict):
    """The full env config: one typed section per subsystem."""

    sim: SimConfig
    action: ActionConfig
    car: CarConfig
    spawn: SpawnConfig
    termination: TerminationConfig
    obs: ObsConfig
    vision: VisionConfig
    reward: RewardConfig
    rand: RandConfig
