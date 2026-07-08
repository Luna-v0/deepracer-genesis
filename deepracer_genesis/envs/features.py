"""Feature-vector construction: what the policy (and the CNN) gets to see.

Two registered sets ship; select with ``FeatureEnvironment(feature_set=...)``:

**classic** (default) — the original 28-dim vector: normalized velocities,
track-relative pose, last action, and K look-ahead waypoints rotated into
the body frame.

**perception** — the sim2real-oriented design. One organizing principle:
*CNN = what's coming (feedforward), error channels = what's wrong now
(feedback)*, everything normalized by FIXED constants (never episodic max):

- CNN targets (readable from pixels at deploy; supervised from sim here):
  ``lateral_offset`` (±1 lane position), ``heading_error`` (/pi),
  ``speed`` (/4 m/s — grip depends on absolute speed, so the divisor is the
  constant), ``yaw_rate`` (/5), ``sideslip beta`` (the reactive slide
  channel), and signed look-ahead **curvature at fixed distances** (default
  1 m and 3 m): sign says which way, magnitude how much to slow, the
  near-far gradient how soon.
- Policy-only channels (involve the commanded action, so they cannot be in
  pixels): ``prev_action`` (last K), ``speed_error`` (commanded − achieved),
  ``steer_response_error`` (nominal-bicycle expected lateral accel −
  actual — the grip/understeer signal), ``yaw_error`` (expected − actual
  yaw rate — separates sliding from plowing). The nominal bicycle is FIXED
  (wheelbase/steering geometry at the center of the DR range).

Custom sets: subclass :class:`FeatureSet`, decorate with
``@register_feature_set("name")``, and select it by name.
"""

from __future__ import annotations

import math

import torch

FEATURE_SETS: dict[str, type] = {}

#: fixed normalization constants — never episodic statistics
MAX_SPEED = 4.0            # m/s (the env's action mapping ceiling)
MIN_SPEED = 0.1
YAW_RATE_NORM = 5.0        # rad/s
BETA_NORM = 0.5            # rad of sideslip ~= heavy slide
CURVATURE_NORM = 2.5       # 1/m (r = 0.4 m — about the tightest sane turn)
A_LAT_NORM = 20.0          # m/s^2
NOMINAL_WHEELBASE = 0.164  # m — DeepRacer geometry; DR scales gains, not geometry
MAX_STEER_RAD = math.radians(30.0)


def register_feature_set(name: str):
    """Class decorator: make `name` selectable as a ``feature_set``."""
    def deco(cls: type) -> type:
        if name in FEATURE_SETS:
            raise ValueError(f"feature set {name!r} already registered")
        FEATURE_SETS[name] = cls
        return cls
    return deco


def make_feature_set(name: str, env, params: dict | None = None) -> "FeatureSet":
    """Instantiate a registered feature set bound to a live env.

    Args:
        name: registered feature-set name.
        env: the DeepRacerEnv instance the set reads from.
        params: set-specific parameters (see each set's ``__init__``).

    Raises:
        ValueError: unknown name (lists what is registered).
    """
    try:
        cls = FEATURE_SETS[name]
    except KeyError:
        raise ValueError(
            f"unknown feature_set {name!r}; registered: {sorted(FEATURE_SETS)}"
        ) from None
    return cls(env, dict(params or {}))


def feature_dim(name: str, *, lookahead_k: int = 10,
                params: dict | None = None) -> int:
    """Vector width for a feature set WITHOUT building an env (used by the
    ONNX exporter and the model card)."""
    return FEATURE_SETS[name].dim_for(lookahead_k=lookahead_k,
                                      params=dict(params or {}))


def feature_layout(name: str, *, lookahead_k: int = 10,
                   params: dict | None = None) -> str:
    """Human-readable channel layout (for model cards / dataset metas)."""
    return FEATURE_SETS[name].layout_for(lookahead_k=lookahead_k,
                                         params=dict(params or {}))


class FeatureSet:
    """One feature-vector recipe bound to a live env.

    Subclasses implement :meth:`compute` (called once per control step,
    after the env refreshed its kinematic/track-frame attributes) and may
    keep per-env history buffers, cleared in :meth:`reset`.
    """

    def __init__(self, env, params: dict):
        self.env = env
        self.params = params

    @property
    def dim(self) -> int:
        raise NotImplementedError

    def compute(self) -> torch.Tensor:
        """Return the (N, dim) feature tensor for the current step."""
        raise NotImplementedError

    def reset(self, env_ids: torch.Tensor) -> None:
        """Clear any per-env history for freshly respawned envs."""

    @classmethod
    def dim_for(cls, *, lookahead_k: int, params: dict) -> int:
        raise NotImplementedError

    @classmethod
    def layout_for(cls, *, lookahead_k: int, params: dict) -> str:
        raise NotImplementedError

    #: (start, end) slice of channels a deployed CNN must predict; None if
    #: the whole vector is privileged-only (no pixel-readable subset)
    cnn_target_slice: tuple[int, int] | None = None


@register_feature_set("classic")
class ClassicFeatures(FeatureSet):
    """The original vector: pose/velocity + body-frame look-ahead waypoints."""

    @property
    def dim(self) -> int:
        return self.dim_for(lookahead_k=self.env.lookahead_k, params=self.params)

    @classmethod
    def dim_for(cls, *, lookahead_k: int, params: dict) -> int:
        return 8 + 2 * lookahead_k

    @classmethod
    def layout_for(cls, *, lookahead_k: int, params: dict) -> str:
        return ("v_forward/4, v_lateral, yaw_rate/5, lateral/half_width, "
                "sin(heading_err), cos(heading_err), last_action[2], "
                f"lookahead_rel_x[{lookahead_k}]/scale, "
                f"lookahead_rel_y[{lookahead_k}]/scale")

    def compute(self) -> torch.Tensor:
        env = self.env
        cy, sy = torch.cos(env.yaw), torch.sin(env.yaw)
        la_idx = env.track.lookahead(env.wp_idx, env.lookahead_k,
                                     env.cfg["lookahead_stride"],
                                     dir_sign=env.dir_sign)
        la_pts = env.track.lookahead_points(la_idx)              # (N, K, 2)
        rel = la_pts - env.base_pos[:, None, :2]
        rel_x = rel[..., 0] * cy[:, None] + rel[..., 1] * sy[:, None]
        rel_y = -rel[..., 0] * sy[:, None] + rel[..., 1] * cy[:, None]
        la_scale = env.cfg["lookahead_scale"]
        return torch.cat(
            [
                (env.v_forward / env.cfg["max_speed"]).unsqueeze(1),
                env.v_lateral.unsqueeze(1),
                (env.yaw_rate / YAW_RATE_NORM).unsqueeze(1),
                (env.lateral * env.dir_sign
                 / env.half_width.clamp(min=0.1)).unsqueeze(1),
                torch.sin(env.heading_err).unsqueeze(1),
                torch.cos(env.heading_err).unsqueeze(1),
                env.actions,
                rel_x / la_scale,
                rel_y / la_scale,
            ],
            dim=1,
        )


@register_feature_set("perception")
class PerceptionFeatures(FeatureSet):
    """CNN targets + action-conditioned error channels (module docstring).

    Params (all optional):
        horizons: look-ahead curvature probe distances in meters,
            default ``(1.0, 3.0)``.
        k_prev: how many past actions the policy sees, default 2.
        k_speed: speed_error history length (near-memoryless), default 2.
        k_steer: steer_response/yaw error history length — give it the
            corner-spanning window, default 8.
    """

    def __init__(self, env, params: dict):
        super().__init__(env, params)
        self.horizons = tuple(params.get("horizons", (1.0, 3.0)))
        self.k_prev = int(params.get("k_prev", 2))
        self.k_speed = int(params.get("k_speed", 2))
        self.k_steer = int(params.get("k_steer", 8))
        n, dev = env.num_envs, env.device
        self._prev_actions = torch.zeros(n, self.k_prev, 2, device=dev)
        self._speed_err = torch.zeros(n, self.k_speed, device=dev)
        self._steer_err = torch.zeros(n, self.k_steer, device=dev)
        self._yaw_err = torch.zeros(n, self.k_steer, device=dev)

    # ---- static shape info -------------------------------------------
    @staticmethod
    def _counts(params: dict) -> tuple[int, int, int, int]:
        h = len(tuple(params.get("horizons", (1.0, 3.0))))
        return (5 + h, int(params.get("k_prev", 2)),
                int(params.get("k_speed", 2)), int(params.get("k_steer", 8)))

    @property
    def dim(self) -> int:
        return self.dim_for(lookahead_k=0, params=self.params)

    @classmethod
    def dim_for(cls, *, lookahead_k: int, params: dict) -> int:
        targets, k_prev, k_speed, k_steer = cls._counts(params)
        return targets + 2 * k_prev + k_speed + 2 * k_steer

    @classmethod
    def layout_for(cls, *, lookahead_k: int, params: dict) -> str:
        h = tuple(params.get("horizons", (1.0, 3.0)))
        targets, k_prev, k_speed, k_steer = cls._counts(params)
        return (f"CNN targets[{targets}]: lateral_offset, heading_err/pi, "
                f"speed/4, yaw_rate/5, sideslip_beta/{BETA_NORM}, "
                f"curvature@{list(h)}m/{CURVATURE_NORM} | policy-only: "
                f"prev_action[{k_prev}x2], speed_error[{k_speed}], "
                f"steer_response_error[{k_steer}], yaw_error[{k_steer}]")

    @property
    def cnn_target_slice(self) -> tuple[int, int]:
        return (0, self._counts(self.params)[0])

    # ---- per-step ------------------------------------------------------
    def compute(self) -> torch.Tensor:
        env = self.env

        # -- CNN targets: what a camera could tell you ------------------
        lateral = env.lateral * env.dir_sign / env.half_width.clamp(min=0.1)
        heading = env.heading_err / math.pi
        speed = env.v_forward / MAX_SPEED
        yaw_rate = env.yaw_rate / YAW_RATE_NORM
        beta = torch.atan2(env.v_lateral,
                           env.v_forward.clamp(min=0.05)) / BETA_NORM
        kappa = env.track.curvature_ahead(
            env.progress_m, self.horizons, env.dir_sign) / CURVATURE_NORM

        # -- error channels vs the FIXED nominal bicycle ----------------
        steer_cmd = env.actions[:, 0] * MAX_STEER_RAD
        speed_cmd = MIN_SPEED + (env.actions[:, 1] + 1) * 0.5 * (MAX_SPEED - MIN_SPEED)
        yaw_expected = env.v_forward * torch.tan(steer_cmd) / NOMINAL_WHEELBASE
        a_lat_expected = env.v_forward * yaw_expected            # v^2 tan(d)/L
        a_lat_actual = env.v_forward * env.yaw_rate

        speed_err = (speed_cmd - env.v_forward) / MAX_SPEED
        steer_err = ((a_lat_expected - a_lat_actual) / A_LAT_NORM).clamp(-2, 2)
        yaw_err = ((yaw_expected - env.yaw_rate) / YAW_RATE_NORM).clamp(-2, 2)

        # -- roll the histories (newest first) --------------------------
        self._prev_actions = torch.cat(
            [env.actions.unsqueeze(1), self._prev_actions[:, :-1]], dim=1)
        self._speed_err = torch.cat(
            [speed_err.unsqueeze(1), self._speed_err[:, :-1]], dim=1)
        self._steer_err = torch.cat(
            [steer_err.unsqueeze(1), self._steer_err[:, :-1]], dim=1)
        self._yaw_err = torch.cat(
            [yaw_err.unsqueeze(1), self._yaw_err[:, :-1]], dim=1)

        return torch.cat(
            [
                lateral.unsqueeze(1), heading.unsqueeze(1),
                speed.unsqueeze(1), yaw_rate.unsqueeze(1), beta.unsqueeze(1),
                kappa,
                self._prev_actions.reshape(env.num_envs, -1),
                self._speed_err,
                self._steer_err,
                self._yaw_err,
            ],
            dim=1,
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        self._prev_actions[env_ids] = 0.0
        self._speed_err[env_ids] = 0.0
        self._steer_err[env_ids] = 0.0
        self._yaw_err[env_ids] = 0.0
