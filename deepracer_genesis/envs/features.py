"""Build the policy/CNN feature vector from the live env state.

Ships ``ClassicFeatures`` and ``PerceptionFeatures``; pass a FeatureSet subclass.
"""

from __future__ import annotations

import math

import torch

#: fixed normalization constants + physical caps — the single source of truth
#: is physics/limits.py; re-exported here so `from ...features import MAX_SPEED`
#: (etc.) keeps working. NOMINAL_WHEELBASE is now the exact URDF wheelbase
#: (0.163974, was a rounded 0.164).
from ..physics.limits import (  # noqa: E402
    A_LAT_NORM, BETA_NORM, CURVATURE_NORM, MAX_SPEED, MAX_STEER_RAD, MIN_SPEED,
    YAW_RATE_NORM, WHEELBASE_M as NOMINAL_WHEELBASE,
)


def resolve_feature_set(feature_set):
    """Return the FeatureSet class to use, defaulting ``None`` to ClassicFeatures.

    Args:
        feature_set: A FeatureSet subclass, or None for the default.

    Returns:
        The FeatureSet subclass (ClassicFeatures when given None).
    """
    return feature_set if feature_set is not None else ClassicFeatures


def feature_dim(feature_set, *, lookahead_k: int = 10,
                params: dict | None = None) -> int:
    """Return a feature set's vector width without building an env.

    Args:
        feature_set: A FeatureSet subclass, or None for the default.
        lookahead_k: Number of lookahead waypoints the set may use.
        params: Set-specific parameters.

    Returns:
        The state-vector width the set produces.
    """
    return resolve_feature_set(feature_set).dim_for(
        lookahead_k=lookahead_k, params=dict(params or {}))


def feature_layout(feature_set, *, lookahead_k: int = 10,
                   params: dict | None = None) -> str:
    """Return a human-readable channel layout (model cards / dataset metas).

    Args:
        feature_set: A FeatureSet subclass, or None for the default.
        lookahead_k: Number of lookahead waypoints the set may use.
        params: Set-specific parameters.

    Returns:
        A string describing each channel of the state vector.
    """
    return resolve_feature_set(feature_set).layout_for(
        lookahead_k=lookahead_k, params=dict(params or {}))


class FeatureSet:
    """One feature-vector recipe bound to a live env.

    Subclasses implement :meth:`compute` and clear history in :meth:`reset`.

    Attributes:
        env: the live env the features are read from.
        params: set-specific configuration for this recipe.
        cnn_target_slice: channel range a deployed CNN must predict, or None.
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

    @classmethod
    def reads_for(cls, *, params: dict) -> tuple[str, ...]:
        """Signal names this set reads (a feature->signal trace, Part K.2/K.5).

        Order-preserving, de-duplicated. Feeds the build-time learnability
        check so the builder can trace each observed feature to its signal(s).
        Default empty; presets override.
        """
        return ()

    #: (start, end) slice of channels a deployed CNN must predict; None if
    #: the whole vector is privileged-only (no pixel-readable subset)
    cnn_target_slice: tuple[int, int] | None = None


class ClassicFeatures(FeatureSet):
    """The original vector: pose/velocity + body-frame look-ahead waypoints.

    Attributes:
        env: the live env the features are read from.
        params: set-specific configuration for this recipe.
    """

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

    @classmethod
    def reads_for(cls, *, params: dict) -> tuple[str, ...]:
        # same signals as the full ordered block selection; see FEATURE_BLOCKS.
        return ("v_forward", "v_lateral", "yaw_rate", "lateral", "half_width",
                "heading_err", "actions")

    def compute(self) -> torch.Tensor:
        env = self.env
        cy, sy = torch.cos(env.yaw), torch.sin(env.yaw)
        la_idx = env.track.lookahead(env.wp_idx, env.lookahead_k,
                                     env.cfg["obs"]["lookahead_stride"],
                                     dir_sign=env.dir_sign)
        la_pts = env.track.lookahead_points(la_idx)              # (N, K, 2)
        rel = la_pts - env.base_pos[:, None, :2]
        rel_x = rel[..., 0] * cy[:, None] + rel[..., 1] * sy[:, None]
        rel_y = -rel[..., 0] * sy[:, None] + rel[..., 1] * cy[:, None]
        la_scale = env.cfg["obs"]["lookahead_scale"]
        return torch.cat(
            [
                (_read(env, "v_forward") / env.cfg["action"]["max_speed"]).unsqueeze(1),
                _read(env, "v_lateral").unsqueeze(1),
                (_read(env, "yaw_rate") / YAW_RATE_NORM).unsqueeze(1),
                (_read(env, "lateral") * env.dir_sign
                 / _read(env, "half_width").clamp(min=0.1)).unsqueeze(1),
                torch.sin(_read(env, "heading_err")).unsqueeze(1),
                torch.cos(_read(env, "heading_err")).unsqueeze(1),
                _read(env, "actions"),
                rel_x / la_scale,
                rel_y / la_scale,
            ],
            dim=1,
        )


class PerceptionFeatures(FeatureSet):
    """CNN targets + action-conditioned error channels.

    Optional params: ``horizons``, ``k_prev``, ``k_speed``, ``k_steer``.

    Attributes:
        env: the live env the features are read from.
        params: set-specific configuration for this recipe.
        horizons: look-ahead distances (m) for curvature target channels.
        k_prev: number of previous actions kept in the history.
        k_speed: number of past speed-error samples kept.
        k_steer: number of past steer- and yaw-error samples kept.
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

    @classmethod
    def reads_for(cls, *, params: dict) -> tuple[str, ...]:
        # signal-backed raw inputs; curvature comes from the (heavy) lookahead
        # geometry the "curvature_ahead" signal also fronts.
        return ("lateral", "half_width", "heading_err", "v_forward",
                "yaw_rate", "v_lateral", "actions", "curvature_ahead")

    # ---- per-step ------------------------------------------------------
    def compute(self) -> torch.Tensor:
        env = self.env
        v_forward = _read(env, "v_forward")
        yaw_rate_raw = _read(env, "yaw_rate")
        actions = _read(env, "actions")

        # -- CNN targets: what a camera could tell you ------------------
        lateral = _read(env, "lateral") * env.dir_sign / _read(env, "half_width").clamp(min=0.1)
        heading = _read(env, "heading_err") / math.pi
        speed = v_forward / MAX_SPEED
        yaw_rate = yaw_rate_raw / YAW_RATE_NORM
        beta = torch.atan2(_read(env, "v_lateral"),
                           v_forward.clamp(min=0.05)) / BETA_NORM
        kappa = env.track.curvature_ahead(
            env.progress_m, self.horizons, env.dir_sign) / CURVATURE_NORM

        # -- error channels vs the FIXED nominal bicycle ----------------
        act = env.cfg["action"]
        steer_cmd = actions[:, 0] * MAX_STEER_RAD
        # the commanded speed must follow the env's action cap (mdp.action_to_command),
        # not the physics constant; the constant stays the normalizer.
        speed_cmd = act["min_speed"] + (actions[:, 1] + 1) * 0.5 * (
            act["max_speed"] - act["min_speed"])
        yaw_expected = v_forward * torch.tan(steer_cmd) / NOMINAL_WHEELBASE
        a_lat_expected = v_forward * yaw_expected            # v^2 tan(d)/L
        a_lat_actual = v_forward * yaw_rate_raw

        speed_err = (speed_cmd - v_forward) / MAX_SPEED
        steer_err = ((a_lat_expected - a_lat_actual) / A_LAT_NORM).clamp(-2, 2)
        yaw_err = ((yaw_expected - yaw_rate_raw) / YAW_RATE_NORM).clamp(-2, 2)

        # -- roll the histories (newest first) --------------------------
        self._prev_actions = torch.cat(
            [actions.unsqueeze(1), self._prev_actions[:, :-1]], dim=1)
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


# ---------------------------------------------------------------------------
# Composable feature blocks: a feature vector is an ordered *selection* of these
# (Part K.2). Each block is one named, normalized group of channels; an
# experiment picks the blocks it wants via ``SelectFeatures`` (see the feature
# examples in ``examples/feature_vector.py``). The block computes read the same
# live env state ClassicFeatures does, so selecting every block in order
# reproduces the classic vector exactly.

from dataclasses import dataclass          # noqa: E402
from typing import Callable, Union         # noqa: E402


@dataclass(frozen=True)
class FeatureBlock:
    """One named group of feature channels.

    A block computes reads its raw inputs from the signal bus (``env.signals``)
    when a matching :data:`~deepracer_genesis.envs.signals.SIGNALS` entry exists
    — via :func:`_read` — so ``reads`` is a truthful, checkable feature->signal
    trace (Part K.2 / K.5). The normalization and any driving-frame
    (``dir_sign``) transform stay in the block: signals expose *raw* env values
    (see :data:`~deepracer_genesis.envs.signals.SIGNALS` frame metadata), the
    feature layer applies the transform. This keeps the vector byte-for-byte
    identical to :class:`ClassicFeatures` while making the bus non-dead.

    Attributes:
        width: Channel count — an int, or a fn of ``lookahead_k`` for
            lookahead-sized blocks.
        compute: ``env -> (N, width)`` tensor for the current step.
        layout: Human-readable channel label(s); ``{k}`` expands to lookahead_k.
        reads: Signal names this block reads (a feature->signal trace). Names
            that match a registered signal are read via ``env.signals``; the
            rest (e.g. composite geometry with no single signal) document the
            env attributes the block touches.
    """

    width: Union[int, Callable[[int], int]]
    compute: Callable[["object"], torch.Tensor]
    layout: str
    reads: tuple[str, ...] = ()


def _block_width(width, lookahead_k: int) -> int:
    return width(lookahead_k) if callable(width) else width


def _read(env, name: str):
    """Read a raw signal value: from the bus if present, else the env attr.

    The bus (``env.signals``) is the single source of truth once K.1 is wired;
    the env-attribute fallback keeps blocks usable on a bare env or a test fake
    that carries the attributes but no bus. Either path returns the same *raw*
    value, so feature numbers are unchanged.
    """
    bus = getattr(env, "signals", None)
    if bus is not None and name in getattr(bus, "registry", {}):
        return bus[name]
    return getattr(env, name)


def _lookahead_xy(env) -> torch.Tensor:
    """Body-frame (rel_x, rel_y) of the upcoming waypoints, (N, 2K)."""
    cy, sy = torch.cos(env.yaw), torch.sin(env.yaw)
    la_idx = env.track.lookahead(env.wp_idx, env.lookahead_k,
                                 env.cfg["obs"]["lookahead_stride"],
                                 dir_sign=env.dir_sign)
    la_pts = env.track.lookahead_points(la_idx)
    rel = la_pts - env.base_pos[:, None, :2]
    rel_x = rel[..., 0] * cy[:, None] + rel[..., 1] * sy[:, None]
    rel_y = -rel[..., 0] * sy[:, None] + rel[..., 1] * cy[:, None]
    la_scale = env.cfg["obs"]["lookahead_scale"]
    return torch.cat([rel_x / la_scale, rel_y / la_scale], dim=1)


#: the block vocabulary experiments select from (order in a selection matters).
#: Each block's ``reads`` names the signals it consumes; ``_read`` pulls those
#: raw values from ``env.signals`` when the bus is present (Part K.2). The
#: ``lateral`` block reads the *raw* ``lateral`` signal and applies ``dir_sign``
#: here — signals are raw, the driving-frame transform lives in the feature.
FEATURE_BLOCKS: dict[str, FeatureBlock] = {
    "v_forward": FeatureBlock(
        1, lambda e: (_read(e, "v_forward") / e.cfg["action"]["max_speed"]).unsqueeze(1),
        "v_forward/max_speed", reads=("v_forward",)),
    "v_lateral": FeatureBlock(
        1, lambda e: _read(e, "v_lateral").unsqueeze(1), "v_lateral",
        reads=("v_lateral",)),
    "yaw_rate": FeatureBlock(
        1, lambda e: (_read(e, "yaw_rate") / YAW_RATE_NORM).unsqueeze(1),
        "yaw_rate/norm", reads=("yaw_rate",)),
    "lateral": FeatureBlock(
        1, lambda e: (_read(e, "lateral") * e.dir_sign
                      / _read(e, "half_width").clamp(min=0.1)).unsqueeze(1),
        "lateral/half_width", reads=("lateral", "half_width")),
    "heading": FeatureBlock(
        2, lambda e: torch.cat([torch.sin(_read(e, "heading_err")).unsqueeze(1),
                                torch.cos(_read(e, "heading_err")).unsqueeze(1)], dim=1),
        "sin(heading_err), cos(heading_err)", reads=("heading_err",)),
    "last_action": FeatureBlock(
        2, lambda e: _read(e, "actions"), "last_action[2]", reads=("actions",)),
    # composite body-frame waypoint geometry: no single registered signal
    # corresponds to it, so reads=() (an honest empty trace).
    "lookahead_xy": FeatureBlock(
        lambda k: 2 * k, _lookahead_xy,
        "lookahead_rel_x[{k}]/scale, lookahead_rel_y[{k}]/scale"),
}


class SelectFeatures(FeatureSet):
    """A feature vector assembled from a chosen, ordered list of blocks.

    ``params['features']`` is a tuple of :data:`FEATURE_BLOCKS` names; the
    vector is their concatenation in that order. This is how an experiment
    chooses exactly which features it uses (Part K.2) without a bespoke
    ``FeatureSet`` subclass.

    Attributes:
        env: the live env the features are read from.
        params: must contain ``features`` (the ordered block names).
    """

    @staticmethod
    def _names(params: dict) -> tuple[str, ...]:
        names = tuple(params.get("features", ()))
        if not names:
            raise ValueError(
                "SelectFeatures needs params['features'] = a non-empty tuple of "
                f"block names from {sorted(FEATURE_BLOCKS)}")
        unknown = [n for n in names if n not in FEATURE_BLOCKS]
        if unknown:
            raise ValueError(
                f"unknown feature block(s) {unknown}; available: "
                f"{sorted(FEATURE_BLOCKS)}")
        return names

    @property
    def dim(self) -> int:
        return self.dim_for(lookahead_k=self.env.lookahead_k, params=self.params)

    @classmethod
    def dim_for(cls, *, lookahead_k: int, params: dict) -> int:
        return sum(_block_width(FEATURE_BLOCKS[n].width, lookahead_k)
                   for n in cls._names(params))

    @classmethod
    def layout_for(cls, *, lookahead_k: int, params: dict) -> str:
        return ", ".join(FEATURE_BLOCKS[n].layout.format(k=lookahead_k)
                         for n in cls._names(params))

    @classmethod
    def reads_for(cls, *, params: dict) -> tuple[str, ...]:
        """Order-preserving, de-duplicated union of the selected blocks' reads."""
        out: list[str] = []
        for n in cls._names(params):
            for s in FEATURE_BLOCKS[n].reads:
                if s not in out:
                    out.append(s)
        return tuple(out)

    def compute(self) -> torch.Tensor:
        return torch.cat([FEATURE_BLOCKS[n].compute(self.env)
                          for n in self._names(self.params)], dim=1)


def blocks_signal_set(names: "tuple[str, ...]") -> "set[str]":
    """Union of the signals the given feature blocks read (Part K.3/K.5).

    Args:
        names: feature-block names (must exist in :data:`FEATURE_BLOCKS`).

    Returns:
        The set of signal names those blocks recover (for the learnability check).
    """
    out: set[str] = set()
    for n in names:
        out.update(FEATURE_BLOCKS[n].reads)
    return out


class RoutedFeatures(FeatureSet):
    """Two feature vectors — actor-visible and critic-visible — over one base (K.3).

    ``params`` = ``{"base": (block names...), "actor": sel, "critic": sel}`` where
    each ``sel`` is a subset of ``base`` or the sentinel ``("*",)`` (= all base
    blocks). The routed vectors are assembled from :data:`FEATURE_BLOCKS`, so
    they carry the SAME numbers as the equivalent :class:`SelectFeatures`; the
    actor sees a (possibly strict) subset of what the critic sees — the
    vector-mode analogue of the privileged-critic vision setup. ``compute()``
    returns the critic vector (the superset the env's NaN guard and terminal
    snapshot run on).

    Attributes:
        env: the live env the features are read from.
        params: must contain ``base``; ``actor``/``critic`` default to all base.
    """

    def __init__(self, env, params: dict):
        super().__init__(env, params)
        base = tuple(params["base"])
        self._base = base
        self._actor = self._resolve(params.get("actor", ("*",)), base)
        self._critic = self._resolve(params.get("critic", ("*",)), base)

    @staticmethod
    def _resolve(sel: "tuple[str, ...]", base: "tuple[str, ...]") -> "tuple[str, ...]":
        """Resolve a selection to base order; ``("*",)`` = the whole base list."""
        sel = tuple(sel)
        if sel == ("*",):
            return base
        chosen = set(sel)
        return tuple(n for n in base if n in chosen)   # base order, de-duplicated

    def _widths(self, names: "tuple[str, ...]") -> int:
        return sum(_block_width(FEATURE_BLOCKS[n].width, self.env.lookahead_k)
                   for n in names)

    @property
    def actor_dim(self) -> int:
        return self._widths(self._actor)

    @property
    def critic_dim(self) -> int:
        return self._widths(self._critic)

    @property
    def dim(self) -> int:
        return self.critic_dim   # the env sizes state_buf to the critic superset

    def compute_actor(self) -> torch.Tensor:
        return torch.cat([FEATURE_BLOCKS[n].compute(self.env)
                          for n in self._actor], dim=1)

    def compute_critic(self) -> torch.Tensor:
        return torch.cat([FEATURE_BLOCKS[n].compute(self.env)
                          for n in self._critic], dim=1)

    def compute(self) -> torch.Tensor:
        return self.compute_critic()

    def reset(self, env_ids: torch.Tensor) -> None:
        pass   # FEATURE_BLOCKS are stateless — no per-env history to clear

    @classmethod
    def dim_for(cls, *, lookahead_k: int, params: dict) -> int:
        crit = cls._resolve(params.get("critic", ("*",)), tuple(params["base"]))
        return sum(_block_width(FEATURE_BLOCKS[n].width, lookahead_k) for n in crit)

    @classmethod
    def layout_for(cls, *, lookahead_k: int, params: dict) -> str:
        crit = cls._resolve(params.get("critic", ("*",)), tuple(params["base"]))
        return ", ".join(FEATURE_BLOCKS[n].layout.format(k=lookahead_k) for n in crit)

    @classmethod
    def reads_for(cls, *, params: dict) -> tuple[str, ...]:
        crit = cls._resolve(params.get("critic", ("*",)), tuple(params["base"]))
        out: list[str] = []
        for n in crit:
            for s in FEATURE_BLOCKS[n].reads:
                if s not in out:
                    out.append(s)
        return tuple(out)
