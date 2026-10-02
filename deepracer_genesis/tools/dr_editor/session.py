"""Live inspection sessions: a small DR-stripped scene + synchronized grabs.

The editor's live tier owns ONE small "inspection scene" (default 12 envs,
top-down + spectator declared up front — the view set is fixed at
``scene.build()``), built DR-STRIPPED so ``env.image_buf`` is truly raw; every
post-render stage is then replayed offline (:mod:`.pipeline`), and renderer-
side knobs are realized by live re-rolls (mount) or rebuild-time cfg updates
(env maps, physics, statics). Teleporting via ``car.set_qpos`` +
``_post_physics`` (the sweep-dataset recipe) pins every env to one exact pose,
so a grab is synchronized to a single sim instant by construction.

Genesis is imported lazily inside functions — importing this module is cheap
and safe on GPU-less machines (:func:`capabilities` reports what would run).
"""

from __future__ import annotations

import importlib.util
import os
from typing import Optional

import torch

from .pipeline import dr_from_spec, replay_stages
from .rng import seeded


def capabilities() -> dict:
    """Probe what the live tier can do on this machine.

    Returns:
        ``{"cuda": bool, "gs_madrona": bool, "gs_nyx": bool, "genesis": bool}``
        — the offline tier (banks, replay, sheets, emit) needs none of them.
    """
    return {
        "cuda": torch.cuda.is_available(),
        "genesis": importlib.util.find_spec("genesis") is not None,
        "gs_madrona": importlib.util.find_spec("gs_madrona") is not None,
        "gs_nyx": importlib.util.find_spec("gs_nyx") is not None,
    }


def set_dotted(cfg: dict, path: str, value) -> None:
    """Set one dotted-path key in a nested cfg dict in place.

    Args:
        cfg: The nested env-cfg dict.
        path: Dotted location, e.g. ``"vision.env_map"`` or
            ``"rand.friction_range"``.
        value: The value to write.
    """
    node = cfg
    parts = path.split(".")
    for p in parts[:-1]:
        node = node[p]
    node[parts[-1]] = value


class EditorSession:
    """One live inspection scene plus the DR params being inspected.

    Attributes:
        env: The built ``DeepRacerEnv`` (DR-stripped unless ``cfg_updates``
            deliberately enabled a rebuild-class knob).
        dr: DR parameters in spec shape, used by the offline stage replay
            (empty when inspecting a bare scene).
        spec: The originating ``ExperimentSpec``, when built from a target.
        num_envs: Parallel env count of the inspection scene.
        settle_steps: Scene steps run after a teleport before frames count
            (Nyx accumulates temporally; 0 elsewhere — the teleport recipe
            needs no dynamics).
    """

    def __init__(self, env_cfg: dict, num_envs: int, *, dr: Optional[dict] = None,
                 spec=None) -> None:
        """Build the inspection scene from a ready env cfg.

        Args:
            env_cfg: The nested env config (spawn randomization should already
                be off; see the classmethod constructors).
            num_envs: Parallel env count.
            dr: DR parameters in spec shape for the offline replay (default
                empty).
            spec: Originating spec, kept for emit/round-trip context.
        """
        # a small inspection scene needs nowhere near Madrona's default 4 GiB
        # device-malloc heap; on a busy shared GPU that reservation fails the
        # first kernel launch (an explicit env var still wins)
        os.environ.setdefault("MADRONA_MWGPU_DEVICE_HEAP_SIZE", str(1 << 30))
        from ..._gs import ensure_init
        from ...envs import DeepRacerEnv

        ensure_init(env_cfg["sim"].get("backend", "gpu"))
        self.env = DeepRacerEnv(num_envs=num_envs, env_cfg=env_cfg)
        self.dr = dict(dr or {})
        self.spec = spec
        self.num_envs = num_envs
        renderer = env_cfg["vision"].get("vision_renderer", "batch")
        self.settle_steps = 6 if renderer == "nyx" else 0
        ids = torch.arange(num_envs, device=self.env.device)
        self.env.reset_idx(ids)
        self.refresh()

    # ------------------------------------------------------------ constructors
    @classmethod
    def from_defaults(cls, *, track: str = "reinvent_base",
                      renderer: str = "batch", num_envs: int = 12,
                      res: tuple = (160, 120), dr: Optional[dict] = None,
                      cfg_updates: Optional[dict] = None,
                      spectator: bool = True) -> "EditorSession":
        """Build a bare inspection scene (no experiment needed).

        Args:
            track: Track name, or comma-joined names for tiled variants.
            renderer: ``"batch"`` (Madrona), ``"nyx"``, or ``"rasterizer"``.
            num_envs: Parallel env count (12 keeps every grab interactive).
            res: Camera resolution ``(W, H)``.
            dr: DR parameters for the offline replay (default none).
            cfg_updates: Dotted-path cfg overrides applied before the build —
                the hook rebuild-class knobs use (e.g.
                ``{"vision.env_map": {...}}``).
            spectator: Declare the spectator camera at build.

        Returns:
            The built session.
        """
        from ...configs.cfgs import get_env_cfg

        names = [t.strip() for t in track.split(",")] if "," in track else track
        cfg = get_env_cfg(vision=True, track=names, topdown=True)
        cfg["vision"]["camera_res"] = tuple(res)
        if renderer in ("nyx", "rasterizer"):
            cfg["vision"]["vision_renderer"] = renderer
        cfg["vision"]["spectator"] = bool(spectator)
        cfg["spawn"]["random_start"] = False
        cfg["spawn"]["spawn_lateral_noise"] = 0.0
        cfg["spawn"]["spawn_yaw_noise"] = 0.0
        for path, value in (cfg_updates or {}).items():
            set_dotted(cfg, path, value)
        return cls(cfg, num_envs, dr=dr)

    @classmethod
    def from_target(cls, target: str, *, num_envs: int = 12,
                    cfg_updates: Optional[dict] = None,
                    spectator: bool = True) -> "EditorSession":
        """Build the inspection scene for an experiment's own config.

        The experiment's DR is carried as the session's replay params while
        the LIVE env is built from the DR-STRIPPED spec (so ``image_buf`` is
        truly raw and every stage can be shown explicitly).

        Args:
            target: ``module:ClassName`` path of an Experiment (the
                ``python -m deepracer_genesis.experiment`` convention).
            num_envs: Inspection env count (overrides the spec's).
            cfg_updates: Dotted-path cfg overrides applied after sim_cfg.
            spectator: Declare the spectator camera at build.

        Returns:
            The built session, with ``session.dr``/``session.spec`` set.
        """
        from dataclasses import replace

        from ...experiment.builder import Builder
        from ...experiment.run import build
        from ...experiment.spec import ActionDRSpec, ObsDRSpec

        module, _, clsname = target.partition(":")
        import importlib
        import sys
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        spec = build(getattr(importlib.import_module(module), clsname))
        dr = dr_from_spec(spec)
        stripped = replace(spec, obs_dr=ObsDRSpec(), action_dr=ActionDRSpec())
        cfg = Builder(stripped).sim_cfg()
        cfg["vision"]["topdown_camera"] = True
        cfg["vision"]["spectator"] = bool(spectator)
        cfg["spawn"]["random_start"] = False
        cfg["spawn"]["spawn_lateral_noise"] = 0.0
        cfg["spawn"]["spawn_yaw_noise"] = 0.0
        for path, value in (cfg_updates or {}).items():
            set_dotted(cfg, path, value)
        return cls(cfg, num_envs, dr=dr, spec=spec)

    # ---------------------------------------------------------------- geometry
    def teleport(self, waypoint: int = 5, lateral_frac: float = 0.0,
                 yaw_off: float = 0.0) -> None:
        """Park every car at one exact pose of its own track variant.

        Places each env at waypoint ``waypoint`` of its variant (tile offsets
        are already baked into the padded centerlines), offset laterally as a
        fraction of the local half-width, facing the tangent plus ``yaw_off``
        — then refreshes state and cameras with NO dynamics stepping (the
        proven sweep-dataset recipe).

        Args:
            waypoint: Waypoint index shared by all envs.
            lateral_frac: Signed lateral offset as a fraction of the local
                half-width (0 = centerline, 1 = at the rule boundary).
            yaw_off: Heading offset in radians from the track tangent.
        """
        env, mt = self.env, self.env.track
        ev = mt.variant_idx
        wp = waypoint % int(mt.n_wps_v.min().item())
        yaw = mt.track_yaw[ev, wp] + yaw_off
        lateral_m = mt.half_width[ev, wp] * lateral_frac
        qpos = torch.zeros(env.num_envs, 13, device=env.device)
        qpos[:, 0:2] = mt.center[ev, wp] + mt.normal[ev, wp] * lateral_m[:, None]
        qpos[:, 2] = env.cfg["spawn"]["spawn_height"]
        qpos[:, 3] = torch.cos(yaw / 2)
        qpos[:, 6] = torch.sin(yaw / 2)
        env.car.set_qpos(qpos)
        self.refresh()

    def refresh(self) -> None:
        """Re-render the current pose (with the renderer's settle policy).

        Runs the Nyx temporal-settle steps (no-op elsewhere), then
        ``_post_physics`` to refresh the localizer, signals, and all cameras
        at ONE sim instant.
        """
        for _ in range(self.settle_steps):
            self.env.scene.step()
        ids = torch.arange(self.num_envs, device=self.env.device)
        self.env._post_physics(ids)

    # ------------------------------------------------------------------- views
    def raw(self) -> torch.Tensor:
        """The truly-raw onboard frames of the current instant.

        Returns:
            ``(N, 3, H, W)`` float frames in [0, 1] (DR-stripped render —
            no world colour, no pixel noise, no aug).
        """
        return self.env.image_buf.clone()

    def stages(self, *, dr: Optional[dict] = None, seed: int = 0,
               frame_stack: Optional[int] = None) -> dict[str, torch.Tensor]:
        """Replay every pipeline stage on the current instant's raw frames.

        Args:
            dr: DR parameters (defaults to the session's own).
            seed: Seed for every random draw.
            frame_stack: Stack depth (defaults to the env's cfg value).

        Returns:
            ``{stage: (N, ...) tensor}`` for all seven pipeline stages.
        """
        vision = self.env.cfg["vision"]
        return replay_stages(
            self.raw(), dr if dr is not None else self.dr,
            policy_res=vision.get("policy_res"),
            frame_stack=(frame_stack if frame_stack is not None
                         else vision.get("frame_stack", 1)),
            seed=seed)

    def topdown(self) -> torch.Tensor:
        """Per-env bird's-eye frames of the current instant.

        Returns:
            ``(N, H, W, 3)`` uint8 frames (same instant as the last
            ``refresh`` — the batch renderer caches by sim tick).
        """
        return self.env.render_topdown()

    def spectator(self):
        """One all-cars spectator frame of the current instant.

        Returns:
            ``(H, W, 3)`` uint8 array.

        Raises:
            AssertionError: If the session was built with ``spectator=False``.
        """
        return self.env.render_spectator()

    # -------------------------------------------------------------- live pokes
    def reroll_mount(self, pitch_deg: float, pos_m: float, *,
                     seed: int = 0) -> None:
        """Re-roll the per-env camera mounts and re-render (reroll-class knob).

        Writes the jitter magnitudes into the live cfg and re-invokes the
        renderer's ``randomize_mount`` (Madrona/rasterizer; a declared no-op
        on Nyx) under a seeded fork, then refreshes the frames.

        Args:
            pitch_deg: Mount pitch jitter magnitude (deg); 0 disables.
            pos_m: Mount position jitter magnitude (m); 0 disables.
            seed: Seed for the mount draw.
        """
        env = self.env
        env.cfg["rand"]["camera_pitch_jitter_deg"] = float(pitch_deg)
        env.cfg["rand"]["camera_pos_jitter_m"] = float(pos_m)
        ids = torch.arange(self.num_envs, device=env.device)
        with seeded(seed, env.device):
            env.renderer.randomize_mount(env, ids)
        # the batch renderer caches frames by scene.t; a mount change alone
        # does not advance the sim clock, so force one tick or the re-render
        # would return the cached (pre-jitter) frames
        env.scene.step()
        self.refresh()

    def poke_action_dr(self, steer_noise: float = 0.0,
                       speed_noise: float = 0.0) -> None:
        """Enable/disable env-side action noise on the live env.

        ``delay_steps`` is deliberately NOT pokable: its ring buffer is sized
        at env construction — build the session with
        ``cfg_updates={"action_dr": {...}}`` for delays.

        Args:
            steer_noise: Gaussian steering-command noise scale.
            speed_noise: Gaussian speed-command noise scale.
        """
        if steer_noise or speed_noise:
            self.env.action_dr = {"steer_noise": float(steer_noise),
                                  "speed_noise": float(speed_noise),
                                  "delay_steps": 0}
        else:
            self.env.action_dr = {}
