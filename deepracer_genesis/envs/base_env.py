"""Define the batched DeepRacer environment on Genesis (shared base).

``DeepRacerEnv`` is both base class and factory, dispatching by ``env_cfg["vision"]``.
"""

from __future__ import annotations

import math
import os

import torch
from tensordict import TensorDict

import genesis as gs

from . import mdp, rules
from .scene import build_scene
from .track import MultiTrack
from ..randomization.physics import randomize_armature, randomize_physics


class DeepRacerEnv:
    """Batched DeepRacer sim owning the control/observation loop, and factory
    returning a vision or vector subclass per ``env_cfg["vision"]``.

    Attributes:
        device: Torch device the sim and all buffers live on.
        cfg: Environment config dict driving every subsystem.
        num_envs: Count of parallel environments.
        num_actions: Dimensionality of the continuous action space.
        vision: Whether this env emits camera observations.
        action_table: Discrete steer/speed lookup table, or None for continuous.
        dt: Control timestep (physics dt times decimation).
        max_episode_length: Step budget before a truncation timeout.
        track: Multi-track geometry and localization helper.
        renderer: Injected renderer owning cameras and appearance state.
        scene: Genesis scene handle.
        car: The car entity wrapper.
        steer_dofs: Steering degree-of-freedom handles mirrored from the car.
        wheel_dofs: Wheel degree-of-freedom handles mirrored from the car.
        wheel_radius: Wheel radius mirrored from the car.
        steering_model: Steering kinematics model mirrored from the car.
        episode_length_buf: Per-env step counter within the current episode.
        reset_buf: Per-env done flags for the current step.
        rew_buf: Per-env reward for the current step.
        time_out_buf: Per-env truncation flags.
        actions: Current control actions applied this step.
        last_actions: Actions applied on the previous step.
        progress_m: Per-env arc-length progress along the track.
        laps: Per-env completed-lap count.
        dir_sign: Per-env driving direction (+1 forward, -1 reversed).
        offtrack_buf: Per-env off-track flags.
        flipped_buf: Per-env flipped-over flags.
        emit_cost: Whether a separate cost stream is produced.
        cost_fn: Name of the cost term when costs are emitted.
        cost_buf: Per-env cost for the current step (when emitting cost).
        cost_episode_sum: Accumulated per-episode cost (when emitting cost).
        extras: Auxiliary step outputs including per-episode logs.
        lookahead_k: Number of lookahead waypoints in the state vector.
        num_state_obs: Length of the state observation vector.
        state_buf: Current per-env state observation.
        reward_terms: Reward function producing the shaping terms.
        reward_scales: Weights applied to each reward term.
        episode_sums: Running per-episode sums of each reward term.
        base_pos: Cached car position each step.
        yaw: Cached car heading each step.
        v_forward: Forward speed in the car frame.
        v_lateral: Lateral speed in the car frame.
        yaw_rate: Yaw angular velocity.
        up_z: Body up-axis z component used for flip detection.
        wp_idx: Nearest-waypoint index from localization.
        lateral: Signed lateral offset from the track centerline.
        half_width: Local track half-width.
        heading_err: Heading error relative to the track tangent.
        d_progress: Signed progress delta this step.
        step_info: Pre-reset snapshot of terminal per-step quantities.
    """

    def __new__(cls, num_envs: int, env_cfg: dict, show_viewer: bool = False,
                device=None):
        """Dispatch construction to the vision or vector subclass.

        Args:
            num_envs: Number of parallel environments to allocate.
            env_cfg: Environment config dict; ``env_cfg["vision"]`` selects the
                concrete subclass.
            show_viewer: Whether to open the interactive Genesis viewer.
            device: Torch device for the sim; defaults to the Genesis device.

        Returns:
            An uninitialized instance of the vision or vector subclass when
            called on the base class, or of ``cls`` itself when called on a
            subclass.
        """
        # constructing the base dispatches to the vision/vector subclass
        if cls is DeepRacerEnv:
            from .vector_env import VectorDeepRacerEnv
            from .vision_env import VisionDeepRacerEnv
            cls = VisionDeepRacerEnv if env_cfg["vision"]["vision"] else VectorDeepRacerEnv
        return object.__new__(cls)

    def __init__(self, num_envs: int, env_cfg: dict, show_viewer: bool = False,
                 device=None) -> None:
        """Build the scene, car, track, renderer, and per-env buffers.

        Resolves timestep/episode length and allocates state via :meth:`_init_buffers`.

        Args:
            num_envs: Number of parallel environments.
            env_cfg: Environment config dict (timestep, decimation, track name(s),
                camera/lookahead/reward settings, optional ``action_table``,
                domain-randomization flags, and so on).
            show_viewer: Whether to open the interactive Genesis viewer.
            device: Torch device for the sim; defaults to the Genesis device.
        """
        self.device = torch.device(device) if device is not None else gs.device
        self.cfg = env_cfg
        self.num_envs = num_envs
        self.num_actions = 2
        self.vision = env_cfg["vision"]["vision"]
        # Opt-in coarse fence (DRG_STEP_SYNC=1): one device sync per control step
        # after the physics kernels, before torch (PPO) proceeds. Drains in-flight
        # quadrants kernels so its async free never overlaps them — mitigates the
        # sporadic CUDA_ERROR_ILLEGAL_ADDRESS in cuMemFreeAsync (a timing race in
        # quadrants' async allocator; lever-confirmed that synchronization fixes
        # it). Far cheaper than PYTORCH_NO_CUDA_MEMORY_CACHING (per-free sync).
        self._step_sync = (os.environ.get("DRG_STEP_SYNC") == "1"
                           and torch.cuda.is_available())
        # action-space DR (env-side; the rsl-rl backend has no transform pipeline):
        # k-step command latency then per-channel gaussian noise. Off by default.
        self.action_dr = dict(env_cfg.get("action_dr", {}))
        self._act_delay = int(self.action_dr.get("delay_steps", 0))
        # image-space DR config (applied on the camera obs by the vision subclass)
        self.image_aug = dict(env_cfg.get("image_aug", {}))
        # discrete action support: a (K, 2) table of [steer, speed] pairs.
        # step() accepts index tensors (N,) and looks them up — every consumer
        # (training, eval, collection) stays agnostic.
        table = env_cfg["action"]["action_table"]
        self.action_table = (torch.tensor(table, dtype=torch.float32,
                                          device=self.device)
                             if table else None)

        sim = env_cfg["sim"]
        self.dt = sim["dt"] * sim["decimation"]  # control dt
        self.max_episode_length = math.ceil(sim["episode_length_s"] / self.dt)

        track = sim["track"]
        names = track if isinstance(track, (list, tuple)) else [track]
        # Part O: camera multi-track needs spatial tiling (each variant on its
        # own world tile) so an env's camera sees only its home track. Feature
        # and single-track envs get no offset (spacing 0), so they are unchanged.
        tiling = self.vision and len(names) > 1
        grid_spacing = sim.get("track_grid_spacing", 100.0) if tiling else 0.0
        self.track = MultiTrack(names, num_envs, self.device, grid_spacing=grid_spacing)

        # view="gui" opens the interactive Genesis viewer window (Part M); the
        # explicit show_viewer arg still works and either one enables it.
        self.view = sim.get("view", "none")
        show_viewer = show_viewer or self.view == "gui"
        build_scene(self, env_cfg, show_viewer)      # -> self.renderer, scene, car, track_entity
        self.car.configure(env_cfg["car"], self.device)
        # mirror the car's dof handles on the env (domain randomization reads them)
        self.steer_dofs = self.car.steer_dofs
        self.wheel_dofs = self.car.wheel_dofs
        self.wheel_radius = self.car.wheel_radius
        self.steering_model = self.car.steering_model
        self.renderer.finalize(self, env_cfg["vision"])   # attach cameras, appearance/obs state
        self._init_buffers(env_cfg)

    # ------------------------------------------------------------------ hooks
    def _init_obs_buffers(self, env_cfg: dict) -> None:
        """Preallocate camera image buffers (subclass hook, vision only).

        No-op on the base env; vision subclasses override it to size image tensors.

        Args:
            env_cfg: Environment config dict (camera resolution, etc.).
        """

    def _observe_camera(self) -> None:
        """Refresh the camera observation (subclass hook, vision only).

        No-op on the base env; vision subclasses render the current frame here.
        """

    def _finalize_obs(self) -> None:
        """Apply once-per-step observation transforms (subclass hook).

        No-op on the base env. Called exactly once per :meth:`step`, after any
        auto-reset re-render, so vision subclasses can apply stateful obs DR
        (camera latency / frame drop) without double-advancing on reset steps.
        """

    def _reset_obs_dr(self, env_ids: torch.Tensor) -> None:
        """Clear stateful observation DR for respawned envs (subclass hook).

        No-op on the base env; vision subclasses reset per-env camera-latency
        history so a fresh episode does not observe a pre-reset frame.

        Args:
            env_ids: Indices of the envs that were just respawned.
        """

    def _obs_groups(self) -> dict:
        """Assemble the observation groups exposed by :meth:`get_observations`.

        The base emits ``state`` (or, under Part K.3 per-signal routing, the two
        vectors ``obs_actor``/``obs_critic``); vision subclasses add ``camera``.

        Returns:
            Mapping from observation-group name to its tensor.
        """
        if self._routed:
            return {"obs_actor": self.obs_actor_buf, "obs_critic": self.obs_critic_buf}
        return {"state": self.state_buf}

    @property
    def spec_cam(self):
        """Return the spectator debug camera, or ``None`` if unset.

        Forwards to the injected renderer so callers need not reach through it.

        Returns:
            The spectator camera handle, or ``None`` when no such camera was
            attached.
        """
        return self.renderer.spec_cam

    @property
    def camera_stack(self) -> "torch.Tensor | None":
        """Return the stacked camera frames the policy observes, or ``None``.

        Always ``None`` here; vision envs that stack frames override it.

        Returns:
            ``None`` on an env without stacked camera observations.
        """
        return None

    # ---------------------------------------------------------------- buffers
    def _init_buffers(self, env_cfg: dict) -> None:
        """Allocate per-env episode state and resolve the reward function.

        Sizes the ``(N, num_state_obs)`` state vector and reward/termination buffers.

        Args:
            env_cfg: Environment config dict (``lookahead_k``, ``reward_fn`` and
                ``reward_scales``/``reward_scale_overrides``, ``emit_cost``,
                ``cost_fn``, etc.).

        Raises:
            ValueError: If a custom reward function is supplied without explicit
                reward scales.
        """
        N = self.num_envs
        self.episode_length_buf = torch.zeros(N, device=self.device, dtype=torch.long)
        self.reset_buf = torch.ones(N, device=self.device, dtype=torch.bool)
        self.rew_buf = torch.zeros(N, device=self.device)
        self.time_out_buf = torch.zeros(N, device=self.device, dtype=torch.bool)
        self.actions = torch.zeros(N, 2, device=self.device)
        self.last_actions = torch.zeros(N, 2, device=self.device)
        if self._act_delay > 0:      # ring buffer of the last k commands (action DR)
            self._act_buf = torch.zeros(N, self._act_delay, 2, device=self.device)
        self.progress_m = torch.zeros(N, device=self.device)
        self.laps = torch.zeros(N, device=self.device)
        # per-env driving direction: +1 follows waypoint order (counter-
        # clockwise on re:Invent tracks), -1 drives the track reversed.
        self.dir_sign = torch.ones(N, device=self.device)
        # per-env track-width DR (feature mode): scales the rulebook half_width
        # (off_track / lateral norm / centered reward) without touching the mesh;
        # redrawn each reset. 1.0 = nominal. Torch-only, so no genesis-setter cost.
        self.track_width_factor = torch.ones(N, device=self.device)
        self.offtrack_buf = torch.zeros(N, device=self.device, dtype=torch.bool)
        self.flipped_buf = torch.zeros(N, device=self.device, dtype=torch.bool)
        self.emit_cost = bool(env_cfg["reward"]["emit_cost"])
        self.cost_fn = env_cfg["reward"]["cost_fn"] or "offtrack"
        if self.emit_cost:
            self.cost_buf = torch.zeros(N, device=self.device)
            self.cost_episode_sum = torch.zeros(N, device=self.device)
        self.extras = {"log": {}}

        from .features import resolve_feature_set
        obs_cfg = env_cfg["obs"]
        self.lookahead_k = obs_cfg["lookahead_k"]
        # Part K.3: per-signal actor/critic routing emits two vectors
        # (obs_actor/obs_critic) instead of one "state"; None keeps the single
        # vector (byte-identical default path).
        routing = obs_cfg.get("obs_routing")
        self._routed = routing is not None
        if self._routed:
            from .features import RoutedFeatures
            self.feature_set = RoutedFeatures(self, dict(routing))
            self.num_state_obs = self.feature_set.dim          # critic superset
            self.obs_actor_buf = torch.zeros(
                N, self.feature_set.actor_dim, device=self.device)
            self.obs_critic_buf = torch.zeros(
                N, self.feature_set.critic_dim, device=self.device)
            self.state_buf = self.obs_critic_buf               # NaN guard/snapshot
        else:
            self.feature_set = resolve_feature_set(obs_cfg["feature_set"])(
                self, dict(obs_cfg["feature_params"]))
            self.num_state_obs = self.feature_set.dim
            self.state_buf = torch.zeros(N, self.num_state_obs, device=self.device)
        self._init_obs_buffers(env_cfg)

        # Shared lazy signal bus (Part K.1): populated per step, read by any
        # consumer (feature vector / reward / cost). Invalidated in
        # _post_physics; nothing pays for a signal it never reads.
        from .signals import SignalBus
        self.signals = SignalBus(self)

        from .rewards import deepracer
        reward_cfg = env_cfg["reward"]
        reward_fn = reward_cfg["reward"] or deepracer   # None -> default
        self.reward_terms = reward_fn
        overrides = reward_cfg["reward_scale_overrides"]
        if reward_fn is deepracer:
            # tweaks merge over the tuned defaults
            self.reward_scales = dict(reward_cfg["reward_scales"])
            self.reward_scales.update(overrides)
        else:
            # custom fn: its scales stand alone (defaults reference terms the
            # custom fn does not produce)
            if not overrides:
                name = getattr(reward_fn, "__qualname__", reward_fn)
                raise ValueError(
                    f"custom reward fn {name!r} needs explicit scales "
                    "(RewardShaping(fn=..., scales={...}))")
            self.reward_scales = dict(overrides)
        if self.emit_cost:
            # the cost stream replaces the offtrack shaping term (plan: pull
            # offtrack/crash OUT of the reward, constrain them instead)
            self.reward_scales.pop("off_track", None)
        self.episode_sums = {k: torch.zeros(N, device=self.device) for k in self.reward_scales}

        self.reset_idx(torch.arange(N, device=self.device))
        # Physics DR applied ONCE per run (static per-env bodies), not per reset.
        # EMPIRICAL: on genesis 1.2.3 + quadrants 1.1.1, per-episode physics DR
        # still sporadically crashes with CUDA_ERROR_ILLEGAL_ADDRESS (2/3 GPU
        # runs died at iter 21 / iter 571), whereas static (build-time) DR runs
        # clean past the danger zone — so DR is applied here, once. Each env
        # keeps its own randomized friction/mass/COM/gains/armature + camera
        # mount for the run; only spawn/direction (state) randomizes per reset.
        if self.cfg["rand"]["randomize"]:
            all_ids = torch.arange(N, device=self.device)
            randomize_physics(self, all_ids)
            randomize_armature(self, all_ids)
            self.renderer.randomize_mount(self, all_ids)
        self._post_physics()

    # ------------------------------------------------------------------- step
    def _apply_action_dr(self, a: torch.Tensor) -> torch.Tensor:
        """Env-side action DR: k-step latency then per-channel gaussian noise.

        Args:
            a: The clipped ``(N, 2)`` action tensor in [-1, 1].

        Returns:
            The delayed, noised action, clamped to [-1, 1].
        """
        dr = self.action_dr
        if self._act_delay > 0:
            out = self._act_buf[:, -1].clone()
            self._act_buf.copy_(
                torch.cat([a.unsqueeze(1), self._act_buf[:, :-1]], dim=1))
            a = out
        sn, pn = dr.get("steer_noise", 0.0), dr.get("speed_noise", 0.0)
        if sn or pn:
            noise = torch.stack([
                torch.randn(a.shape[0], device=a.device) * sn,
                torch.randn(a.shape[0], device=a.device) * pn], dim=1)
            a = a + noise
        return a.clamp(-1.0, 1.0)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Advance every env by one control step (``decimation`` physics steps).

        Maps actions to limits, steps physics, recomputes obs/reward/termination.

        Args:
            actions: Either ``(N, 2)`` normalized ``[steering, throttle]`` in
                ``[-1, 1]`` or ``(N,)`` action-table indices for discrete policies.

        Returns:
            A tuple ``(obs_tensordict, reward, done, extras)`` where ``reward`` and
            ``done`` are ``(N,)`` tensors. ``extras["log"]`` carries per-episode
            stats at reset boundaries and ``extras["time_outs"]`` flags
            truncations.
        """
        if self.action_table is not None and actions.dim() == 1:
            actions = self.action_table[actions.long()]
        self.actions = torch.clip(actions, -1.0, 1.0)
        if self.action_dr:
            self.actions = self._apply_action_dr(self.actions)
        steer, speed = mdp.map_action(self)
        self.car.drive(steer, speed)
        for _ in range(self.cfg["sim"]["decimation"]):
            self.scene.step()
        if self._step_sync:
            torch.cuda.synchronize()   # coarse fence (see __init__); opt-in

        self.episode_length_buf += 1
        self._post_physics()
        mdp.compute_reward(self)
        mdp.check_termination(self)

        # pre-reset snapshot: reset_idx destroys these for done rows, but
        # wrappers/evaluators need the values of the step that just happened
        self.step_info = {
            "progress_delta": self.d_progress.clone(),
            "laps": self.laps.clone(),
            "offtrack": self.offtrack_buf.clone(),
            "flipped": self.flipped_buf.clone(),
            "time_out": self.time_out_buf.clone(),
            "terminal_state": self.state_buf.clone(),
        }
        if self.emit_cost:
            self.step_info["cost"] = self.cost_buf.clone()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
            self._post_physics(env_ids)

        self.last_actions[:] = self.actions
        self._finalize_obs()   # once-per-step obs hook (e.g. camera latency DR)
        self.extras["time_outs"] = self.time_out_buf
        # No genesis<->torch stream fence needed: quadrants and torch both run on
        # the legacy NULL stream (stream 0), so allocator reuse is stream-ordered
        # by construction. The sporadic illegal-access came from quadrants 1.0.2
        # allocator bugs, fixed in >=1.2.3 — not a cross-stream race.
        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    # ---------------------------------------------------------- post-physics
    def _post_physics(self, env_ids: torch.Tensor | None = None) -> None:
        """Refresh cached kinematics, track-frame quantities, and observations.

        Reads pose/velocity, localizes the car, and rebuilds the state vector.

        Args:
            env_ids: Envs that were just reset this step; their ``d_progress`` is
                zeroed so a respawn does not register as forward progress. ``None``
                updates all envs without that correction.
        """
        self.signals.invalidate()   # new step: drop the per-step signal cache
        pos, quat, vel, ang = self.car.kinematics()

        self.base_pos = pos
        self.yaw = rules.yaw_from_quat(quat)
        cy, sy = torch.cos(self.yaw), torch.sin(self.yaw)
        self.v_forward = vel[:, 0] * cy + vel[:, 1] * sy
        self.v_lateral = -vel[:, 0] * sy + vel[:, 1] * cy
        self.yaw_rate = ang[:, 2]
        self.up_z = rules.up_z_from_quat(quat)   # flip detection

        loc = self.track.localize(pos[:, :2])
        self.wp_idx = loc["wp_idx"]
        self.lateral = loc["lateral"]
        # track-width DR: one per-env scalar changes the effective road width
        # everywhere the rulebook reads it (off_track, lateral/half_width,
        # centered reward). 1.0 when the knob is off.
        self.half_width = loc["half_width"] * self.track_width_factor
        # all track-frame quantities are expressed in the env's own driving
        # direction (dir_sign): a reversed car aligned with the reversed tangent
        # has heading_err 0 and accumulates positive progress
        rev = (self.dir_sign < 0).float()
        self.heading_err = rules.wrap(self.yaw - loc["track_yaw"] - rev * math.pi)
        new_progress = loc["progress_m"]
        # signed per-step progress: lap-wrapped, direction-oriented, and clamped
        # to the physical per-step bound (kills spurious progress from localize
        # nearest-waypoint jumps at track pinches — see rules.lap_progress).
        max_step = 1.5 * self.cfg["action"]["max_speed"] * self.dt
        self.d_progress, crossed = rules.lap_progress(
            self.progress_m, new_progress, self.track.total_len_env,
            self.dir_sign, max_step)
        if env_ids is not None and len(env_ids) > 0:
            self.d_progress[env_ids] = 0.0
        # wrap through the finish line while moving forward = one lap completed
        self.laps += (crossed & (self.d_progress > 0)).float()
        self.progress_m = new_progress

        # ---- state obs (assembled by the selected feature set) ----
        obs_noise = self.cfg["obs"]["obs_noise"]
        if self._routed:
            # Part K.3: assemble both vectors; the critic superset backs the NaN
            # guard + terminal snapshot (state_buf), obs_noise perturbs the ACTOR
            # copy only (privileged clean critic — the value-of-asymmetry point).
            self.obs_actor_buf = self.feature_set.compute_actor()
            self.obs_critic_buf = self.feature_set.compute_critic()
            self.state_buf = self.obs_critic_buf
            if obs_noise > 0:
                self.obs_actor_buf = (self.obs_actor_buf
                                      + torch.randn_like(self.obs_actor_buf) * obs_noise)
        else:
            self.state_buf = self.feature_set.compute()
            if obs_noise > 0:
                self.state_buf += torch.randn_like(self.state_buf) * obs_noise

        self._observe_camera()

    # ------------------------------------------------------------------ reset
    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """Respawn the given envs and resample their per-episode DR draws.

        Spawns at random waypoints, redraws randomizations, and clears counters.

        Args:
            env_ids: Indices of the envs to respawn. An empty tensor is a no-op.
        """
        n = len(env_ids)
        if n == 0:
            return
        spawn = self.cfg["spawn"]
        pos_xy, yaw = self.track.spawn_pose(
            env_ids, spawn["random_start"],
            lateral_noise=spawn["spawn_lateral_noise"], yaw_noise=spawn["spawn_yaw_noise"])
        if spawn["random_direction"]:
            # coin-flip the driving direction each episode; the spawn faces the
            # chosen direction and all track-frame quantities follow it
            flip = torch.rand(n, device=self.device) < 0.5
            self.dir_sign[env_ids] = torch.where(flip, -1.0, 1.0)
            yaw = yaw + flip.float() * math.pi

        # track-width DR (feature mode only): the rendered mesh width is fixed,
        # so scaling only the rulebook width would desync the camera view — scope
        # it to feature envs (torch-only; no crash-prone genesis setter).
        if self.cfg["rand"]["randomize"] and not self.vision:
            lo, hi = self.cfg["rand"].get("track_width_scale", (1.0, 1.0))
            if (lo, hi) != (1.0, 1.0):
                self.track_width_factor[env_ids] = (
                    lo + (hi - lo) * torch.rand(n, device=self.device))

        self.renderer.resample_appearance(env_ids)   # world-color DR (vision only)

        qpos = torch.zeros(n, 13, device=self.device)
        qpos[:, 0:2] = pos_xy
        qpos[:, 2] = spawn["spawn_height"]
        qpos[:, 3] = torch.cos(yaw / 2)
        qpos[:, 6] = torch.sin(yaw / 2)
        self.car.reset_pose(qpos, env_ids)
        # NOTE: physics DR is applied once at build (see __init__), not per reset
        # — per-episode physics setters sporadically crash even on genesis 1.2.3.
        # World-color DR above is torch-only, so it stays per reset.

        # episode logging
        self.extras["log"] = {}
        for key, sums in self.episode_sums.items():
            self.extras["log"][f"Episode/rew_{key}"] = sums[env_ids].mean()
            sums[env_ids] = 0.0
        self.extras["log"]["Episode/length"] = self.episode_length_buf[env_ids].float().mean()
        if self.emit_cost:
            self.extras["log"]["Episode/cost"] = self.cost_episode_sum[env_ids].mean()
            self.cost_episode_sum[env_ids] = 0.0

        self.episode_length_buf[env_ids] = 0
        self.laps[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        if self._act_delay > 0:              # clear delayed commands on reset
            self._act_buf[env_ids] = 0.0
        self.feature_set.reset(env_ids)   # clear any per-env feature history
        self._reset_obs_dr(env_ids)       # clear stateful obs DR (camera latency)
        self.progress_m[env_ids] = self.track.localize(pos_xy, envs_idx=env_ids)["progress_m"]

    # ----------------------------------------------------------- observations
    def get_observations(self) -> TensorDict:
        """Assemble the current observation as a TensorDict.

        Wraps the groups from :meth:`_obs_groups` with the batch size and device.

        Returns:
            A TensorDict of batch size ``[num_envs]`` holding the ``state`` (and,
            on vision envs, ``camera``) observation groups.
        """
        return TensorDict(self._obs_groups(), batch_size=[self.num_envs], device=self.device)

    def render_topdown(self):
        """Render a per-env top-down view for validation.

        Delegates to the renderer's top-down pass.

        Returns:
            A ``(N, H, W, 3)`` uint8 tensor: one bird's-eye view per env.
        """
        return self.renderer.topdown(self)

    def render_spectator(self):
        """Render a single high-res top-down view of all envs.

        Delegates to the renderer's spectator pass.

        Returns:
            An ``(H, W, 3)`` uint8 image showing every env's car in one
            bird's-eye view.
        """
        return self.renderer.spectator(self)
