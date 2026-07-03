"""DeepRacer environment on Genesis, exposing the rsl-rl-lib 5.x VecEnv contract.

Observation groups (TensorDict):
  - "state":  (N, 28) proprioception + track-frame features (teacher / critic obs)
  - "camera": (N, 3, H, W) float in [0,1], front RGB camera (only when vision=True)

Action space (2,): [steering, throttle] in [-1, 1], mapped to the original
DeepRacer Box([-30deg, 0.1 m/s], [+30deg, 4.0 m/s]).

The runner never calls reset(); done envs are re-spawned inside step().
"""

import math

import numpy as np
import torch
from tensordict import TensorDict

import genesis as gs

from .track import MultiTrack
from ..randomization.domain_rand import randomize_physics, randomize_camera_mount

WHEEL_DOFS = ["left_rear_wheel_joint", "right_rear_wheel_joint",
              "left_front_wheel_joint", "right_front_wheel_joint"]
STEER_DOFS = ["left_steering_hinge_joint", "right_steering_hinge_joint"]


def _yaw_from_quat(q):
    # q: (N, 4) wxyz
    w, x, y, z = q.unbind(dim=1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap(a):
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


class DeepRacerEnv:
    def __init__(self, num_envs, env_cfg, show_viewer=False, device=None):
        self.device = torch.device(device) if device is not None else gs.device
        self.cfg = env_cfg
        self.num_envs = num_envs
        self.num_actions = 2
        self.vision = env_cfg["vision"]

        self.dt = env_cfg["dt"] * env_cfg["decimation"]  # control dt
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        names = env_cfg["track"] if isinstance(env_cfg["track"], (list, tuple)) else [env_cfg["track"]]
        self.track = MultiTrack(names, num_envs, self.device)

        # ------------- scene -------------
        renderer = gs.renderers.BatchRenderer(use_rasterizer=True) if self.vision else gs.renderers.Rasterizer()
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=env_cfg["dt"], substeps=1),
            rigid_options=gs.options.RigidOptions(
                dt=env_cfg["dt"],
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                # per-env dofs/links properties (kp/kv/armature/mass/COM DR)
                # need batched physics info, per the Genesis DR guide
                batch_dofs_info=bool(env_cfg.get("randomize", False)),
                batch_links_info=bool(env_cfg.get("randomize", False)),
            ),
            vis_options=gs.options.VisOptions(
                shadow=False,
                ambient_light=(0.35, 0.35, 0.35),
                background_color=tuple(env_cfg.get("background_color", (0.55, 0.72, 0.9))),
            ),
            renderer=renderer,
            show_viewer=show_viewer,
        )
        # green ground doubles as the field: some DAE ground materials render
        # transparent under Madrona, and this is what shows through. Must be a
        # surface color — Madrona does not sample ImageTexture on primitives.
        fc = env_cfg.get("field_color", (0.30, 0.48, 0.32))
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(pos=(0, 0, -0.001)),
            surface=gs.surfaces.Rough(color=(*fc, 1.0)),
        )
        # optional workaround knob for gs-madrona texture channel quirks (the
        # alpha-cutout centerline texture renders R<->G swapped; opaque
        # textures render correctly, so this stays off by default)
        self.rg_swap = bool(env_cfg.get("madrona_rg_swap", False)) and self.vision
        self.car = self.scene.add_entity(
            gs.morphs.URDF(
                file=f"{__file__.rsplit('/envs/', 1)[0]}/assets/urdf/deepracer/deepracer_processed.urdf",
                pos=(0, 0, 0.05),
                merge_fixed_links=True,
                links_to_keep=["camera_link"],
            ),
        )
        track_morphs = [gs.morphs.Mesh(file=p, fixed=True, collision=False)
                        for p in self.track.mesh_paths]
        # a list of morphs makes the entity heterogeneous: each parallel env
        # simulates (and renders) one geometry variant
        self.track_entity = self.scene.add_entity(
            track_morphs if len(track_morphs) > 1 else track_morphs[0])

        self.cam = None
        self.top_cam = None
        self.spec_cam = None
        if env_cfg.get("spectator", False):
            # high-res bird's-eye view rendered by the rasterizer (true colors,
            # any resolution, shows every env's car in one image). With the
            # BatchRenderer active it must be a debug camera to stay off the
            # batch pipeline.
            t0 = self.track.tracks[0]
            c = t0.center.mean(dim=0).cpu().numpy()
            extent = (t0.center.max(dim=0).values - t0.center.min(dim=0).values).max().item()
            sw, sh = env_cfg.get("spectator_res", (1280, 960))
            self.spec_cam = self.scene.add_camera(
                res=(sw, sh),
                pos=(float(c[0]), float(c[1]), extent * 1.1),
                lookat=(float(c[0]), float(c[1]), 0.0),
                up=(0.0, 1.0, 0.0),
                fov=60,
                GUI=False,
                debug=self.vision,
            )
        if self.vision:
            self.scene.add_light(
                pos=(0.0, 0.0, 10.0), dir=(0.4, 0.3, -1.0), directional=True,
                castshadow=False, intensity=float(env_cfg.get("light_intensity", 6.0)),
            )
            res = env_cfg["camera_res"]  # (W, H)
            self.cam = self.scene.add_camera(res=res, fov=env_cfg["camera_fov"], GUI=False)
            if env_cfg.get("topdown_camera", False):
                # per-env bird's-eye pose over each env's own track variant
                centers, heights = [], []
                for t in self.track.tracks:
                    c = t.center.mean(dim=0)
                    extent = (t.center.max(dim=0).values - t.center.min(dim=0).values).max()
                    centers.append(c)
                    heights.append(extent * 1.2)
                ev = self.track.variant_idx
                self.top_cam_center = torch.stack(centers)[ev]          # (N, 2)
                self.top_cam_height = torch.stack(heights)[ev]          # (N,)
                c0 = centers[0].cpu().numpy()
                self.top_cam = self.scene.add_camera(
                    res=res,
                    pos=(float(c0[0]), float(c0[1]), float(heights[0])),
                    lookat=(float(c0[0]), float(c0[1]), 0.0),
                    up=(0.0, 1.0, 0.0),
                    fov=60,
                    GUI=False,
                )

        self.scene.build(n_envs=num_envs)

        if self.top_cam is not None:
            pos = torch.cat([self.top_cam_center,
                             self.top_cam_height[:, None]], dim=1)
            lookat = torch.cat([self.top_cam_center,
                                torch.zeros(num_envs, 1, device=self.device)], dim=1)
            up = torch.tensor([[0.0, 1.0, 0.0]], device=self.device).expand(num_envs, 3)
            self.top_cam.set_pose(pos=pos, lookat=lookat, up=up)

        # ------------- dof bookkeeping -------------
        self.wheel_dofs = [self.car.get_joint(n).dof_idx_local for n in WHEEL_DOFS]
        self.steer_dofs = [self.car.get_joint(n).dof_idx_local for n in STEER_DOFS]
        self.car.set_dofs_kp(
            torch.full((2,), env_cfg["steer_kp"], device=self.device), self.steer_dofs)
        self.car.set_dofs_kv(
            torch.full((2,), env_cfg["steer_kv"], device=self.device), self.steer_dofs)
        self.car.set_dofs_kv(
            torch.full((4,), env_cfg["wheel_kv"], device=self.device), self.wheel_dofs)
        # cap drive torque near the traction limit; unbounded torque with a
        # P velocity controller causes wheel-slip limit cycles at high speed
        tq = env_cfg["wheel_max_torque"]
        self.car.set_dofs_force_range(
            torch.full((4,), -tq, device=self.device),
            torch.full((4,), tq, device=self.device),
            self.wheel_dofs)

        import trimesh
        wheel_mesh = trimesh.load(
            f"{__file__.rsplit('/envs/', 1)[0]}/assets/meshes/deepracer/left_rear_wheel.STL")
        self.wheel_radius = float(wheel_mesh.extents[2]) / 2.0

        # ------------- camera mount -------------
        if self.cam is not None:
            self.cam_offset_T = self._camera_offset_T(env_cfg.get("camera_pitch_deg", 0.0))
            self.cam.attach(self.car.get_link("camera_link"), self.cam_offset_T)

        # ------------- buffers -------------
        N = num_envs
        self.episode_length_buf = torch.zeros(N, device=self.device, dtype=torch.long)
        self.reset_buf = torch.ones(N, device=self.device, dtype=torch.bool)
        self.rew_buf = torch.zeros(N, device=self.device)
        self.time_out_buf = torch.zeros(N, device=self.device, dtype=torch.bool)
        self.actions = torch.zeros(N, 2, device=self.device)
        self.last_actions = torch.zeros(N, 2, device=self.device)
        self.progress_m = torch.zeros(N, device=self.device)
        self.laps = torch.zeros(N, device=self.device)
        self.extras = {"log": {}}

        self.lookahead_k = env_cfg["lookahead_k"]
        self.num_state_obs = 8 + 2 * self.lookahead_k
        self.state_buf = torch.zeros(N, self.num_state_obs, device=self.device)
        if self.vision:
            w, h = env_cfg["camera_res"]
            self.image_buf = torch.zeros(N, 3, h, w, device=self.device)
            # when rendering above the policy's training resolution (demo
            # videos), the policy still receives a downscaled frame
            self.policy_res = env_cfg.get("policy_res") or env_cfg["camera_res"]
            self.obs_image_buf = self.image_buf

        self.reward_scales = dict(env_cfg["reward_scales"])
        self.episode_sums = {k: torch.zeros(N, device=self.device) for k in self.reward_scales}

        self.reset_idx(torch.arange(N, device=self.device))
        self._post_physics()

    # ------------------------------------------------------------------
    def _camera_offset_T(self, pitch_deg):
        """camera_link frame -> Genesis camera frame (camera looks along -z)."""
        base = np.array([
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        p = math.radians(pitch_deg)  # positive pitches the view down
        rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(p), -math.sin(p)],
            [0.0, math.sin(p), math.cos(p)],
        ])
        T = np.eye(4)
        T[:3, :3] = base @ rx
        return T

    # ------------------------------------------------------------------
    def step(self, actions):
        self.actions = torch.clip(actions, -1.0, 1.0)
        steer = self.actions[:, 0:1] * math.radians(self.cfg["max_steering_deg"])
        speed = self.cfg["min_speed"] + (self.actions[:, 1:2] + 1) * 0.5 * (
            self.cfg["max_speed"] - self.cfg["min_speed"])
        wheel_omega = (speed / self.wheel_radius).repeat(1, 4)

        self.car.control_dofs_position(steer.repeat(1, 2), self.steer_dofs)
        self.car.control_dofs_velocity(wheel_omega, self.wheel_dofs)
        for _ in range(self.cfg["decimation"]):
            self.scene.step()

        self.episode_length_buf += 1
        self._post_physics()
        self._compute_reward()
        self._check_termination()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
            self._post_physics(env_ids)

        self.last_actions[:] = self.actions
        self.extras["time_outs"] = self.time_out_buf
        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    # ------------------------------------------------------------------
    def _post_physics(self, env_ids=None):
        """Refresh cached kinematics + track-frame quantities + observations."""
        pos = self.car.get_pos()
        quat = self.car.get_quat()
        vel = self.car.get_vel()
        ang = self.car.get_ang()

        self.base_pos = pos
        self.yaw = _yaw_from_quat(quat)
        cy, sy = torch.cos(self.yaw), torch.sin(self.yaw)
        self.v_forward = vel[:, 0] * cy + vel[:, 1] * sy
        self.v_lateral = -vel[:, 0] * sy + vel[:, 1] * cy
        self.yaw_rate = ang[:, 2]
        # z component of body-frame up vector (flip detection)
        w, x, y, z = quat.unbind(dim=1)
        self.up_z = 1 - 2 * (x * x + y * y)

        loc = self.track.localize(pos[:, :2])
        self.wp_idx = loc["wp_idx"]
        self.lateral = loc["lateral"]
        self.half_width = loc["half_width"]
        self.heading_err = _wrap(self.yaw - loc["track_yaw"])
        new_progress = loc["progress_m"]
        d = new_progress - self.progress_m
        L = self.track.total_len_env
        self.d_progress = torch.where(d > 0.5 * L, d - L, torch.where(d < -0.5 * L, d + L, d))
        if env_ids is not None and len(env_ids) > 0:
            self.d_progress[env_ids] = 0.0
        self.progress_m = new_progress

        # ---- state obs ----
        la_idx = self.track.lookahead(self.wp_idx, self.lookahead_k, self.cfg["lookahead_stride"])
        la_pts = self.track.lookahead_points(la_idx)             # (N, K, 2)
        rel = la_pts - pos[:, None, :2]
        rel_x = rel[..., 0] * cy[:, None] + rel[..., 1] * sy[:, None]
        rel_y = -rel[..., 0] * sy[:, None] + rel[..., 1] * cy[:, None]
        la_scale = self.cfg["lookahead_scale"]
        self.state_buf = torch.cat(
            [
                (self.v_forward / self.cfg["max_speed"]).unsqueeze(1),
                self.v_lateral.unsqueeze(1),
                (self.yaw_rate / 5.0).unsqueeze(1),
                (self.lateral / self.half_width.clamp(min=0.1)).unsqueeze(1),
                torch.sin(self.heading_err).unsqueeze(1),
                torch.cos(self.heading_err).unsqueeze(1),
                self.actions,
                rel_x / la_scale,
                rel_y / la_scale,
            ],
            dim=1,
        )
        if self.cfg.get("obs_noise", 0.0) > 0:
            self.state_buf += torch.randn_like(self.state_buf) * self.cfg["obs_noise"]

        # ---- camera obs ----
        if self.vision:
            self.cam.move_to_attach()
            rgb = self.cam.render(rgb=True)[0]                   # (N, H, W, 3) uint8 cuda
            if self.rg_swap:
                rgb = rgb[..., [1, 0, 2]]
            img = rgb.permute(0, 3, 1, 2).float() / 255.0
            if self.cfg.get("pixel_noise", 0.0) > 0:
                img = (img + torch.randn_like(img) * self.cfg["pixel_noise"]).clamp(0, 1)
            self.image_buf = img
            if tuple(self.policy_res) != tuple(self.cfg["camera_res"]):
                pw, ph = self.policy_res
                self.obs_image_buf = torch.nn.functional.interpolate(
                    img, size=(ph, pw), mode="area")
            else:
                self.obs_image_buf = img

    # ------------------------------------------------------------------
    def _compute_reward(self):
        on_track = self.lateral.abs() < (self.half_width - self.cfg["wheel_margin"])
        terms = {
            "progress": self.d_progress,
            "speed": self.v_forward.clamp(0.0, self.cfg["max_speed"]) * self.dt,
            "centered": torch.exp(-((self.lateral / self.half_width.clamp(min=0.1)) ** 2)) * self.dt,
            "heading": -self.heading_err.abs() * self.dt,
            "steering": -self.actions[:, 0].abs() * self.dt,
            "action_rate": -((self.actions - self.last_actions) ** 2).sum(dim=1) * self.dt,
            "off_track": (~on_track).float() * self.dt,
        }
        self.rew_buf.zero_()
        for name, scale in self.reward_scales.items():
            r = terms[name] * scale
            self.rew_buf += r
            self.episode_sums[name] += r

    # ------------------------------------------------------------------
    def _check_termination(self):
        off = self.lateral.abs() > (self.half_width + self.cfg["off_track_margin"])
        flipped = self.up_z < 0.3
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length
        self.reset_buf = off | flipped | self.time_out_buf
        # terminal penalty for genuine failures (not timeouts)
        self.rew_buf += (off | flipped).float() * self.cfg["crash_penalty"]

    # ------------------------------------------------------------------
    def reset_idx(self, env_ids):
        n = len(env_ids)
        if n == 0:
            return
        pos_xy, yaw = self.track.spawn_pose(
            env_ids, self.cfg["random_start"],
            lateral_noise=self.cfg["spawn_lateral_noise"], yaw_noise=self.cfg["spawn_yaw_noise"])

        qpos = torch.zeros(n, 13, device=self.device)
        qpos[:, 0:2] = pos_xy
        qpos[:, 2] = self.cfg["spawn_height"]
        qpos[:, 3] = torch.cos(yaw / 2)
        qpos[:, 6] = torch.sin(yaw / 2)
        self.car.set_qpos(qpos, envs_idx=env_ids)
        self.car.zero_all_dofs_velocity(envs_idx=env_ids)
        self.car.control_dofs_position(torch.zeros(n, 2, device=self.device), self.steer_dofs, envs_idx=env_ids)
        self.car.control_dofs_velocity(torch.zeros(n, 4, device=self.device), self.wheel_dofs, envs_idx=env_ids)

        if self.cfg.get("randomize", False):
            randomize_physics(self, env_ids)
            if self.vision:
                randomize_camera_mount(self, env_ids)

        # episode logging
        self.extras["log"] = {}
        for key, sums in self.episode_sums.items():
            self.extras["log"][f"Episode/rew_{key}"] = sums[env_ids].mean()
            sums[env_ids] = 0.0
        self.extras["log"]["Episode/length"] = self.episode_length_buf[env_ids].float().mean()

        self.episode_length_buf[env_ids] = 0
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        # progress buffer refreshed on next _post_physics via env_ids d_progress zeroing
        self.progress_m[env_ids] = self.track.localize(pos_xy, envs_idx=env_ids)["progress_m"]

    # ------------------------------------------------------------------
    def get_observations(self):
        groups = {"state": self.state_buf}
        if self.vision:
            groups["camera"] = self.obs_image_buf
        return TensorDict(groups, batch_size=[self.num_envs], device=self.device)

    def render_topdown(self):
        """(N, H, W, 3) uint8 per-env bird's-eye view (validation only)."""
        assert self.top_cam is not None
        rgb = self.top_cam.render(rgb=True)[0]
        return rgb[..., [1, 0, 2]] if self.rg_swap else rgb

    def render_spectator(self):
        """(H, W, 3) uint8 high-res bird's-eye view showing all envs' cars."""
        assert self.spec_cam is not None
        rgb = np.asarray(self.spec_cam.render(rgb=True)[0])
        return rgb.reshape(rgb.shape[-3:])
