"""Pluggable rendering strategies for the DeepRacer env.

The env holds ONE ``Renderer``; the strategy owns every vision decision.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

import genesis as gs

# Visual DR *definitions* live in randomization/visual.py; the renderer is the
# application site that imports and calls them (Part L).
from ..randomization.visual import (  # noqa: E402
    add_pixel_noise,
    sample_env_map,
    sample_mount_transforms,
    sample_world_color,
)

if TYPE_CHECKING:
    from .base_env import DeepRacerEnv


def camera_offset_T(pitch_deg: float) -> np.ndarray:
    """Build the mount transform from the ``camera_link`` frame to the camera.

    Maps into the Genesis camera frame (-z), pitching down by ``pitch_deg``.

    Args:
        pitch_deg: Downward pitch of the view in degrees; positive values tilt
            the camera toward the ground.

    Returns:
        A (4, 4) homogeneous transform from the ``camera_link`` frame to the
        Genesis camera frame, including the pitch rotation.
    """
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


def _track_extent(track):
    """Compute the centroid and bounding extent of a track's centerline.

    Used to place bird's-eye cameras so a whole track variant fits in frame.

    Args:
        track: A Track whose ``center`` centerline points are inspected.

    Returns:
        A pair ``(center_xy, max_extent)`` where ``center_xy`` is the mean
        centerline position and ``max_extent`` is the larger of the two
        centerline bounding-box side lengths.
    """
    c = track.center.mean(dim=0)
    extent = (track.center.max(dim=0).values - track.center.min(dim=0).values).max()
    return c, extent


def make_renderer(vision_cfg: dict) -> "Renderer":
    """Select and instantiate the rendering strategy from the env config.

    ``vision`` off returns :class:`NullRenderer`; else Nyx or default Madrona.

    Args:
        vision_cfg: Env config; ``vision`` gates whether any camera renderer is
            built, and ``vision_renderer`` (``"batch"`` default, ``"nyx"``, or
            ``"rasterizer"`` for the CPU per-env path) picks the vision backend.

    Returns:
        The renderer strategy matching the config: :class:`NullRenderer`,
        :class:`NyxRenderer`, :class:`RasterizerObsRenderer`, or
        :class:`MadronaRenderer`.
    """
    if not vision_cfg["vision"]:
        return NullRenderer()
    renderer = vision_cfg.get("vision_renderer", "batch")
    if renderer == "nyx":
        return NyxRenderer()
    if renderer == "rasterizer":
        return RasterizerObsRenderer()
    return MadronaRenderer()


class Renderer:
    """Base rendering strategy: no camera observation, optional debug view.

    Defines the strategy interface the env drives across its lifecycle.

    Attributes:
        has_camera: Whether this strategy produces camera observations.
        merge_fixed_links: Whether the scene may merge fixed links.
        spec_cam: The shared bird's-eye spectator debug camera, or None.
    """

    has_camera: bool = False
    merge_fixed_links: bool = True
    _scene_batch_renderer: bool = False
    _spectator_debug: bool = False

    def scene_renderer(self):
        """Return the Genesis scene renderer this strategy requires.

        Returns:
            A Madrona ``BatchRenderer`` (rasterizer-backed) when this strategy
            needs batched camera observations, otherwise a plain
            ``Rasterizer``.
        """
        return (gs.renderers.BatchRenderer(use_rasterizer=True)
                if self._scene_batch_renderer else gs.renderers.Rasterizer())

    # ---------------------------------------------------------- build lifecycle
    def build(self, env: "DeepRacerEnv", vision_cfg: dict) -> None:
        """Add cameras / lights / sensors to the scene before it is built.

        Sets up the shared spectator camera, then delegates to :meth:`_build`.

        Args:
            env: The env being built; its ``scene`` and ``track`` are used to
                place the spectator camera.
            vision_cfg: Env config; ``spectator`` enables the debug camera and
                ``spectator_res`` sets its resolution.
        """
        self.spec_cam = None
        if vision_cfg.get("spectator", False):
            # high-res bird's-eye view (rasterizer, true colors, all cars in one
            # image). With a BatchRenderer active it must be a debug camera to
            # stay off the batch pipeline (Madrona sets _spectator_debug=True).
            c, extent = _track_extent(env.track.tracks[0])
            c = c.cpu().numpy()
            sw, sh = vision_cfg.get("spectator_res", (1280, 960))
            self.spec_cam = env.scene.add_camera(
                res=(sw, sh),
                pos=(float(c[0]), float(c[1]), float(extent) * 1.1),
                lookat=(float(c[0]), float(c[1]), 0.0),
                up=(0.0, 1.0, 0.0), fov=60, GUI=False, debug=self._spectator_debug)
        self._build(env, vision_cfg)

    def _build(self, env: "DeepRacerEnv", vision_cfg: dict) -> None:
        """Add the strategy's observation camera / sensors and top-down camera.

        Subclass hook run by :meth:`build`; the base adds nothing (no vision).

        Args:
            env: The env being built; its ``scene`` receives the cameras or
                sensors.
            vision_cfg: Env config supplying resolution, FOV, and top-down
                options.
        """

    def finalize(self, env: "DeepRacerEnv", vision_cfg: dict) -> None:
        """Finish setup after the scene is built.

        Post-build hook for attaching cameras, posing, and appearance state.

        Args:
            env: The built env, providing the car links and device.
            vision_cfg: Env config supplying pose, appearance, and observation
                parameters.
        """

    # ------------------------------------------------- per-step / per-episode
    def render(self, env: "DeepRacerEnv"):
        """Produce the per-step camera images for this strategy.

        Args:
            env: The env whose current frame is rendered.

        Returns:
            A pair ``(full_image, obs_image)``, each an ``(N, 3, H, W)`` tensor
            (the full-resolution render and the policy-resolution observation),
            or ``(None, None)`` for a no-vision strategy.
        """
        return None, None

    def resample_appearance(self, env_ids: torch.Tensor) -> None:
        """Redraw the per-episode world-color remap for the given envs.

        No-op on the base strategy; only vision renderers carry a color remap.

        Args:
            env_ids: Indices of the envs whose color palettes are resampled
                (typically the envs resetting this step).
        """

    def randomize_mount(self, env: "DeepRacerEnv", env_ids: torch.Tensor) -> None:
        """Jitter the camera mount per episode for the given envs.

        No-op except on the Madrona strategy, which owns an attached camera.

        Args:
            env: The env owning the camera being jittered.
            env_ids: Indices of the envs whose camera mounts are re-randomized.
        """

    # ------------------------------------------------------------- debug views
    def topdown(self, env: "DeepRacerEnv") -> torch.Tensor:
        """Render the per-env top-down (bird's-eye) view.

        Args:
            env: The env to render from above.

        Returns:
            An ``(N, H, W, 3)`` batch of top-down RGB frames.

        Raises:
            NotImplementedError: Always, on the base strategy; the top-down
                view requires a vision renderer.
        """
        raise NotImplementedError("top-down view requires a vision renderer")

    def spectator(self, env: "DeepRacerEnv") -> np.ndarray:
        """Render the single high-res spectator (bird's-eye) debug frame.

        Args:
            env: The env to render (unused directly; the spectator camera is
                already positioned over the track).

        Returns:
            An ``(H, W, 3)`` RGB image of the whole track. Under a renderer that
            sets ``env_separate_rigid`` every camera is batched and each env is
            drawn in isolation, so the frame shows env 0's car alone rather than
            the whole fleet.

        Raises:
            AssertionError: If the spectator camera was not enabled via
                ``cfg['spectator']`` during :meth:`build`.
        """
        assert self.spec_cam is not None, "spectator camera not enabled (cfg['spectator'])"
        rgb = np.asarray(self.spec_cam.render(rgb=True)[0])
        if rgb.ndim == 4:                      # batched camera: keep env 0
            rgb = rgb[0]
        return rgb.reshape(rgb.shape[-3:])


class NullRenderer(Renderer):
    """Feature / no-vision strategy: state observations only, no camera.

    Used when ``vision`` is off; inherits the base strategy unchanged.
    """


class _CameraRenderer(Renderer):
    """Shared base for the vision strategies.

    Shares the world-color remap, pixel noise, and downscaling to policy res.

    Attributes:
        has_camera: Whether this strategy produces camera observations.
        rg_swap: Whether to swap the red/green channels of the raw frame.
        world_color_s: Strength of the per-episode world-color remap.
        policy_res: Resolution of the observation handed to the policy.
        color_mat: Per-env color-remap linear transform.
        color_bias: Per-env color-remap additive bias.
    """

    has_camera = True

    def finalize(self, env: "DeepRacerEnv", vision_cfg: dict) -> None:
        """Cache render parameters and initialize per-env color-remap state.

        Reads render settings and allocates the per-env color transform.

        Args:
            env: The built env, providing ``num_envs`` and ``device``.
            vision_cfg: Env config; reads ``madrona_rg_swap``, ``appearance``
                (``world_color`` strength), ``policy_res``, ``camera_res``, and
                ``pixel_noise``.
        """
        self.rg_swap = bool(vision_cfg.get("madrona_rg_swap", False))
        self._device = env.device
        appearance = vision_cfg.get("appearance") or {}
        self.world_color_s = float(appearance.get("world_color", 0.0))
        self.policy_res = vision_cfg.get("policy_res") or vision_cfg["camera_res"]
        self._camera_res = vision_cfg["camera_res"]
        self._pixel_noise = float(vision_cfg.get("pixel_noise", 0.0))
        if self.world_color_s > 0:
            # per-env, EPISODE-static color remap (resampled each reset): each
            # agent sees the same world through its own random palette
            n = env.num_envs
            self.color_mat = torch.eye(3, device=env.device).repeat(n, 1, 1)
            self.color_bias = torch.zeros(n, 1, 3, device=env.device)

    def _acquire_rgb(self, env: "DeepRacerEnv") -> torch.Tensor:
        """Grab the raw camera frame from the backend for this step.

        Subclass hook: Madrona renders its camera, Nyx reads its sensor.

        Args:
            env: The env whose current camera frame is captured.

        Returns:
            An ``(N, H, W, 3)`` uint8 tensor of RGB pixels on device.

        Raises:
            NotImplementedError: Always, on this shared base; subclasses must
                override.
        """
        raise NotImplementedError

    def render(self, env: "DeepRacerEnv"):
        """Acquire and post-process the per-step camera images.

        Applies the color remap, optional pixel noise, and policy downscaling.

        Args:
            env: The env whose current frame is rendered.

        Returns:
            A pair ``(full_image, obs_image)``: the full-resolution
            ``(N, 3, H, W)`` render and the ``(N, 3, ph, pw)`` policy
            observation (identical when render and policy resolutions match).
        """
        rgb = self._acquire_rgb(env)                          # (N, H, W, 3) uint8
        imgf = rgb.float().div_(255.0)
        if self.world_color_s > 0:
            # color remap in native NHWC: (N, H*W, 3) is a free view here, and
            # the tall-skinny batched GEMM is ~10x cheaper than any NCHW form
            n, h, w, c = imgf.shape
            imgf = ((imgf.view(n, h * w, c) @ self.color_mat.transpose(1, 2)
                     + self.color_bias).clamp_(0.0, 1.0).view(n, h, w, c))
        img = imgf.permute(0, 3, 1, 2)
        img = add_pixel_noise(img, self._pixel_noise)
        if tuple(self.policy_res) != tuple(self._camera_res):
            # rendering above the policy's resolution (demo videos); the policy
            # still receives a downscaled frame
            pw, ph = self.policy_res
            obs = torch.nn.functional.interpolate(img, size=(ph, pw), mode="area")
        else:
            obs = img
        return img, obs

    def resample_appearance(self, env_ids: torch.Tensor) -> None:
        """Draw a fresh per-env world-color remap for the given envs.

        Composes hue rotation, sat/val scaling, mixing, and bias per env.

        Args:
            env_ids: Indices of the envs whose color palettes are redrawn
                (typically the envs resetting this step).
        """
        if self.world_color_s <= 0:
            return
        mat, bias = sample_world_color(len(env_ids), self.world_color_s,
                                       self._device)
        self.color_mat[env_ids] = mat
        self.color_bias[env_ids] = bias


class MadronaRenderer(_CameraRenderer):
    """Madrona batch-renderer camera obs with camera-mount randomization.

    Uses the ``BatchRenderer`` and jitters the attached camera's mount.

    Attributes:
        merge_fixed_links: Whether the scene may merge fixed links.
        cam: The car-attached observation camera.
        top_cam: The per-env top-down camera, or None.
        cam_offset_T: The base mount transform from camera_link to the camera.
    """

    merge_fixed_links = True
    _scene_batch_renderer = True
    _spectator_debug = True

    def _build(self, env: "DeepRacerEnv", vision_cfg: dict) -> None:
        """Add the directional light, observation camera, and top-down camera.

        Adds the light, the FOV-set obs camera, and an optional top-down camera.

        Args:
            env: The env being built; its ``scene`` and ``track`` receive the
                light and cameras.
            vision_cfg: Env config; reads ``light_intensity``, ``camera_res``
                (W, H), ``camera_fov``, and ``topdown_camera``.
        """
        env.scene.add_light(pos=(0.0, 0.0, 10.0), dir=(0.4, 0.3, -1.0),
                            directional=True, castshadow=False,
                            intensity=float(vision_cfg.get("light_intensity", 6.0)))
        res = vision_cfg["camera_res"]  # (W, H)
        self.cam = env.scene.add_camera(res=res, fov=vision_cfg["camera_fov"], GUI=False)
        self.top_cam = None
        if vision_cfg.get("topdown_camera", False):
            # per-env bird's-eye pose over each env's own track variant
            centers, heights = [], []
            for t in env.track.tracks:
                c, extent = _track_extent(t)
                centers.append(c)
                heights.append(extent * 1.2)
            ev = env.track.variant_idx
            self._top_center = torch.stack(centers)[ev]          # (N, 2)
            self._top_height = torch.stack(heights)[ev]          # (N,)
            c0 = centers[0].cpu().numpy()
            self.top_cam = env.scene.add_camera(
                res=res, pos=(float(c0[0]), float(c0[1]), float(heights[0])),
                lookat=(float(c0[0]), float(c0[1]), 0.0),
                up=(0.0, 1.0, 0.0), fov=60, GUI=False)

    def finalize(self, env: "DeepRacerEnv", vision_cfg: dict) -> None:
        """Attach the observation camera to the car and pose the top-down cam.

        Runs shared setup, mounts the obs camera, and poses the top-down cam.

        Args:
            env: The built env, providing the car link, ``num_envs``, and
                ``device``.
            vision_cfg: Env config; reads ``camera_pitch_deg`` (plus everything
                the base :meth:`_CameraRenderer.finalize` consumes).
        """
        super().finalize(env, vision_cfg)
        self.cam_offset_T = camera_offset_T(vision_cfg.get("camera_pitch_deg", 0.0))
        self.cam.attach(env.car.get_link("camera_link"), self.cam_offset_T)
        if self.top_cam is not None:
            pos = torch.cat([self._top_center, self._top_height[:, None]], dim=1)
            lookat = torch.cat([self._top_center,
                                torch.zeros(env.num_envs, 1, device=env.device)], dim=1)
            up = torch.tensor([[0.0, 1.0, 0.0]], device=env.device).expand(env.num_envs, 3)
            self.top_cam.set_pose(pos=pos, lookat=lookat, up=up)

    def _acquire_rgb(self, env: "DeepRacerEnv") -> torch.Tensor:
        """Move the attached camera into place and render the batched frame.

        Applies the optional ``madrona_rg_swap`` channel-order correction.

        Args:
            env: The env whose current frame is captured.

        Returns:
            An ``(N, H, W, 3)`` uint8 CUDA tensor of RGB pixels.
        """
        self.cam.move_to_attach()
        rgb = self.cam.render(rgb=True)[0]                       # (N, H, W, 3) uint8 cuda
        if self.rg_swap:
            rgb = rgb[..., [1, 0, 2]]
        return rgb

    def randomize_mount(self, env: "DeepRacerEnv", env_ids: torch.Tensor) -> None:
        """Re-randomize the camera mount pitch and position for the given envs.

        Perturbs the camera-offset transform by uniform pitch/position jitter.

        Args:
            env: The env owning the attached camera; ``cfg['rand']`` supplies
                ``camera_pitch_jitter_deg`` and ``camera_pos_jitter_m``.
            env_ids: Indices of the envs whose camera mounts are re-randomized.
        """
        cfg = env.cfg["rand"]
        jitter_deg = cfg.get("camera_pitch_jitter_deg", 0.0)
        jitter_pos = cfg.get("camera_pos_jitter_m", 0.0)
        if jitter_deg <= 0 and jitter_pos <= 0:
            return
        cam = self.cam
        base = torch.as_tensor(self.cam_offset_T, dtype=torch.float32, device=env.device)
        if cam._attached_offset_T.dim() == 2:
            cam._attached_offset_T = base.expand(env.num_envs, 4, 4).clone()
        T = sample_mount_transforms(self.cam_offset_T, jitter_deg, jitter_pos,
                                    len(env_ids), env.device)
        cam._attached_offset_T[env_ids] = T

    def topdown(self, env: "DeepRacerEnv") -> torch.Tensor:
        """Render the per-env top-down view from the batch top-down camera.

        Applies the optional channel swap to match the obs camera's order.

        Args:
            env: The env to render from above.

        Returns:
            An ``(N, H, W, 3)`` batch of top-down RGB frames.

        Raises:
            AssertionError: If the top-down camera was not enabled via
                ``topdown_camera``.
        """
        assert self.top_cam is not None
        rgb = self.top_cam.render(rgb=True)[0]
        return rgb[..., [1, 0, 2]] if self.rg_swap else rgb


class RasterizerObsRenderer(_CameraRenderer):
    """Per-env CPU rasterizer camera obs — the backend=='cpu' vision path (M.2).

    Madrona and Nyx are GPU-only, so on the CPU backend the policy camera is
    rendered with the same ``gs.renderers.Rasterizer()`` that already backs the
    spectator/top-down debug views. It holds ONE camera per env (``env_idx=i``)
    and renders them in a Python loop, so it is unbatched and far slower than
    Madrona — a debug / small-``num_envs`` / no-GPU path, not a throughput path.
    Reuses ``_CameraRenderer``'s device-agnostic post-processing (world-color
    remap, pixel noise, policy downscale) verbatim; only the frame source swaps.

    Note:
        ``randomize_mount`` (``camera_jitter`` DR) is a no-op on this path — the
        per-env mount jitter is Madrona-only for now; image-space DR (distortion,
        crop, photometric) still applies, since it is renderer-agnostic.

    Attributes:
        merge_fixed_links: Whether the scene may merge fixed links.
        cams: One car-attached observation camera per env.
        top_cams: One per-env top-down camera each, or None.
        cam_offset_T: The base mount transform from camera_link to the camera.
    """

    merge_fixed_links = True
    _scene_batch_renderer = False    # plain Rasterizer, not a BatchRenderer
    _spectator_debug = False         # no batch pipeline, so no debug camera needed
    # render each env's rigid bodies in isolation: one batched camera instead of
    # one per env (a single render call), and no foreign car in frame. This is
    # the CPU-path answer to Madrona's Part O spatial tiling.
    env_separate_rigid = True

    def _build(self, env: "DeepRacerEnv", vision_cfg: dict) -> None:
        """Add one rasterizer camera per env.

        The plain ``Rasterizer`` scene renderer does not support ``add_light``
        (that is BatchRenderer-only); it lights the scene from the ambient light
        + background in ``gs.Scene``'s VisOptions, exactly like the spectator /
        top-down rasterizer views already do — so no explicit light is added.

        Args:
            env: The env being built; its ``scene`` receives the per-env cameras.
            vision_cfg: Env config; reads ``camera_res`` (W, H), ``camera_fov``,
                and ``topdown_camera``.
        """
        res = vision_cfg["camera_res"]  # (W, H)
        fov = vision_cfg["camera_fov"]
        # no env_idx: with env_separate_rigid the camera is batched, so one
        # render() call yields every env's frame and each sees only its own car.
        self.cam = env.scene.add_camera(res=res, fov=fov, GUI=False)
        self.top_cam = None
        if vision_cfg.get("topdown_camera", False):
            # camera obs on this path is single-track, so one pose fits every env
            c, extent = _track_extent(env.track.tracks[0])
            c = c.cpu().numpy()
            self.top_cam = env.scene.add_camera(
                res=res, pos=(float(c[0]), float(c[1]), float(extent) * 1.2),
                lookat=(float(c[0]), float(c[1]), 0.0),
                up=(0.0, 1.0, 0.0), fov=60, GUI=False)

    def finalize(self, env: "DeepRacerEnv", vision_cfg: dict) -> None:
        """Attach the batched camera to the cars' ``camera_link``.

        Args:
            env: The built env, providing the car link, ``num_envs``, and
                ``device``.
            vision_cfg: Env config; reads ``camera_pitch_deg`` (plus everything
                the base :meth:`_CameraRenderer.finalize` consumes).
        """
        super().finalize(env, vision_cfg)
        self.cam_offset_T = camera_offset_T(vision_cfg.get("camera_pitch_deg", 0.0))
        self.cam.attach(env.car.get_link("camera_link"), self.cam_offset_T)

    def _acquire_rgb(self, env: "DeepRacerEnv") -> torch.Tensor:
        """Render each per-env camera in turn and stack the frames.

        Args:
            env: The env whose current frame is captured.

        Returns:
            An ``(N, H, W, 3)`` uint8 tensor of RGB pixels on ``env.device``.
        """
        self.cam.move_to_attach()
        return self._as_batch(self.cam.render(rgb=True)[0], env)

    def topdown(self, env: "DeepRacerEnv") -> torch.Tensor:
        """Render the per-env top-down view from each env's rasterizer camera.

        Args:
            env: The env to render from above.

        Returns:
            An ``(N, H, W, 3)`` batch of top-down RGB frames.

        Raises:
            AssertionError: If the top-down cameras were not enabled via
                ``topdown_camera``.
        """
        assert self.top_cam is not None
        return self._as_batch(self.top_cam.render(rgb=True)[0], env)

    @staticmethod
    def _as_batch(rgb, env: "DeepRacerEnv") -> torch.Tensor:
        """Normalize a batched rasterizer frame to ``(N, H, W, 3)`` on the device.

        The rasterizer hands back a vertically-flipped view (negative stride),
        which torch cannot wrap, so the array is made contiguous first.
        """
        rgb = np.ascontiguousarray(np.asarray(rgb))
        if rgb.ndim == 3:                      # single env: add the batch axis
            rgb = rgb[None]
        return torch.as_tensor(rgb, device=env.device)


class NyxRenderer(_CameraRenderer):
    """Nyx path-tracer sensor obs with true texture colors.

    Uses Nyx forward-path-tracer sensors instead of the batch renderer.

    Attributes:
        merge_fixed_links: Whether the scene may merge fixed links.
        nyx_cam: The car-attached Nyx observation sensor.
        nyx_top: The bird's-eye top-down Nyx sensor, or None.
    """

    merge_fixed_links = False   # the Nyx exporter refuses merged fixed links
    _scene_batch_renderer = False
    _spectator_debug = False

    def _build(self, env: "DeepRacerEnv", vision_cfg: dict) -> None:
        """Add the Nyx observation sensor and optional top-down sensor.

        Adds the obs sensor and an optional bird's-eye top-down sensor.

        Args:
            env: The env being built; its ``scene``, ``car``, and ``track``
                supply the mount link and framing.
            vision_cfg: Env config; reads ``nyx_light_intensity``, ``nyx_mode``,
                ``camera_res``, ``nyx_spp``, ``camera_fov``,
                ``camera_pitch_deg``, and ``topdown_camera``.
        """
        import gs_nyx.nyx_py_renderer as npr
        import gs_nyx.nyx_py_sdk as nps
        from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions

        sun = {"type": "directional", "dir": (0.4, 0.3, -1.0), "color": (1.0, 1.0, 1.0),
               "intensity": float(vision_cfg.get("nyx_light_intensity", 3.0)), "shadow": False}
        mode = getattr(npr.ERenderMode, vision_cfg.get("nyx_mode", "Forward"))
        res = vision_cfg["camera_res"]
        # denoise/AA off: their temporal history smears moving objects across
        # frames — bad for RL observations and for validation diffs
        common = dict(spp=int(vision_cfg.get("nyx_spp", 4)), render_mode=mode, lights=[sun],
                      denoise=False, anti_aliasing=nps.EAntiAliasing.Off)
        # Part P.1: per-env HDRI sky DR. Nyx registers the sensor's env_maps
        # into the scene at build (indices 0..N-1) and selects env-map i for env
        # i via set_env_map(env_index) in its render loop — so one texture-less
        # EnvironmentMapAsset per env is a cheap per-env-fixed sky (baked at
        # build; not per-episode). Attach to the OBS sensor only so the indices
        # line up 1:1 with the Genesis env index.
        env_maps = self._build_env_maps(env, vision_cfg, nps)
        # same link->camera mount transform as the Madrona path (looks along -z
        # of offset_T incl. the downward pitch); sensors ignore pos/euler offset
        self.nyx_cam = env.scene.add_sensor(NyxCameraOptions(
            res=res, fov=vision_cfg["camera_fov"],
            entity_idx=env.car.idx,
            link_idx_local=env.car.get_link("camera_link").idx_local,
            offset_T=camera_offset_T(vision_cfg.get("camera_pitch_deg", 0.0)),
            env_maps=env_maps, **common))
        self.nyx_top = None
        if vision_cfg.get("topdown_camera", False):
            c, extent = _track_extent(env.track.tracks[0])
            c = c.cpu().numpy()
            self.nyx_top = env.scene.add_sensor(NyxCameraOptions(
                res=res, fov=60,
                pos=(float(c[0]), float(c[1]), float(extent) * 1.2),
                lookat=(float(c[0]), float(c[1]), 0.0), up=(0.0, 1.0, 0.0),
                **common))

    def _acquire_rgb(self, env: "DeepRacerEnv") -> torch.Tensor:
        """Read the current RGB frame from the Nyx observation sensor.

        Args:
            env: The env whose current frame is captured (unused directly; the
                sensor is already attached).

        Returns:
            An ``(N, H, W, 3)`` uint8 CUDA tensor of RGB pixels (alpha dropped).
        """
        return self.nyx_cam.read().rgb[..., :3]                  # (N, H, W, 3) uint8 cuda

    def topdown(self, env: "DeepRacerEnv") -> torch.Tensor:
        """Read the per-env top-down view from the Nyx top-down sensor.

        Args:
            env: The env to render from above (unused directly; the sensor is
                already positioned).

        Returns:
            An ``(N, H, W, 3)`` batch of top-down RGB frames (alpha dropped).

        Raises:
            AssertionError: If the top-down sensor was not enabled via
                ``topdown_camera``.
        """
        assert self.nyx_top is not None
        return self.nyx_top.read().rgb[..., :3]

    @staticmethod
    def _build_env_maps(env: "DeepRacerEnv", vision_cfg: dict, nps):
        """Sample one per-env environment map (tint + exposure) for Nyx (P.1).

        Args:
            env: The env being built (supplies ``num_envs`` and ``device``).
            vision_cfg: Env config; ``env_map`` gives ``{"tint": (lo, hi),
                "multiplier": (lo, hi)}`` ranges (either key optional).
            nps: The ``gs_nyx.nyx_py_sdk`` module (passed in to keep the Nyx
                import at the single site in :meth:`_build`).

        Returns:
            A tuple of ``num_envs`` ``EnvironmentMapAsset`` (empty when the
            ``env_map`` knob is off), one texture-less uniform-radiance sky per
            env, indexable 1:1 by Genesis env index.
        """
        cfg = vision_cfg.get("env_map") or {}
        if not cfg:
            return ()
        kw = {}
        if cfg.get("tint"):
            kw["tint_range"] = tuple(cfg["tint"])
        if cfg.get("multiplier"):
            kw["mult_range"] = tuple(cfg["multiplier"])
        tint, mult = sample_env_map(env.num_envs, device=env.device, **kw)
        tint, mult = tint.tolist(), mult.tolist()
        maps = []
        for i in range(env.num_envs):
            m = nps.EnvironmentMapAsset()
            m.tint = nps.float3(*tint[i])
            m.multiplier = float(mult[i])
            maps.append(m)
        return tuple(maps)
