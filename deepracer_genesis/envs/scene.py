"""Assemble the Genesis simulation scene for the DeepRacer env.

Entry point ``build_scene(env, ...)`` builds the scene, plane, car, and track morph(s).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import genesis as gs

from .entities import Car
from .renderers import NyxRenderer, make_renderer

if TYPE_CHECKING:
    from .base_env import DeepRacerEnv


def _realtime_factor(env_cfg: dict) -> float | None:
    """Viewer pacing factor from the sim cfg; <=0 means uncapped (None).

    Args:
        env_cfg: The env config; reads ``sim.realtime_factor`` (default 1.0).

    Returns:
        The factor for ViewerOptions, or None to step as fast as compute allows.
    """
    rf = env_cfg["sim"].get("realtime_factor", 1.0)
    return None if rf is None or rf <= 0 else float(rf)


def build_scene(env: "DeepRacerEnv", env_cfg: dict, show_viewer: bool) -> None:
    """Create the scene, add the plane/car/track morph(s), and build it.

    Populates ``env.renderer``, ``env.scene``, ``env.plane``, ``env.car``, and ``env.track_entity``.

    Args:
        env: The DeepRacer env being constructed; its ``track`` and
            ``num_envs`` are read, and the scene attributes above are written.
        env_cfg: Scene configuration, including ``dt``, optional
            ``randomize`` (enables batched dofs/links physics info for domain
            randomization), and optional ``background_color``, ``field_color``.
        show_viewer: Whether to open an interactive Genesis viewer window.

    Raises:
        NotImplementedError: If more than one track morph is requested with the
            Nyx renderer (heterogeneous tracks are unsupported there), or with
            any camera-capable renderer (genesis 1.2.1 never feeds
            ``vgeom.active_envs_mask`` to the batch renderer, so every env would
            render all track variants superimposed with z-fighting;
            feature-mode multi-track without rendering is fine).
    """
    env.renderer = make_renderer(env_cfg["vision"])
    randomize = bool(env_cfg["rand"]["randomize"])

    env.scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=env_cfg["sim"]["dt"], substeps=1),
        rigid_options=gs.options.RigidOptions(
            dt=env_cfg["sim"]["dt"],
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
            # per-env dofs/links properties (kp/kv/armature/mass/COM DR) need
            # batched physics info, per the Genesis DR guide
            batch_dofs_info=randomize,
            batch_links_info=randomize,
        ),
        vis_options=gs.options.VisOptions(
            shadow=False,
            ambient_light=(0.35, 0.35, 0.35),
            background_color=tuple(env_cfg["vision"].get("background_color", (0.55, 0.72, 0.9))),
            # renderers that hold ONE batched camera need each env's rigid bodies
            # rendered in isolation, so no foreign car enters frame.
            env_separate_rigid=getattr(env.renderer, "env_separate_rigid", False),
        ),
        renderer=env.renderer.scene_renderer(),
        show_viewer=show_viewer,
        # viewer pacing: realtime_factor x real time (None/<=0 = uncapped, i.e.
        # step as fast as compute allows). Only matters when the viewer is shown.
        viewer_options=gs.options.ViewerOptions(realtime_factor=_realtime_factor(env_cfg)),
    )

    # green ground doubles as the field: some DAE ground materials render
    # transparent under Madrona, and this is what shows through. Must be a
    # surface color — Madrona does not sample ImageTexture on primitives.
    fc = env_cfg["vision"].get("field_color", (0.30, 0.48, 0.32))
    env.plane = env.scene.add_entity(
        gs.morphs.Plane(pos=(0, 0, -0.001)),
        surface=gs.surfaces.Rough(color=(*fc, 1.0)),
    )

    env.car = Car(env.scene, merge_fixed_links=env.renderer.merge_fixed_links)

    # Nyx cannot read DAE; use the OBJ conversions (same geometry/textures)
    nyx = isinstance(env.renderer, NyxRenderer)
    mesh_paths = env.track.obj_paths if nyx else env.track.mesh_paths
    tiled = float(getattr(env.track, "grid_spacing", 0.0)) > 0.0
    if nyx and len(mesh_paths) > 1:
        raise NotImplementedError("heterogeneous tracks are not supported with the Nyx renderer")

    if tiled:
        # Part O spatial tiling: load ALL variant meshes into EVERY env as
        # homogeneous geometry, each translated to its own world tile
        # (env.track.variant_offset). Cars spawn on their home tile (the offset
        # falls out of the offset waypoints/spawn_pose) and tiles are spaced
        # beyond camera reach, so each camera sees only its home track — no
        # z-fighting, no reliance on the broken heterogeneous-morph render path.
        offsets = env.track.variant_offset.tolist()
        env.track_entity = [
            env.scene.add_entity(gs.morphs.Mesh(
                file=p, pos=(offsets[k][0], offsets[k][1], 0.0),
                fixed=True, collision=False))
            for k, p in enumerate(mesh_paths)]
    else:
        if env.renderer.has_camera and len(mesh_paths) > 1:
            raise NotImplementedError(
                "heterogeneous multi-track CAMERA training is unsound under the "
                "batch renderer: genesis 1.2.1 never feeds vgeom.active_envs_mask "
                "to it, so every env renders ALL track variants superimposed "
                "(z-fighting). Use spatial tiling (track_grid_spacing) for camera "
                "multi-track; feature-mode multi-track (no rendering) is fine.")
        track_morphs = [gs.morphs.Mesh(file=p, fixed=True, collision=False)
                        for p in mesh_paths]
        # a list of morphs makes the entity heterogeneous: each parallel env
        # simulates (and renders) one geometry variant
        env.track_entity = env.scene.add_entity(
            track_morphs if len(track_morphs) > 1 else track_morphs[0])

    # renderer adds its cameras / lights / sensors, then we build the scene
    env.renderer.build(env, env_cfg["vision"])
    env.scene.build(n_envs=env.num_envs)
