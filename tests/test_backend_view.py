"""CPU backend + view config plumbing (Part M) — no sim needed."""

import warnings

import pytest

from deepracer_genesis.configs.cfgs import get_env_cfg
from deepracer_genesis.experiment import (
    AsymmetricCameraPolicy,
    CameraEnvironment,
    FeatureEnvironment,
    VectorPolicy,
)
from deepracer_genesis.experiment.spec import SpecError


def test_get_env_cfg_carries_backend_and_view():
    c = get_env_cfg(backend="cpu", view="gui")
    assert c["sim"]["backend"] == "cpu"
    assert c["sim"]["view"] == "gui"
    d = get_env_cfg()                       # defaults
    assert d["sim"]["backend"] == "gpu" and d["sim"]["view"] == "none"


def test_env_stage_routes_backend_view_to_spec():
    s = (FeatureEnvironment(num_envs=8, backend="cpu", view="gui")
         >> VectorPolicy()).build()
    assert s.env.backend == "cpu" and s.env.view == "gui"


def test_ensure_init_rejects_bad_backend():
    from deepracer_genesis._gs import ensure_init
    with pytest.raises(ValueError, match="gpu.*cpu"):
        ensure_init("tpu")


def _camera_cpu_spec(tracks=("reinvent_base",)):
    return (CameraEnvironment(backend="cpu", tracks=tracks)
            >> AsymmetricCameraPolicy(actor_keys=("camera",),
                                      critic_keys=("camera", "state")))


def test_camera_on_cpu_builds_with_slow_path_warning():
    """Part M.2: camera+cpu is now supported via the per-env rasterizer — it
    builds (no SpecError) and warns that it is the unbatched slow path."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        spec = _camera_cpu_spec().build()
        assert any("RasterizerObsRenderer" in str(x.message)
                   and "slower" in str(x.message) for x in w)
    assert spec.env.backend == "cpu" and spec.env.modality == "camera"


def test_camera_on_cpu_multitrack_builds():
    """Spatial tiling is renderer-agnostic, so the per-env rasterizer handles
    multi-track too: each track sits on its own world tile and a car's camera
    only ever frames its own. The rasterizer still walks every tile's geometry,
    so the cost grows with the track count -- hence the slow-path warning."""
    spec = _camera_cpu_spec(tracks=("reinvent_base", "reInvent2019_track")).build()
    assert len(spec.env.tracks) == 2
    assert spec.env.backend == "cpu" and spec.env.modality == "camera"


def test_builder_routes_camera_cpu_to_rasterizer():
    from deepracer_genesis.experiment.builder import Builder
    cpu = Builder(_camera_cpu_spec().build()).sim_cfg()
    assert cpu["vision"]["vision_renderer"] == "rasterizer"
    gpu = Builder((CameraEnvironment(backend="gpu")
                   >> AsymmetricCameraPolicy(actor_keys=("camera",),
                                             critic_keys=("camera", "state"))
                   ).build()).sim_cfg()
    assert gpu["vision"]["vision_renderer"] == "batch"


def test_make_renderer_selects_rasterizer_on_cpu_flag():
    import genesis as gs
    from deepracer_genesis.envs.renderers import (
        MadronaRenderer, NullRenderer, NyxRenderer, RasterizerObsRenderer,
        make_renderer,
    )
    base = {"vision": True}
    assert isinstance(make_renderer({**base, "vision_renderer": "rasterizer"}),
                      RasterizerObsRenderer)
    assert isinstance(make_renderer({**base, "vision_renderer": "batch"}), MadronaRenderer)
    assert isinstance(make_renderer({**base, "vision_renderer": "nyx"}), NyxRenderer)
    assert isinstance(make_renderer(base), MadronaRenderer)          # default
    assert isinstance(make_renderer({"vision": False}), NullRenderer)
    # the CPU strategy uses a plain Rasterizer scene renderer, not a BatchRenderer
    r = RasterizerObsRenderer()
    assert r.has_camera and r._scene_batch_renderer is False
    assert isinstance(r.scene_renderer(), gs.renderers.Rasterizer)


def test_gui_large_batch_warns_but_builds():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        (FeatureEnvironment(num_envs=1024, view="gui") >> VectorPolicy()).build()
        assert any("interactive window" in str(x.message) for x in w)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        (FeatureEnvironment(num_envs=16, view="gui") >> VectorPolicy()).build()
        assert not any("interactive window" in str(x.message) for x in w)


def test_gui_plus_gpu_dr_builds_on_both_backends():
    """view='gui' + physics DR builds on GPU and CPU: the old crash was a
    quadrants 1.0.2 allocator bug, fixed in genesis>=1.2.3 — no guard needed."""
    from deepracer_genesis.experiment import DomainRandomizationPhysics

    for backend in ("gpu", "cpu"):
        spec = (FeatureEnvironment(num_envs=16, view="gui", backend=backend)
                >> DomainRandomizationPhysics()
                >> VectorPolicy(keys=("state",))).build()
        assert spec.env.view == "gui" and spec.env.backend == backend
