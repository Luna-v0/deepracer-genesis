"""dr_editor knob-registry tests: coverage, consistency, sweeps, layouts.

The registry wraps the DR catalog (never edits it) and adds the static scene
knobs; these tests pin that the wrap is complete and faithful — every catalog
knob appears exactly once with the catalog's own compatibility matrix, every
static cfg path resolves in the real env config, and the editor annotations
(schedule / liveness / kind) match the verified application sites documented
in ``tools.dr_editor.knobs``. GPU-free: no genesis import anywhere below.
"""

import pytest

from deepracer_genesis.configs.cfgs import get_env_cfg
from deepracer_genesis.randomization.catalog import BY_NAME, CATALOG
from deepracer_genesis.randomization.spaces import FloatRange, IntRange, SymRange
from deepracer_genesis.tools.dr_editor.knobs import (
    REGISTRY,
    SCENE_KNOBS,
    default_layout,
    dr_for_value,
    get,
    sweep_values,
)

# kinds realizable by pure-torch replay on a raw frame (no sim needed)
OFFLINE_KINDS = {"aug_range", "aug_scalar", "world_color", "pixel_noise"}


# ------------------------------------------------------------- 1. coverage

def test_registry_covers_catalog_and_scene_knobs_exactly_once():
    catalog_names = {k.name for k in CATALOG}
    scene_names = {k.name for k in SCENE_KNOBS}
    assert len(scene_names) == len(SCENE_KNOBS)          # statics unique
    assert not catalog_names & scene_names               # no shadowing
    assert set(REGISTRY) == catalog_names | scene_names
    assert len(REGISTRY) == len(CATALOG) + len(SCENE_KNOBS)


# -------------------------------------------------- 2. annotation matrix

def test_offline_liveness_iff_offline_kind():
    for knob in REGISTRY.values():
        assert (knob.liveness == "offline") == (knob.kind in OFFLINE_KINDS), knob.name


def test_aug_kinds_partition_the_image_layer():
    image_names = {k.name for k in CATALOG if k.layer == "image"}
    aug_range = {k.name for k in REGISTRY.values() if k.kind == "aug_range"}
    aug_scalar = {k.name for k in REGISTRY.values() if k.kind == "aug_scalar"}
    assert aug_range == {"brightness", "contrast", "saturation", "gamma"}
    assert aug_range | aug_scalar == image_names
    assert not aug_range & aug_scalar


EXPECTED_SCHEDULE = {
    # verified application sites (see knobs.py module docstring)
    "mount": "per_run",           # base_env._init_buffers, once per run
    "env_map": "per_run",         # Nyx bakes at scene.build
    "world_color": "per_episode",  # reset_idx -> resample_appearance
    "aug_range": "per_step",      # vision_env._observe_camera
    "aug_scalar": "per_step",
    "pixel_noise": "per_step",
    "action": "per_step",
    "static": "build",
}

EXPECTED_LIVENESS = {
    "aug_range": "offline", "aug_scalar": "offline",
    "world_color": "offline", "pixel_noise": "offline",
    "mount": "reroll", "action": "reroll",
    "physics": "rebuild", "env_map": "rebuild", "static": "rebuild",
}


def test_schedules_match_verified_application_sites():
    for knob in REGISTRY.values():
        if knob.kind == "physics":
            # geometry (track_width_scale) is the rulebook's per-episode
            # scale; catalog-layer physics is applied once, before stepping
            expected = ("per_episode" if knob.source.layer == "geometry"
                        else "per_run")
        else:
            expected = EXPECTED_SCHEDULE[knob.kind]
        assert knob.schedule == expected, knob.name


def test_liveness_matches_kind():
    for knob in REGISTRY.values():
        assert knob.liveness == EXPECTED_LIVENESS[knob.kind], knob.name


# --------------------------------------- 3. catalog matrix mirrored as-is

def test_wrapped_knobs_mirror_catalog_matrix():
    for name, source in BY_NAME.items():
        wrapped = REGISTRY[name]
        assert wrapped.source is source
        assert wrapped.modalities == source.modalities
        assert wrapped.renderers == source.renderers
        assert wrapped.cfg_path == source.cfg_key
        assert wrapped.space is source.space
        assert wrapped.note == source.note


def test_statics_have_no_catalog_source():
    for knob in SCENE_KNOBS:
        assert knob.source is None
        assert knob.kind == "static"


# ------------------------------------------------ 4. static cfg paths real

@pytest.mark.parametrize("knob", SCENE_KNOBS, ids=lambda k: k.name)
def test_static_cfg_path_resolves_in_env_cfg(knob):
    cfg = get_env_cfg(vision=True, topdown=True)
    node = cfg
    for part in knob.cfg_path.split("."):
        assert part in node, f"{knob.name}: {knob.cfg_path!r} missing {part!r}"
        node = node[part]


# --------------------------------------------------------- 5. sweep_values

def test_sweep_float_range_evenly_spaced_with_endpoints():
    knob = get("brightness")                   # FloatRange(0.7, 1.3)
    vals = sweep_values(knob, 5)
    assert vals == pytest.approx([0.7, 0.85, 1.0, 1.15, 1.3])
    assert vals[0] == knob.space.lo
    assert vals[-1] == pytest.approx(knob.space.hi)


def test_sweep_int_range_returns_ints_including_hi():
    knob = get("delay_steps")                  # IntRange(0, 3)
    assert isinstance(knob.space, IntRange)
    vals = sweep_values(knob, 4)
    assert vals == [0, 1, 2, 3]
    assert all(isinstance(v, int) for v in vals)
    sparse = sweep_values(knob, 2)
    assert sparse == [0, 3]                    # hi always present


def test_sweep_sym_range_sweeps_zero_to_magnitude():
    knob = get("mass_shift")                   # SymRange(0.05)
    assert isinstance(knob.space, SymRange)
    vals = sweep_values(knob, 3)
    assert vals == pytest.approx([0.0, 0.025, 0.05])


def test_sweep_single_point_is_hi():
    assert sweep_values(get("brightness"), 1) == [get("brightness").space.hi]
    assert sweep_values(get("mass_shift"), 1) == [get("mass_shift").space.m]


def test_sweep_without_space_raises():
    knob = get("background_color")             # static, space=None
    assert knob.space is None
    with pytest.raises(ValueError, match="no numeric space"):
        sweep_values(knob, 3)


# ---------------------------------------------------------- 6. dr_for_value

def test_dr_for_value_aug_range_is_degenerate():
    assert dr_for_value(get("brightness"), 0.9) == {
        "image_aug": {"brightness": (0.9, 0.9)}}


def test_dr_for_value_aug_scalar_types():
    dr = dr_for_value(get("latency_steps"), 2.0)
    assert dr == {"image_aug": {"latency_steps": 2}}
    assert isinstance(dr["image_aug"]["latency_steps"], int)
    dr = dr_for_value(get("blur"), 0.3)
    assert dr == {"image_aug": {"blur": 0.3}}
    assert isinstance(dr["image_aug"]["blur"], float)


def test_dr_for_value_world_color_and_pixel_noise():
    assert dr_for_value(get("world_color"), 0.4) == {"world_color": 0.4}
    assert dr_for_value(get("pixel_noise"), 0.02) == {"pixel_noise": 0.02}


def test_dr_for_value_refuses_non_offline_knob():
    with pytest.raises(ValueError, match="not offline-replayable"):
        dr_for_value(get("camera_pitch_jitter"), 1.0)


# --------------------------------------------------- 7. layouts and get()

@pytest.mark.parametrize("name,layout", [
    ("brightness", "filmstrip"),       # per_step
    ("pixel_noise", "filmstrip"),      # per_step
    ("world_color", "contact_sheet"),  # per_episode
    ("friction", "contact_sheet"),     # per_run
    ("env_map_tint", "contact_sheet"),  # per_run
    ("light_intensity", "ab"),         # build
])
def test_default_layout(name, layout):
    assert default_layout(get(name)) == layout


def test_get_unknown_knob_lists_known_names():
    with pytest.raises(KeyError, match="unknown knob 'nope'") as excinfo:
        get("nope")
    assert "brightness" in str(excinfo.value)  # the known list is in the message
    assert "light_intensity" in str(excinfo.value)
