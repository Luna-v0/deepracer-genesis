"""Locks the DR knob catalog's compatibility matrix to the runtime it maps.

Pure, no-sim checks that ``randomization.catalog`` stays truthful: every knob
declares valid modality/renderer axes, the name/cfg-key indexes are coherent,
and — the honesty core — the catalogued cfg leaves match the exact key sets
the runtime consumes (``DomainRandomizationCamera`` fields, the image-aug /
vision-env source literals, :data:`NEUTRAL_PHYSICS`, ``ActionDRSpec``). Adding
a knob the runtime ignores, or a runtime key the catalog omits (the historical
"cutout"/"noise" gap), fails here instead of silently sampling nothing.

``envs.vision_env`` pulls genesis, so its key usage is checked against its
SOURCE TEXT (never imported); ``envs.track``-free modules are imported normally.
"""

import dataclasses
import pathlib
import re
import typing

import deepracer_genesis.randomization.catalog as catalog
from deepracer_genesis.experiment.spec import NEUTRAL_PHYSICS, ActionDRSpec
from deepracer_genesis.experiment.stages import DomainRandomizationCamera
from deepracer_genesis.randomization.catalog import BY_NAME, CATALOG, Layer, by_layer

_PKG = pathlib.Path(catalog.__file__).resolve().parent.parent
_MODALITIES = frozenset({"camera", "feature"})
_RENDERERS = frozenset({"madrona", "nyx", "rasterizer"})


# ----------------------------------------------------------------- helpers

def _leaves(prefix, layers=None):
    """Collect the cfg-key leaf names of catalog knobs under a prefix.

    Args:
        prefix: Dotted ``cfg_key`` prefix to filter on (e.g. ``"rand."``).
        layers: Optional collection of layer names; when given, only knobs
            whose ``layer`` is in it are kept.

    Returns:
        The set of final dotted components (leaves) of the matching knobs'
        ``cfg_key`` values.
    """
    return {k.cfg_key.rsplit(".", 1)[1] for k in CATALOG
            if k.cfg_key.startswith(prefix)
            and (layers is None or k.layer in layers)}


# ------------------------------------------------- 1. axis validity
def test_every_knob_declares_valid_modalities_and_renderers():
    """Modalities non-empty and known; camera knobs name >=1 known renderer."""
    for knob in CATALOG:
        assert knob.modalities, f"knob {knob.name!r} has empty modalities"
        assert knob.modalities <= _MODALITIES, (
            f"knob {knob.name!r} has unknown modalities "
            f"{sorted(knob.modalities - _MODALITIES)}")
        assert knob.renderers <= _RENDERERS, (
            f"knob {knob.name!r} has unknown renderers "
            f"{sorted(knob.renderers - _RENDERERS)}")
        if "camera" in knob.modalities:
            assert knob.renderers, (
                f"camera knob {knob.name!r} has empty renderers — it could "
                "never build")


# ------------------------------------------------- 2. index coherence
def test_knob_names_and_cfg_keys_unique():
    """No two knobs share a name or a cfg landing key."""
    names = [k.name for k in CATALOG]
    cfg_keys = [k.cfg_key for k in CATALOG]
    assert len(names) == len(set(names))
    assert len(cfg_keys) == len(set(cfg_keys))


def test_by_name_and_by_layer_mirror_catalog():
    """BY_NAME and by_layer() are exact views of CATALOG, covering it fully."""
    assert BY_NAME == {k.name: k for k in CATALOG}
    layers = typing.get_args(Layer)
    seen = []
    for layer in layers:
        knobs = by_layer(layer)
        assert knobs == [k for k in CATALOG if k.layer == layer]
        seen.extend(knobs)
    # every knob's layer is one of the Layer literals (nothing orphaned)
    assert sorted(k.name for k in seen) == sorted(k.name for k in CATALOG)


# ------------------------------------------------- 3. image-key lock
def test_image_leaves_match_camera_stage_fields():
    """obs_dr.image_aug.* leaves == DomainRandomizationCamera's aug fields.

    ``pixel_noise`` and ``camera_jitter`` ride the same stage but land in
    other cfg locations (they are catalogued as visual knobs), so they are
    excluded from the field side.
    """
    fields = {f.name for f in dataclasses.fields(DomainRandomizationCamera)}
    fields -= {"pixel_noise", "camera_jitter"}
    assert _leaves("obs_dr.image_aug.") == fields


def test_image_leaves_are_consumed_by_the_runtime_source():
    """Every image leaf appears as a string literal where it is applied.

    Batch augs live in ``randomization/image_aug.py``; the stateful temporal
    keys (``latency_steps``/``frame_drop``) are consumed in
    ``envs/vision_env.py``. Read as text: vision_env imports genesis.
    """
    source = ((_PKG / "randomization" / "image_aug.py").read_text()
              + (_PKG / "envs" / "vision_env.py").read_text())
    for leaf in sorted(_leaves("obs_dr.image_aug.")):
        assert re.search(r"[\"']%s[\"']" % re.escape(leaf), source), (
            f"catalogued image knob leaf {leaf!r} is never read by "
            "image_aug.py or vision_env.py — it would silently do nothing")


# ------------------------------------------------- 4. physics neutrality lock
def test_neutral_physics_covers_exactly_the_rand_physics_geometry_leaves():
    """NEUTRAL_PHYSICS keys == the physics/geometry knobs' rand.* leaves."""
    assert set(NEUTRAL_PHYSICS) == _leaves("rand.", layers=("physics", "geometry"))


# ------------------------------------------------- 5. action-DR lock
def test_action_dr_leaves_match_action_dr_spec_fields():
    """action_dr.* leaves == the ActionDRSpec dataclass field names."""
    fields = {f.name for f in dataclasses.fields(ActionDRSpec)}
    assert _leaves("action_dr.") == fields


# ------------------------------------------------- 6. visual rand.* lock
def test_visual_rand_leaves_are_exactly_the_mount_jitter_keys():
    """The only visual knobs landing in cfg['rand'] are the two mount knobs."""
    assert _leaves("rand.", layers=("visual",)) == {
        "camera_pitch_jitter_deg", "camera_pos_jitter_m"}
