"""dr_editor emit/prove/frames tests (GPU-free, no Genesis).

Covers the round-trip emitters (``emit_knob_code`` per kind,
``emit_pipeline_code``), the JSON preset round trip landing on identical
spec values AND identical run ids, the run-identity lock with the editor
imported, the spec-validation refusal checks (``prove_refusal`` — the
matrix-drift detector), and the ``FrameBank`` disk format.
"""

import json
import sys
import warnings

import numpy as np
import pytest
import torch

from deepracer_genesis.experiment import (
    PPO,
    AsymmetricCameraPolicy,
    CameraEnvironment,
)
from deepracer_genesis.tools.dr_editor.emit import (
    emit_knob_code,
    emit_pipeline_code,
    load_preset,
    save_preset,
)
from deepracer_genesis.tools.dr_editor.frames import FrameBank
from deepracer_genesis.tools.dr_editor.knobs import get
from deepracer_genesis.tools.dr_editor.prove import prove_refusal

# ------------------------------------------------------- emit_knob_code

def test_emit_aug_range_knob():
    """A range-shaped aug knob emits a DomainRandomizationCamera range kwarg."""
    code = emit_knob_code(get("brightness"), (0.7, 1.3))
    assert code == "DomainRandomizationCamera(brightness=(0.7, 1.3))"


def test_emit_world_color_knob():
    """world_color emits a TrackAppearance strength."""
    code = emit_knob_code(get("world_color"), 0.45)
    assert code == "DomainRandomizationTrackAppearance(strength=0.45)"


def test_emit_mount_knob():
    """Mount jitter emits a camera_jitter dict keyed by its param name."""
    code = emit_knob_code(get("camera_pitch_jitter"), 2.0)
    assert code == "DomainRandomizationCamera(camera_jitter={'pitch_deg': 2.0})"


def test_emit_env_map_knob():
    """env_map_tint emits a TrackAppearance tuple kwarg."""
    code = emit_knob_code(get("env_map_tint"), (0.35, 0.75))
    assert code == "DomainRandomizationTrackAppearance(env_map_tint=(0.35, 0.75))"


def test_emit_physics_knob():
    """friction maps its cfg leaf back to the Physics stage param name."""
    code = emit_knob_code(get("friction"), (0.6, 1.4))
    assert code == "DomainRandomizationPhysics(friction=(0.6, 1.4))"


def test_emit_action_knob():
    """steer_noise emits an Actions stage kwarg."""
    code = emit_knob_code(get("steer_noise"), 0.08)
    assert code == "DomainRandomizationActions(steer_noise=0.08)"


def test_emit_static_knob_is_extra_cfg_comment():
    """Statics have no stage: the emission is an extra_cfg fragment comment."""
    code = emit_knob_code(get("light_intensity"), 8.0)
    assert code.startswith("#")
    assert "extra_cfg" in code
    assert "'vision'" in code
    assert "'light_intensity'" in code
    assert "8.0" in code


# --------------------------------------------------- emit_pipeline_code

_FULL_DR = {
    "image_aug": {"brightness": (0.7, 1.3), "blur": 0.4},
    "world_color": 0.5,
    "pixel_noise": 0.02,
    "env_map": {"tint": (0.35, 0.75), "multiplier": (0.8, 1.2)},
    "camera_jitter": {"pitch_deg": 2.0},
    "physics": {"friction_range": (0.6, 1.4)},
    "action_dr": {"steer_noise": 0.05, "speed_noise": 0.0, "delay_steps": 2},
}


def test_emit_pipeline_code_full_dr():
    """Every populated DR family appears as its stage, joined by >>."""
    code = emit_pipeline_code(_FULL_DR)
    lines = code.split("\n>> ")
    assert len(lines) == 4
    assert code.count("\n>> ") == 3
    cam, app, phys, act = lines
    assert cam.startswith("DomainRandomizationCamera(")
    assert "brightness=(0.7, 1.3)" in cam
    assert "blur=0.4" in cam
    assert "pixel_noise=0.02" in cam
    assert "camera_jitter={'pitch_deg': 2.0}" in cam
    assert app.startswith("DomainRandomizationTrackAppearance(")
    assert "strength=0.5" in app
    assert "env_map_tint=(0.35, 0.75)" in app
    assert "env_map_multiplier=(0.8, 1.2)" in app
    assert phys == "DomainRandomizationPhysics(friction=(0.6, 1.4))"
    assert act.startswith("DomainRandomizationActions(")
    assert "steer_noise=0.05" in act
    assert "delay_steps=2" in act
    assert "speed_noise" not in act        # zero values are elided


def test_emit_pipeline_code_empty_dr():
    """An empty DR dict emits nothing."""
    assert emit_pipeline_code({}) == ""


# ------------------------------------------------------ preset round trip

# madrona-compatible preset (no env_map — that is Nyx-only and would be
# refused at build, which test_prove_refusal_* pins separately)
_PRESET_DR = {
    "image_aug": {"brightness": (0.7, 1.3), "blur": 0.4},
    "world_color": 0.5,
    "pixel_noise": 0.02,
    "camera_jitter": {"pitch_deg": 2.0},
    "physics": {"friction_range": (0.6, 1.4)},
    "action_dr": {"steer_noise": 0.05, "delay_steps": 2},
}


def _fold(stages):
    """Fold preset stages into a built camera spec."""
    pipe = CameraEnvironment(num_envs=4)
    for stage in stages:
        pipe = pipe >> stage
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (pipe >> AsymmetricCameraPolicy() >> PPO()).build()


def test_preset_round_trip_restores_spec_values(tmp_path):
    """save -> load -> fold lands the source values (tuples, not lists)."""
    root = str(tmp_path)
    path = save_preset("tuned", _PRESET_DR, root=root)
    assert json.load(open(path))["dr"]["image_aug"]["brightness"] == [0.7, 1.3]

    spec = _fold(load_preset("tuned", root=root))
    assert spec.obs_dr.image_aug["brightness"] == (0.7, 1.3)
    assert isinstance(spec.obs_dr.image_aug["brightness"], tuple)
    assert spec.obs_dr.image_aug["blur"] == 0.4
    assert spec.obs_dr.appearance == {"world_color": 0.5}
    assert spec.obs_dr.pixel_noise == 0.02
    assert spec.obs_dr.camera_jitter == {"pitch_deg": 2.0}
    assert spec.obs_dr.physics["friction_range"] == (0.6, 1.4)
    assert isinstance(spec.obs_dr.physics["friction_range"], tuple)
    assert spec.action_dr.steer_noise == 0.05
    assert spec.action_dr.speed_noise == 0.0
    assert spec.action_dr.delay_steps == 2


def test_preset_loads_hash_identically(tmp_path):
    """Two loads of one preset build specs with the SAME run id."""
    root = str(tmp_path)
    save_preset("tuned", _PRESET_DR, root=root)
    a = _fold(load_preset("tuned", root=root))
    b = _fold(load_preset("tuned", root=root))
    assert a.id() == b.id()


def test_run_identity_locked_with_dr_editor_imported():
    """Importing the editor must not shift existing run ids (identity lock)."""
    spec = (CameraEnvironment() >> AsymmetricCameraPolicy() >> PPO()).build()
    assert spec.id() == "3ddf1ade78ca"


# ---------------------------------------------------------- prove_refusal

def test_prove_refusal_is_genesis_free():
    """The refusal probe is pure spec validation: it must succeed with
    genesis imports actively BLOCKED (order-independent in the full suite,
    where other modules may already have pulled genesis in)."""
    saved = {n: sys.modules.pop(n) for n in list(sys.modules)
             if n == "genesis" or n.startswith("genesis.")}

    class _BlockGenesis:
        """Meta-path finder that refuses any genesis import."""

        def find_spec(self, name, path=None, target=None):
            if name == "genesis" or name.startswith("genesis."):
                raise ImportError("genesis import attempted in a GPU-free "
                                  "prove_refusal check")
            return None

    blocker = _BlockGenesis()
    sys.meta_path.insert(0, blocker)
    try:
        v = prove_refusal(get("track_width_scale"), "madrona")
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)
    assert v.check == "declared_refusal"
    assert v.passed is True


def test_prove_refusal_unsupported_combos_pass():
    """Unsupported knob/renderer combos are refused at build => PASS."""
    assert prove_refusal(get("camera_pitch_jitter"), "nyx").passed is True
    assert prove_refusal(get("env_map_tint"), "madrona").passed is True


def test_prove_refusal_supported_combo_fails():
    """A SUPPORTED combo builds fine, so the refusal check FAILS — the
    asymmetry that detects compatibility-matrix drift."""
    v = prove_refusal(get("env_map_tint"), "nyx")
    assert v.check == "declared_refusal"
    assert v.passed is False


# -------------------------------------------------------------- FrameBank

def test_frame_bank_round_trip(tmp_path):
    """A hand-written bank loads with exact uint8/255 pipeline frames."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(2, 3, 4, 6, 3), dtype=np.uint8)
    pose = np.array([[0, 0.0, 0.0], [5, 0.25, 0.1]], dtype=np.float32)
    np.savez(tmp_path / "frames.npz", image=image, pose=pose)
    meta = {"raw": True, "track": "reinvent", "renderer": "batch",
            "camera_res": [6, 4], "num_envs": 3,
            "poses": [list(map(float, p)) for p in pose]}
    (tmp_path / "meta.json").write_text(json.dumps(meta))

    bank = FrameBank(str(tmp_path))
    assert len(bank) == 2
    assert bank.meta["num_envs"] == 3
    np.testing.assert_array_equal(bank.poses, pose)

    raw = bank.raw(1)
    assert raw.shape == (3, 3, 4, 6)
    assert raw.dtype == torch.float32
    assert raw.min().item() >= 0.0 and raw.max().item() <= 1.0
    expected = torch.from_numpy(image[1]).permute(0, 3, 1, 2).float() / 255.0
    assert torch.equal(raw, expected)
