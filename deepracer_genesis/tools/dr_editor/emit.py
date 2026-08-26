"""Round-trip emitters: tuned values -> pasteable stage code / JSON presets.

The editor never edits specs in place (run identity is content-hashed). What
comes out of a tuning session is (a) a pasteable ``>>`` stage line, (b) a DR
parameter dict in spec shape, and (c) optionally a JSON preset under
``configs/dr_presets/`` that :func:`load_preset` turns back into stages —
identical resulting spec values hash identically by construction.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from .knobs import EditorKnob

# cfg["rand"] leaf -> DomainRandomizationPhysics parameter name
_PHYSICS_PARAMS = {
    "friction_range": "friction", "mass_shift_kg": "mass",
    "com_shift_m": "com", "steer_kp_scale": "gains",
    "armature_range": "armature", "track_width_scale": "track_width",
}
_MOUNT_PARAMS = {"camera_pitch_jitter": "pitch_deg", "camera_pos_jitter": "pos_m"}

PRESET_DIR = os.path.join("configs", "dr_presets")


def _fmt(v) -> str:
    """Format one value as it should appear in emitted stage code.

    Args:
        v: A scalar, tuple, or dict value.

    Returns:
        Its Python-literal source form (tuples stay tuples).
    """
    if isinstance(v, (tuple, list)):
        return "(" + ", ".join(_fmt(x) for x in v) + ")"
    return repr(v)


def emit_knob_code(knob: EditorKnob, value) -> str:
    """One pasteable stage line that reproduces ``value`` for ``knob``.

    Args:
        knob: The knob being emitted.
        value: The picked value — a ``(lo, hi)`` tuple for range-shaped knobs,
            a scalar magnitude otherwise.

    Returns:
        A single ``DomainRandomization*`` stage expression (or, for static
        scene knobs, an ``extra_cfg`` fragment comment — statics have no
        stage).
    """
    if knob.kind in ("aug_range", "aug_scalar", "pixel_noise"):
        key = "pixel_noise" if knob.kind == "pixel_noise" else knob.name
        return f"DomainRandomizationCamera({key}={_fmt(value)})"
    if knob.kind == "mount":
        return ("DomainRandomizationCamera(camera_jitter="
                f"{{{_MOUNT_PARAMS[knob.name]!r}: {_fmt(value)}}})")
    if knob.kind == "world_color":
        return f"DomainRandomizationTrackAppearance(strength={_fmt(value)})"
    if knob.kind == "env_map":
        param = ("env_map_tint" if knob.name == "env_map_tint"
                 else "env_map_multiplier")
        return f"DomainRandomizationTrackAppearance({param}={_fmt(value)})"
    if knob.kind == "physics":
        leaf = knob.cfg_path.rsplit(".", 1)[1]
        return f"DomainRandomizationPhysics({_PHYSICS_PARAMS[leaf]}={_fmt(value)})"
    if knob.kind == "action":
        return f"DomainRandomizationActions({knob.name}={_fmt(value)})"
    # static scene knob: no stage exists; the value rides the cfg
    section, leaf = knob.cfg_path.split(".", 1)
    return (f"# static scene knob — no DR stage; pass at build time:\n"
            f"# Builder(spec).sim(extra_cfg={{{section!r}: "
            f"{{{leaf!r}: {_fmt(value)}}}}})")


def emit_pipeline_code(dr: dict) -> str:
    """Emit the full ``>>`` DR tail reproducing a DR-parameter dict.

    Args:
        dr: DR parameters in spec shape (see ``pipeline.dr_from_spec``):
            optional keys ``image_aug``, ``world_color``, ``pixel_noise``,
            ``env_map``, ``camera_jitter``, ``physics``, ``action_dr``.

    Returns:
        Newline-joined stage expressions (empty string when nothing is set),
        ready to paste after an Environment stage.
    """
    lines: list[str] = []
    cam_kwargs = {k: v for k, v in dict(dr.get("image_aug", {})).items() if v}
    if dr.get("pixel_noise"):
        cam_kwargs["pixel_noise"] = dr["pixel_noise"]
    if dr.get("camera_jitter"):
        cam_kwargs["camera_jitter"] = dict(dr["camera_jitter"])
    if cam_kwargs:
        args = ", ".join(f"{k}={_fmt(v)}" for k, v in cam_kwargs.items())
        lines.append(f"DomainRandomizationCamera({args})")
    app_kwargs = {}
    if dr.get("world_color"):
        app_kwargs["strength"] = dr["world_color"]
    env_map = dict(dr.get("env_map", {}))
    if env_map.get("tint"):
        app_kwargs["env_map_tint"] = tuple(env_map["tint"])
    if env_map.get("multiplier"):
        app_kwargs["env_map_multiplier"] = tuple(env_map["multiplier"])
    if app_kwargs:
        args = ", ".join(f"{k}={_fmt(v)}" for k, v in app_kwargs.items())
        lines.append(f"DomainRandomizationTrackAppearance({args})")
    phys = {k: v for k, v in dict(dr.get("physics", {})).items() if v}
    if phys:
        args = ", ".join(f"{_PHYSICS_PARAMS[k]}={_fmt(v)}"
                         for k, v in phys.items() if k in _PHYSICS_PARAMS)
        lines.append(f"DomainRandomizationPhysics({args})")
    act = {k: v for k, v in dict(dr.get("action_dr", {})).items() if v}
    if act:
        args = ", ".join(f"{k}={_fmt(v)}" for k, v in act.items())
        lines.append(f"DomainRandomizationActions({args})")
    return "\n>> ".join(lines)


def save_preset(name: str, dr: dict, *, root: str = ".") -> str:
    """Persist a DR-parameter dict as a JSON preset.

    Args:
        name: Preset name (the file becomes ``configs/dr_presets/<name>.json``).
        dr: DR parameters in spec shape.
        root: Repo root the preset directory hangs off.

    Returns:
        The written preset path.
    """
    path = os.path.join(root, PRESET_DIR, f"{name}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"name": name, "dr": dr}, f, indent=2, sort_keys=True)
    return path


def load_preset(name: str, *, root: str = ".") -> list:
    """Load a preset back as ready-to-``>>`` stage objects.

    The stages are built from the stored values, so an identical preset
    produces an identical spec — and therefore an identical run id.

    Args:
        name: Preset name (or a direct path to a preset JSON).
        root: Repo root the preset directory hangs off.

    Returns:
        Stage instances (possibly empty) to fold after an Environment stage.

    Raises:
        FileNotFoundError: If no such preset exists.
    """
    from ...experiment.stages import (DomainRandomizationActions,
                                      DomainRandomizationCamera,
                                      DomainRandomizationPhysics,
                                      DomainRandomizationTrackAppearance)

    path = name if name.endswith(".json") else os.path.join(
        root, PRESET_DIR, f"{name}.json")
    with open(path) as f:
        dr = json.load(f)["dr"]

    def _t(v):
        return tuple(v) if isinstance(v, list) else v

    stages: list = []
    cam = {k: _t(v) for k, v in dict(dr.get("image_aug", {})).items() if v}
    if dr.get("pixel_noise"):
        cam["pixel_noise"] = dr["pixel_noise"]
    if dr.get("camera_jitter"):
        cam["camera_jitter"] = dict(dr["camera_jitter"])
    if cam:
        stages.append(DomainRandomizationCamera(**cam))
    app = {}
    if dr.get("world_color"):
        app["strength"] = dr["world_color"]
    env_map = dict(dr.get("env_map", {}))
    if env_map.get("tint"):
        app["env_map_tint"] = _t(env_map["tint"])
    if env_map.get("multiplier"):
        app["env_map_multiplier"] = _t(env_map["multiplier"])
    if app:
        stages.append(DomainRandomizationTrackAppearance(**app))
    phys = {k: _t(v) for k, v in dict(dr.get("physics", {})).items() if v}
    if phys:
        stages.append(DomainRandomizationPhysics(
            **{_PHYSICS_PARAMS[k]: v for k, v in phys.items()
               if k in _PHYSICS_PARAMS}))
    act = {k: v for k, v in dict(dr.get("action_dr", {})).items() if v}
    if act:
        stages.append(DomainRandomizationActions(**act))
    return stages
