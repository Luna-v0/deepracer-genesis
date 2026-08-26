"""The editor's knob registry: catalog knobs + static scene knobs, annotated.

``randomization.catalog`` is the source of truth for what gets randomized and
where it may act (``Knob.modalities`` / ``Knob.renderers``). The editor needs
three more facts per knob — its **schedule** (which axis it varies on), its
**liveness** (offline replay / live re-roll / scene rebuild), and its **kind**
(how a sweep value is applied) — plus the *static* scene knobs a scene editor
edits but the catalog rightly excludes (they are never randomized). This
module wraps the catalog rather than changing it, so the catalog stays a
truthful list of randomized knobs.

Verified schedules (source, not docs): image aug per step
(``vision_env._observe_camera``); world colour resampled per episode
(``base_env.reset_idx`` -> ``resample_appearance``); mount jitter and physics
ONCE per run (``base_env._init_buffers``); Nyx env maps baked at
``scene.build``; statics fixed at build.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from ...randomization.catalog import BY_NAME, CATALOG, Knob
from ...randomization.spaces import FloatRange, IntRange, Space

Schedule = Literal["per_step", "per_episode", "per_run", "build"]
Liveness = Literal["offline", "reroll", "rebuild"]
Kind = Literal["aug_range", "aug_scalar", "world_color", "pixel_noise",
               "env_map", "mount", "physics", "action", "static"]

# image-aug keys whose cfg value is a (lo, hi) range — a sweep point v becomes
# the degenerate range (v, v), i.e. fully deterministic; every other aug key
# is a scalar magnitude the env samples within, so sweep tiles are seeded
_AUG_RANGE_KEYS = frozenset({"brightness", "contrast", "saturation", "gamma"})

_ALL_RENDERERS = frozenset({"madrona", "nyx", "rasterizer"})


@dataclass(frozen=True)
class EditorKnob:
    """One editable quantity: a catalog knob or a static scene knob.

    Attributes:
        name: Knob name (catalog name, or the cfg leaf for statics).
        schedule: Axis the knob varies on (``build`` = fixed for the run and
            decided at scene build).
        liveness: How the editor realizes a new value — ``offline`` (pure
            torch replay on a raw frame, no sim needed), ``reroll`` (poke the
            live env and re-render), ``rebuild`` (a new scene is required).
        kind: How a sweep/grid value is applied (see module constants).
        cfg_path: Dotted env-cfg path the value lands in (catalog ``cfg_key``
            for randomized knobs; the literal cfg path for statics).
        space: Suggested sweep range (catalog space, or an editor-chosen one
            for sweepable statics; None for non-numeric statics).
        modalities: Env modalities where the knob has any effect.
        renderers: Camera renderers under which the knob acts.
        note: One-line clarification (catalog note for randomized knobs).
        source: The catalog :class:`Knob`, or None for statics.
    """

    name: str
    schedule: Schedule
    liveness: Liveness
    kind: Kind
    cfg_path: str
    space: Optional[Space]
    modalities: frozenset = frozenset({"camera", "feature"})
    renderers: frozenset = _ALL_RENDERERS
    note: str = ""
    source: Optional[Knob] = None


def _wrap(k: Knob) -> EditorKnob:
    """Annotate one catalog knob with its editor schedule/liveness/kind.

    Args:
        k: The catalog entry to wrap.

    Returns:
        The corresponding :class:`EditorKnob`.

    Raises:
        ValueError: If the knob's layer/cfg_key combination is unknown (a new
            catalog layer needs an explicit rule here).
    """
    common = dict(name=k.name, cfg_path=k.cfg_key, space=k.space,
                  modalities=k.modalities, renderers=k.renderers,
                  note=k.note, source=k)
    if k.layer == "image":
        kind = "aug_range" if k.name in _AUG_RANGE_KEYS else "aug_scalar"
        return EditorKnob(schedule="per_step", liveness="offline",
                          kind=kind, **common)
    if k.layer == "actuation":
        return EditorKnob(schedule="per_step", liveness="reroll",
                          kind="action", **common)
    if k.layer == "physics":
        return EditorKnob(schedule="per_run", liveness="rebuild",
                          kind="physics", **common)
    if k.layer == "geometry":     # track_width_scale: feature-only rulebook
        return EditorKnob(schedule="per_episode", liveness="rebuild",
                          kind="physics", **common)
    if k.layer == "visual":
        if k.name == "world_color":
            return EditorKnob(schedule="per_episode", liveness="offline",
                              kind="world_color", **common)
        if k.name == "pixel_noise":
            return EditorKnob(schedule="per_step", liveness="offline",
                              kind="pixel_noise", **common)
        if k.name.startswith("camera_"):
            return EditorKnob(schedule="per_run", liveness="reroll",
                              kind="mount", **common)
        if k.name.startswith("env_map"):
            return EditorKnob(schedule="per_run", liveness="rebuild",
                              kind="env_map", **common)
    raise ValueError(f"no editor rule for catalog knob {k.name!r} "
                     f"(layer={k.layer!r})")


def _static(name: str, cfg_path: str, space: Optional[Space] = None, *,
            renderers: frozenset = _ALL_RENDERERS, note: str = "") -> EditorKnob:
    """Build one static (never-randomized) scene knob entry.

    Args:
        name: Display name (the cfg leaf).
        cfg_path: Dotted env-cfg path.
        space: Sweep range for numeric statics, or None.
        renderers: Renderers the knob affects.
        note: One-line clarification.

    Returns:
        The static :class:`EditorKnob` (schedule ``build``, liveness
        ``rebuild``, camera-only).
    """
    return EditorKnob(name=name, schedule="build", liveness="rebuild",
                      kind="static", cfg_path=cfg_path, space=space,
                      modalities=frozenset({"camera"}), renderers=renderers,
                      note=note)


# static scene knobs — the cfg surface a scene editor edits that is not (and
# must not be) in the DR catalog. Defaults live in configs/cfgs.py.
SCENE_KNOBS: list[EditorKnob] = [
    _static("light_intensity", "vision.light_intensity", FloatRange(1.0, 12.0),
            renderers=frozenset({"madrona"}),
            note="the single directional light (Madrona adds the only add_light)"),
    _static("nyx_light_intensity", "vision.nyx_light_intensity", FloatRange(0.5, 8.0),
            renderers=frozenset({"nyx"}), note="Nyx sun intensity"),
    _static("nyx_spp", "vision.nyx_spp", IntRange(1, 16),
            renderers=frozenset({"nyx"}), note="Nyx samples per pixel"),
    _static("camera_fov", "vision.camera_fov", FloatRange(60.0, 120.0),
            note="onboard camera field of view (deg)"),
    _static("camera_pitch_deg", "vision.camera_pitch_deg", FloatRange(0.0, 25.0),
            note="base mount pitch; mount JITTER is the camera_pitch_jitter knob"),
    _static("background_color", "vision.background_color",
            note="scene background RGB (rasterizer/spectator ground colour)"),
    _static("field_color", "vision.field_color",
            note="green ground-plane RGB (doubles as the field under Madrona)"),
    _static("madrona_rg_swap", "vision.madrona_rg_swap",
            renderers=frozenset({"madrona"}),
            note="R<->G correction for the alpha-cutout centerline texture"),
    _static("track", "sim.track", note="track name(s); >1 name = tiled variants "
            "(width variants ride this: tools.track_builder.width_variants)"),
    _static("track_grid_spacing", "sim.track_grid_spacing", FloatRange(50.0, 200.0),
            note="metres between world tiles for multi-track camera scenes"),
]

REGISTRY: dict[str, EditorKnob] = {
    **{k.name: _wrap(k) for k in CATALOG},
    **{k.name: k for k in SCENE_KNOBS},
}


def get(name: str) -> EditorKnob:
    """Look one knob up by name.

    Args:
        name: Catalog knob name or static scene-knob name.

    Returns:
        The registry entry.

    Raises:
        KeyError: With the list of known names, if ``name`` is unknown.
    """
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown knob {name!r}; known: {sorted(REGISTRY)}") from None


def sweep_values(knob: EditorKnob, points: int) -> list:
    """Evenly spaced sweep values across a knob's suggested space.

    Args:
        knob: The knob being swept.
        points: Number of sample points requested.

    Returns:
        Floats across ``[lo, hi]`` for ranges, ints for integer ranges
        (deduplicated), or magnitudes across ``[0, m]`` for symmetric knobs.

    Raises:
        ValueError: If the knob has no numeric space to sweep.
    """
    s = knob.space
    if s is None:
        raise ValueError(f"knob {knob.name!r} has no numeric space to sweep")
    if isinstance(s, IntRange):
        step = max(1, round((s.hi - s.lo) / max(points - 1, 1)))
        return sorted(set(range(s.lo, s.hi + 1, step)) | {s.hi})
    lo, hi = (0.0, s.m) if hasattr(s, "m") else (s.lo, s.hi)
    if points == 1:
        return [hi]
    return [lo + (hi - lo) * i / (points - 1) for i in range(points)]


def dr_for_value(knob: EditorKnob, value) -> dict:
    """The offline DR-parameter dict realizing one sweep value of a knob.

    Range-shaped aug keys become the degenerate range ``(v, v)`` — fully
    deterministic; scalar-magnitude keys keep the magnitude and are sampled
    under the caller's seed.

    Args:
        knob: An OFFLINE-liveness knob (image aug, world colour, pixel noise).
        value: The sweep value.

    Returns:
        A dict for :func:`~.pipeline.replay_stages`'s ``dr`` argument.

    Raises:
        ValueError: If the knob is not offline-replayable (mount/env-map/
            physics/static values are realized live or at build instead).
    """
    if knob.kind == "aug_range":
        return {"image_aug": {knob.name: (float(value), float(value))}}
    if knob.kind == "aug_scalar":
        v = int(value) if isinstance(knob.space, IntRange) else float(value)
        return {"image_aug": {knob.name: v}}
    if knob.kind == "world_color":
        return {"world_color": float(value)}
    if knob.kind == "pixel_noise":
        return {"pixel_noise": float(value)}
    raise ValueError(
        f"knob {knob.name!r} ({knob.kind}) is not offline-replayable — "
        f"liveness is {knob.liveness!r}")


def default_layout(knob: EditorKnob) -> str:
    """Pick the layout that shows this knob's variation axis honestly.

    Args:
        knob: The knob being visualized.

    Returns:
        ``"filmstrip"`` (one env, k values — per-step image knobs),
        ``"contact_sheet"`` (N envs, one shared pose — per-env axes), or
        ``"ab"`` (two builds side by side — build/static knobs).
    """
    if knob.schedule == "per_step":
        return "filmstrip"
    if knob.schedule in ("per_episode", "per_run"):
        return "contact_sheet"
    return "ab"
