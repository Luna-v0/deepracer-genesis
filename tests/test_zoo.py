"""Track zoo compiler: lint, deterministic naming, cached baking, baked bytes.

Pure math plus tmp-dir filesystem checks — no sim, no genesis. Only the
GPU-free surface of ``tools.zoo`` is exercised here (``view_zoo`` and
``watch_zoo`` import genesis lazily and are never called).
"""

import os
import re

import numpy as np
import pytest

from deepracer_genesis.envs import track as track_mod
from deepracer_genesis.envs.track import TRACKS
from deepracer_genesis.tools import track_builder
from deepracer_genesis.tools import zoo as zoo_mod
from deepracer_genesis.tools.zoo import (FIELDS, PALETTES, TrackVariant, Zoo,
                                         compile_zoo, demo_zoo, lint_variant,
                                         variant_name)


def _ellipse_route(n: int = 64) -> np.ndarray:
    """Closed ellipse route with per-waypoint width variation.

    Args:
        n: Number of waypoints (no repeated closing point).

    Returns:
        (n, 6) float64 route of [center, inner, outer] whose half_width
        (0.5 * |outer - inner|) varies per waypoint as 0.4 + 0.15*sin(3t).
    """
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    center = np.stack([3.0 * np.cos(t), 1.5 * np.sin(t)], axis=1)
    nrm = np.stack([1.5 * np.cos(t), 3.0 * np.sin(t)], axis=1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    hw = 0.4 + 0.15 * np.sin(3.0 * t)
    inner = center + nrm * hw[:, None]
    outer = center - nrm * hw[:, None]
    return np.concatenate([center, inner, outer], axis=1)


def _half_width(route: np.ndarray) -> np.ndarray:
    """Per-waypoint half width, exactly as track.py derives it (0.5*|outer-inner|)."""
    return 0.5 * np.linalg.norm(route[:, 4:6] - route[:, 2:4], axis=1)


def _variant_sig(obj_path: str) -> str:
    """The '# variant <sig>' comment line baked into a track OBJ.

    Args:
        obj_path: Path of the baked track.obj.

    Returns:
        The full signature line (stripped).

    Raises:
        AssertionError: If the OBJ carries no variant signature line.
    """
    with open(obj_path) as f:
        for line in f:
            if line.startswith("# variant "):
                return line.strip()
    raise AssertionError(f"no '# variant' signature line in {obj_path}")


def _vertex_lines(obj_path: str) -> list:
    """All 'v x y z' vertex lines of an OBJ, newline-stripped."""
    with open(obj_path) as f:
        return [ln for ln in f.read().splitlines() if ln.startswith("v ")]


# ------------------------------------------------------------------ fixtures
@pytest.fixture
def registry():
    """Snapshot the shared TRACKS registry and restore it after the test.

    Yields:
        The live TRACKS dict (tests may add entries freely).
    """
    snapshot = dict(TRACKS)
    yield TRACKS
    TRACKS.clear()
    TRACKS.update(snapshot)


@pytest.fixture
def sandbox(tmp_path, monkeypatch, registry):
    """Point every asset root at a tmp dir and register a fake source track.

    Monkeypatches ``track_builder.ASSETS_DIR``/``GENERATED_DIR`` (bake side),
    ``envs.track.ASSETS_DIR`` (load side), and — because ``tools.zoo`` imports
    both names into its own namespace — ``zoo.ASSETS_DIR``/``GENERATED_DIR``
    (compile side) to the same tmp root, then saves a synthetic variable-width
    route as registered track ``"src"``.

    Yields:
        (assets_root, generated_root, source_route) as (str, str, ndarray).
    """
    assets = tmp_path / "assets"
    generated = assets / "tracks" / "generated"
    (assets / "routes").mkdir(parents=True)
    monkeypatch.setattr(track_builder, "ASSETS_DIR", str(assets))
    monkeypatch.setattr(track_builder, "GENERATED_DIR", str(generated))
    monkeypatch.setattr(track_mod, "ASSETS_DIR", str(assets))
    monkeypatch.setattr(zoo_mod, "ASSETS_DIR", str(assets))
    monkeypatch.setattr(zoo_mod, "GENERATED_DIR", str(generated))
    route = _ellipse_route()
    np.save(assets / "routes" / "src.npy", route)
    registry["src"] = ("routes/src.obj", "routes/src.npy", None)
    yield str(assets), str(generated), route


# --------------------------------------------------------------- lint_variant
@pytest.mark.parametrize("width", [0.0, -0.5])
def test_lint_rejects_nonpositive_width(width):
    with pytest.raises(ValueError, match="> 0"):
        lint_variant(TrackVariant("src", width=width))


def test_lint_unknown_palette_name_lists_known():
    """The error for a bad palette name teaches the valid names."""
    with pytest.raises(ValueError) as err:
        lint_variant(TrackVariant("src", palette="neon"))
    msg = str(err.value)
    assert "unknown palette" in msg and "neon" in msg
    for known in PALETTES:
        assert known in msg


def test_lint_unknown_field_name_lists_known():
    with pytest.raises(ValueError) as err:
        lint_variant(TrackVariant("src", field="lava"))
    msg = str(err.value)
    assert "unknown field" in msg and "lava" in msg
    for known in FIELDS:
        assert known in msg


def test_lint_rejects_bad_palette_key():
    """A palette dict may only recolour road/border/centerline."""
    with pytest.raises(ValueError) as err:
        lint_variant(TrackVariant("src", palette={"sky": (120, 160, 220)}))
    msg = str(err.value)
    assert "sky" in msg and "allowed" in msg


@pytest.mark.parametrize("line", ["centerline", "border"])
def test_lint_rejects_low_contrast_lines(line):
    """A line colour near the road's luminance erases the driving signal."""
    with pytest.raises(ValueError, match="visible") as err:
        lint_variant(TrackVariant("src", palette={line: (60, 60, 60)}))
    assert line in str(err.value)


@pytest.mark.parametrize("name", sorted(PALETTES))
def test_shipped_palettes_lint_clean(name):
    """Self-consistency: every named palette passes its own contrast lint."""
    lint_variant(TrackVariant("src", palette=name))


@pytest.mark.parametrize("name", sorted(FIELDS))
def test_shipped_fields_lint_clean(name):
    lint_variant(TrackVariant("src", field=name))


# --------------------------------------------------------------- variant_name
def test_all_default_name_is_base():
    assert variant_name(TrackVariant("src")) == "src"


def test_width_only_name():
    assert variant_name(TrackVariant("src", width=0.9)) == "src_w090"


def test_named_palette_name():
    assert variant_name(TrackVariant("src", palette="dusk")) == "src_dusk"


def test_dict_palette_name_is_stable_hash():
    """An explicit palette dict hashes to a p<6hex> tag, stable across calls."""
    pal = {"road": (10, 12, 14), "centerline": (250, 245, 240)}
    name = variant_name(TrackVariant("src", palette=pal))
    assert re.fullmatch(r"src_p[0-9a-f]{6}", name)
    assert variant_name(TrackVariant("src", palette=dict(pal))) == name
    other = variant_name(TrackVariant(
        "src", palette={"road": (11, 12, 14), "centerline": (250, 245, 240)}))
    assert other != name


def test_field_name_tag():
    assert variant_name(TrackVariant("src", field="sand")) == "src_fsand"


def test_combined_name_ordering():
    """Axes always compose as base_wNNN_<palette>_f<field>."""
    v = TrackVariant("src", width=1.15, palette="faded", field="concrete")
    assert variant_name(v) == "src_w115_faded_fconcrete"


# ---------------------------------------------------------------- compile_zoo
def _zoo3() -> Zoo:
    """The canonical 3-variant test zoo: default + width + palette."""
    return Zoo("test3", (TrackVariant("src"),
                         TrackVariant("src", width=0.9),
                         TrackVariant("src", palette="dusk")))


def test_compile_names_assets_and_registration(sandbox):
    """Names come back in manifest order; baked variants get assets + entries."""
    _assets, generated, _route = sandbox
    names = compile_zoo(_zoo3())
    assert names == ("src", "src_w090", "src_dusk")
    for name in ("src_w090", "src_dusk"):
        track_dir = os.path.join(generated, name)
        assert os.path.exists(os.path.join(track_dir, "route.npy"))
        assert os.path.exists(os.path.join(track_dir, "track.obj"))
        rel = f"tracks/generated/{name}"
        assert TRACKS[name] == (f"{rel}/track.obj", f"{rel}/route.npy", None)


def test_all_default_variant_bakes_nothing(sandbox):
    """An all-default variant reuses the base track — no duplicate assets."""
    _assets, generated, _route = sandbox
    names = compile_zoo(Zoo("solo", (TrackVariant("src"),)))
    assert names == ("src",)
    assert not os.path.exists(os.path.join(generated, "src"))
    assert TRACKS["src"] == ("routes/src.obj", "routes/src.npy", None)


def test_existing_variants_are_not_rebaked_unless_forced(sandbox):
    """A repeat compile reuses baked assets; force=True rewrites them."""
    _assets, generated, _route = sandbox
    compile_zoo(_zoo3())
    paths = [os.path.join(generated, name, part)
             for name in ("src_w090", "src_dusk")
             for part in ("route.npy", "track.obj")]
    stale = os.path.getmtime(paths[0]) - 1000.0
    for p in paths:
        os.utime(p, (stale, stale))

    assert compile_zoo(_zoo3()) == ("src", "src_w090", "src_dusk")
    for p in paths:
        assert os.path.getmtime(p) == stale

    assert compile_zoo(_zoo3(), force=True) == ("src", "src_w090", "src_dusk")
    for p in paths:
        assert os.path.getmtime(p) > stale


def test_cached_assets_reregister_without_rebake(sandbox):
    """Assets baked by an earlier process re-enter TRACKS without a rebake."""
    _assets, generated, _route = sandbox
    compile_zoo(_zoo3())
    route_path = os.path.join(generated, "src_dusk", "route.npy")
    stale = os.path.getmtime(route_path) - 1000.0
    os.utime(route_path, (stale, stale))
    del TRACKS["src_dusk"]

    assert compile_zoo(_zoo3()) == ("src", "src_w090", "src_dusk")
    assert os.path.getmtime(route_path) == stale
    rel = "tracks/generated/src_dusk"
    assert TRACKS["src_dusk"] == (f"{rel}/track.obj", f"{rel}/route.npy", None)


# --------------------------------------------------------------- baked content
def test_palette_variant_mtl_carries_overridden_colours(sandbox):
    """The variant's .mtl holds the palette's Kd values, not the defaults."""
    _assets, generated, _route = sandbox
    compile_zoo(Zoo("pal", (TrackVariant("src", palette="dusk"),)))
    with open(os.path.join(generated, "src_dusk", "track.mtl")) as f:
        mtl = f.read()
    for mat, rgb in PALETTES["dusk"].items():
        kd = " ".join(f"{c / 255:.4f}" for c in rgb)
        assert f"newmtl {mat}\nKd {kd}\n" in mtl


def test_variant_signature_lines_differ_between_palettes(sandbox):
    """Each palette bakes its own '# variant <sig>' line into the OBJ."""
    _assets, generated, _route = sandbox
    compile_zoo(Zoo("pals", (TrackVariant("src", palette="dusk"),
                             TrackVariant("src", palette="asphalt_light"))))
    sig_dusk = _variant_sig(os.path.join(generated, "src_dusk", "track.obj"))
    sig_light = _variant_sig(
        os.path.join(generated, "src_asphalt_light", "track.obj"))
    assert "road:30.30.38" in sig_dusk
    assert sig_dusk != sig_light


def test_field_variant_bakes_ground_quad(sandbox):
    """A field variant's OBJ gains a 'usemtl field' quad: 4 verts at z 0.0005."""
    _assets, generated, _route = sandbox
    compile_zoo(Zoo("fld", (TrackVariant("src", field="sand"),
                            TrackVariant("src", palette="dusk"))))
    field_obj = os.path.join(generated, "src_fsand", "track.obj")
    with open(field_obj) as f:
        assert "usemtl field\n" in f.read()
    field_verts = [ln for ln in _vertex_lines(field_obj)
                   if ln.endswith(" 0.0005")]
    assert len(field_verts) == 4
    # exactly the quad on top of the identical road geometry (the dusk bake
    # shares the same route, differing only in colours)
    plain_obj = os.path.join(generated, "src_dusk", "track.obj")
    assert len(_vertex_lines(field_obj)) == len(_vertex_lines(plain_obj)) + 4
    with open(os.path.join(generated, "src_fsand", "track.mtl")) as f:
        mtl = f.read()
    kd = " ".join(f"{c / 255:.4f}" for c in FIELDS["sand"])
    assert f"newmtl field\nKd {kd}\n" in mtl


def test_same_geometry_palette_variants_are_byte_unique(sandbox):
    """The mesh-cache dedup guard: identical geometry, different OBJ bytes."""
    _assets, generated, _route = sandbox
    compile_zoo(Zoo("pals", (TrackVariant("src", palette="dusk"),
                             TrackVariant("src", palette="asphalt_light"))))
    with open(os.path.join(generated, "src_dusk", "track.obj"), "rb") as f:
        dusk = f.read()
    with open(os.path.join(generated, "src_asphalt_light", "track.obj"),
              "rb") as f:
        light = f.read()
    assert dusk != light


# ------------------------------------------------------- rulebook follows mesh
def test_look_only_variants_do_not_move_geometry(sandbox):
    """Palette/field variants keep the base route bit-for-bit (colours must
    never move geometry — spawns, localization, and the rulebook depend on it).
    """
    _assets, generated, route = sandbox
    compile_zoo(Zoo("look", (TrackVariant("src", palette="dusk"),
                             TrackVariant("src", field="sand"))))
    for name in ("src_dusk", "src_fsand"):
        baked = np.load(os.path.join(generated, name, "route.npy"))
        assert np.array_equal(baked, route)


def test_width_variant_scales_rulebook_width(sandbox):
    """A width variant's baked half_width is scale * the source's, per waypoint."""
    _assets, generated, route = sandbox
    compile_zoo(Zoo("wide", (TrackVariant("src", width=1.15),)))
    baked = np.load(os.path.join(generated, "src_w115", "route.npy"))
    assert np.allclose(_half_width(baked), 1.15 * _half_width(route),
                       rtol=1e-12)
    assert np.array_equal(baked[:, 0:2], route[:, 0:2])  # center bit-identical


# ------------------------------------------------------------------- demo_zoo
def test_demo_zoo_is_six_lintable_unique_variants():
    """The shipped demo: 6 variants, all lint-clean, all uniquely named."""
    zoo = demo_zoo()
    assert len(zoo.variants) == 6
    for variant in zoo.variants:
        lint_variant(variant)
    names = [variant_name(v) for v in zoo.variants]
    assert len(set(names)) == 6
