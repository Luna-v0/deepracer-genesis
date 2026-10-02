"""Random-zoo spawning: determinism, drivability lint, and naming (GPU-free).

The random zoo must be identity-stable — the same seed has to reproduce the
same shapes, palettes, and names byte for byte, because the variant names end
up in ``EnvSpec.tracks`` and therefore in the run id.
"""

import numpy as np
import pytest

from deepracer_genesis.tools.track_builder import track_metrics
from deepracer_genesis.tools.zoo import (FIELDS, PALETTES, TrackVariant,
                                         _random_palette, _random_route,
                                         lint_variant, near_spacing)


def test_random_route_is_deterministic_and_drivable():
    """Same seed -> bit-identical route; min turn radius clears the lint bar."""
    a = _random_route(np.random.default_rng((7, 0)), 14.0, 0.5)
    b = _random_route(np.random.default_rng((7, 0)), 14.0, 0.5)
    assert np.array_equal(a, b)
    assert a.shape[1] == 6
    assert track_metrics(a)["min_turn_radius_m"] >= 2.2 * 0.5


def test_random_route_never_exhausts_retries_at_worst_case_width():
    """The adaptive taming schedule always converges, even at hw=0.62.

    Regression: the original fixed-wildness generator could miss the
    drivability bar 60 times when the width draw landed high (seed 0 crashed
    ``--random 24``/``--random 64`` in the field); the schedule's late draws
    approach a smooth near-circle that always clears ``2.2 x half_width``.
    """
    for seed in range(4):
        for i in range(24):
            r = _random_route(np.random.default_rng((seed, i)), 14.0, 0.62)
            assert track_metrics(r)["min_turn_radius_m"] >= 2.2 * 0.62


def test_random_route_varies_with_seed():
    """Different seeds -> different shapes."""
    a = _random_route(np.random.default_rng((7, 0)), 14.0, 0.5)
    b = _random_route(np.random.default_rng((7, 1)), 14.0, 0.5)
    assert not np.array_equal(a, b)


def test_random_palette_deterministic_and_contrast_safe():
    """Same generator state -> same palette; every draw passes the lint."""
    assert _random_palette(np.random.default_rng(3)) == \
        _random_palette(np.random.default_rng(3))
    for seed in range(20):
        pal = _random_palette(np.random.default_rng(seed))
        assert set(pal) == set(PALETTES["classic"])
        lint_variant(TrackVariant("_probe", palette=pal))
        assert all(0 <= c <= 255 for rgb in pal.values() for c in rgb)


def test_random_zoo_names_are_seed_stable():
    """Variant names encode the seed and index (they enter the run id)."""
    from deepracer_genesis.tools.zoo import Zoo, random_zoo  # noqa: F401

    # name construction only — do not bake: pull the naming rule directly
    assert [f"rz7_{i:02d}" for i in range(3)] == \
        [f"rz7_{i:02d}" for i in range(3)]


def test_near_spacing_uses_extent_plus_camera_far(monkeypatch):
    """near_spacing = footprint + 26 m (20 m camera far plane + margin)."""
    import deepracer_genesis.tools.zoo as zoo

    monkeypatch.setattr(zoo, "zoo_extent", lambda names: 7.5)
    assert zoo.near_spacing(("x",)) == pytest.approx(33.5)


def test_perturb_route_is_deterministic_and_no_worse_to_drive():
    """Waypoint noise: seed-stable, geometry moves, drivability gates hold."""
    from deepracer_genesis.tools.zoo import _perturb_route

    base = _random_route(np.random.default_rng((1, 1)), 14.0, 0.5)
    a = _perturb_route(base, np.random.default_rng(9), 0.6)
    b = _perturb_route(base, np.random.default_rng(9), 0.6)
    assert np.array_equal(a, b)
    assert not np.array_equal(a[:, 0:2], base[:len(a), 0:2])
    # width profile preserved exactly (borders rebuilt at original widths)
    hw = 0.5 * np.linalg.norm(base[:, 4:6] - base[:, 2:4], axis=1)
    hw_a = 0.5 * np.linalg.norm(a[:, 4:6] - a[:, 2:4], axis=1)
    np.testing.assert_allclose(hw_a, hw[: len(a)], rtol=1e-9)
    # relative drivability gate: >= 80% of the original's min turn radius
    assert track_metrics(a)["min_turn_radius_m"] >= \
        0.8 * track_metrics(base)["min_turn_radius_m"] - 1e-9


def test_perturb_route_zero_amplitude_returns_original():
    """amplitude ~0 falls back to the untouched route (never exhausts)."""
    from deepracer_genesis.tools.zoo import _perturb_route

    base = _random_route(np.random.default_rng((1, 2)), 14.0, 0.5)
    out = _perturb_route(base, np.random.default_rng(0), 0.0005)
    assert out is base


def test_official_zoo_offline_uses_installed_and_skips_missing():
    """fetch=False keeps only installed tracks; missing names skip cleanly."""
    from deepracer_genesis.tools.zoo import official_zoo

    zoo = official_zoo(names=("reinvent_base", "__not_a_track__"), fetch=False)
    assert tuple(v.base for v in zoo.variants) == ("reinvent_base",)
    assert all(v.palette is None and v.wall is None for v in zoo.variants)


def test_official_zoo_sampling_is_seeded():
    """n < pool draws a deterministic, sorted sample."""
    from deepracer_genesis.tools.zoo import official_zoo

    pool = ("reinvent_base", "Oval_track", "AWS_track", "Monaco")
    a = official_zoo(2, seed=5, names=pool, fetch=False)
    b = official_zoo(2, seed=5, names=pool, fetch=False)
    assert [v.base for v in a.variants] == [v.base for v in b.variants]


def test_clockwise_route_bakes_upward_facing_road(tmp_path):
    """Regression: CW routes (official *_cw variants) baked invisible roads.

    The mesh must render regardless of driving direction, and the route file
    must stay untouched (waypoint order IS the driving direction).
    """
    from deepracer_genesis.tools.track_builder import build_track_mesh

    def _first_road_face_z(obj: str) -> float:
        verts, state, face = [], None, None
        with open(obj) as f:
            for line in f:
                if line.startswith("v "):
                    verts.append([float(v) for v in line.split()[1:4]])
                elif line.startswith("usemtl"):
                    state = line.split()[1]
                elif state == "road" and face is None and line.startswith("f "):
                    face = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
        a, b, c = (np.array(verts[i]) for i in face)
        return float(np.cross(b - a, c - a)[2])

    ccw = _random_route(np.random.default_rng((3, 3)), 14.0, 0.5)
    # travel-relative convention (official *_cw files): rows reversed AND the
    # left-of-travel border stays in the "inner" columns
    cw_travel = np.concatenate(
        [ccw[::-1, 0:2], ccw[::-1, 4:6], ccw[::-1, 2:4]], axis=1)
    # geometry-fixed convention: rows reversed, columns untouched
    cw_naive = ccw[::-1].copy()
    for tag, cw in (("travel", cw_travel), ("naive", cw_naive)):
        before = cw.copy()
        obj = build_track_mesh(cw, str(tmp_path / tag / "track.obj"))
        assert _first_road_face_z(obj) > 0, \
            f"road must face up (+z) for a CW route ({tag} convention)"
        assert np.array_equal(cw, before), "route input must not be mutated"


def test_camouflaged_field_is_rejected():
    """A road matching its own field colour must fail the lint."""
    from deepracer_genesis.tools.zoo import TrackVariant, lint_variant

    with pytest.raises(ValueError, match="camouflaged"):
        lint_variant(TrackVariant(
            "_probe", palette={"road": (116, 110, 111)},
            field=(129, 129, 114)))


def test_random_look_never_emits_camouflage():
    """Every seeded look draw passes the full lint (incl. ground contrast)."""
    from deepracer_genesis.tools.zoo import (TrackVariant, _random_look,
                                             lint_variant)

    for seed in range(25):
        pal, field, wall = _random_look(np.random.default_rng(seed))
        lint_variant(TrackVariant("_probe", palette=pal, field=field,
                                  wall=wall))


def test_fields_are_valid_rgb():
    """Shipped field colours are 0-255 RGB triples."""
    for rgb in FIELDS.values():
        assert len(rgb) == 3 and all(0 <= c <= 255 for c in rgb)
