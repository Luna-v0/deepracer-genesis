"""Zoo manifests (config-as-code): loading, resolution, and expansion order.

The manifest file is the single declaration surface — these tests pin the
loader's resolution rules and that ``compile_zoo`` expands mixed sources in
declaration order without touching disk for already-registered tracks.
"""

import textwrap

import pytest

from deepracer_genesis.tools.zoo import (OfficialSample, TrackVariant, Zoo,
                                         compile_zoo, default_zoo,
                                         load_manifest)


def _write(tmp_path, body: str):
    """Write a manifest module to a temp file and return its path."""
    p = tmp_path / "zoos.py"
    p.write_text(textwrap.dedent(body))
    return str(p)


def test_load_manifest_named_attr(tmp_path):
    """path.py:name resolves that Zoo (callables are called)."""
    path = _write(tmp_path, """
        from deepracer_genesis.tools.zoo import TrackVariant, Zoo
        a = Zoo("a", (TrackVariant("reinvent_base"),))
        def b():
            return Zoo("b", (TrackVariant("Oval_track"),))
    """)
    assert load_manifest(f"{path}:a").name == "a"
    assert load_manifest(f"{path}:b").name == "b"


def test_load_manifest_single_zoo_autoresolves(tmp_path):
    """A module defining exactly one Zoo needs no :name."""
    path = _write(tmp_path, """
        from deepracer_genesis.tools.zoo import TrackVariant, Zoo
        only = Zoo("only", (TrackVariant("reinvent_base"),))
    """)
    assert load_manifest(path).name == "only"


def test_load_manifest_prefers_zoo_attr_and_rejects_ambiguity(tmp_path):
    """Two Zoos: 'zoo' wins by convention; otherwise the loader refuses."""
    path = _write(tmp_path, """
        from deepracer_genesis.tools.zoo import TrackVariant, Zoo
        zoo = Zoo("convention", (TrackVariant("reinvent_base"),))
        other = Zoo("other", (TrackVariant("Oval_track"),))
    """)
    assert load_manifest(path).name == "convention"
    path2 = _write(tmp_path / "..", """
        from deepracer_genesis.tools.zoo import TrackVariant, Zoo
        a = Zoo("a", (TrackVariant("reinvent_base"),))
        b = Zoo("b", (TrackVariant("Oval_track"),))
    """)
    with pytest.raises(SystemExit, match="a.*b"):
        load_manifest(path2)


def test_compile_expands_sources_in_declaration_order():
    """Mixed manifests expand in order; registered tracks bake nothing."""
    zoo = Zoo("mixed", (
        OfficialSample(names=("Oval_track", "reinvent_base"), fetch=False,
                       jitter=0.0, looks=False),
        TrackVariant("reinvent_base"),
    ))
    assert compile_zoo(zoo) == ("Oval_track", "reinvent_base",
                                "reinvent_base")


def test_default_zoo_is_the_declared_official_population():
    """The bare-CLI default equals a one-source official manifest."""
    zoo = default_zoo()
    assert len(zoo.variants) == 1
    src = zoo.variants[0]
    assert isinstance(src, OfficialSample)
    assert (src.n, src.jitter > 0, src.looks) == (32, True, True)
