"""RolloutDataset caching: per-set dirs (P16), raw resolution caches, stacks.

Synthetic two-frame-wide parquet shards; nothing here touches real data.
"""

import io
import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from deepracer_genesis.perception import dataset as ds

HW = (24, 32)          # collected (h, w) — small so the suite stays fast
FRAMES = 6
N_STATE = 10
SLICE = (2, 9)         # 7 targets


def _png(rng):
    img = Image.fromarray(rng.integers(0, 255, (*HW, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _make_track(root, name, seed):
    rng = np.random.default_rng(seed)
    d = root / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"cnn_target_slice": list(SLICE)}))
    states = [rng.uniform(-0.5, 0.5, N_STATE).astype(np.float32)
              for _ in range(FRAMES)]
    pd.DataFrame({
        "image": [_png(rng) for _ in range(FRAMES)],
        "state": states,
        "env": np.zeros(FRAMES, np.int32),
        "episode": np.zeros(FRAMES, np.int32),
        "t": np.arange(FRAMES, dtype=np.int64),
    }).to_parquet(d / "rollout_000.parquet")


@pytest.fixture()
def data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(ds, "CACHE", tmp_path / "cache")
    _make_track(tmp_path, "alpha", seed=1)
    _make_track(tmp_path, "beta", seed=2)
    return tmp_path


def test_stacks_have_the_declared_shape_and_count(data_root):
    d = ds.RolloutDataset(tracks=("alpha",), k=2)
    assert len(d) == FRAMES - 1          # k-1 fewer stacks than frames
    frames, target = d[0]
    assert frames.shape == (6, *HW) and target.shape == (SLICE[1] - SLICE[0],)
    assert frames.dtype == torch.float32
    assert 0.0 <= frames.min() and frames.max() <= 1.0


def test_two_subsets_coexist_without_clobbering_each_other(data_root):
    """P16: a second dataset over a different subset must not rebuild the
    first's blob — that corrupted the first dataset's lazily-read frames."""
    train = ds.RolloutDataset(tracks=("alpha",), k=2)
    before, _ = train[0]
    val = ds.RolloutDataset(tracks=("beta",), k=2)
    train._blob = None                    # force a fresh memmap open
    after, _ = train[0]
    assert torch.equal(before, after), "beta's cache build corrupted alpha's"
    assert ds._cache_dir(("alpha",)) != ds._cache_dir(("beta",))
    assert val._dir != train._dir


def test_cache_is_not_rebuilt_when_sources_are_unchanged(data_root):
    ds.RolloutDataset(tracks=("alpha",), k=2)
    blob = ds._cache_dir(("alpha",)) / "images.bin"
    stamp = blob.stat().st_mtime_ns
    ds.RolloutDataset(tracks=("alpha",), k=2)
    assert blob.stat().st_mtime_ns == stamp


def test_resolution_serves_lanczos_resized_frames_without_png(data_root):
    half = (HW[0] // 2, HW[1] // 2)
    d = ds.RolloutDataset(tracks=("alpha",), k=2, resolution=half)
    frames, _ = d[0]
    assert frames.shape == (6, *half)
    # the raw cache must hold exactly the LANCZOS downscale of the original
    native = ds.RolloutDataset(tracks=("alpha",), k=2)
    first = (native[0][0][:3].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    expected = np.asarray(
        Image.fromarray(first).resize((half[1], half[0]), Image.LANCZOS),
        dtype=np.float32) / 255.0
    assert np.allclose(frames[:3].permute(1, 2, 0).numpy(), expected)


def test_raw_cache_is_materialized_once(data_root):
    half = (HW[0] // 2, HW[1] // 2)
    ds.RolloutDataset(tracks=("alpha",), k=2, resolution=half)
    raw = ds._cache_dir(("alpha",)) / f"raw_{half[0]}x{half[1]}.bin"
    stamp = raw.stat().st_mtime_ns
    ds.RolloutDataset(tracks=("alpha",), k=2, resolution=half)
    assert raw.stat().st_mtime_ns == stamp


def test_upscaling_is_refused(data_root):
    with pytest.raises(ValueError, match="UPSCALE"):
        ds.RolloutDataset(tracks=("alpha",), k=2,
                          resolution=(HW[0] * 2, HW[1] * 2))


def test_jitter_composes_with_resolution(data_root):
    half = (HW[0] // 2, HW[1] // 2)
    plain = ds.RolloutDataset(tracks=("alpha",), k=2, resolution=half)
    jittered = ds.RolloutDataset(tracks=("alpha",), k=2, resolution=half,
                                 jitter=True, seed=3)
    a, _ = plain[0]
    b, _ = jittered[0]
    assert a.shape == b.shape
    assert not torch.equal(a, b)
