"""dr_editor sheet tests: frame conventions and PNG canvas geometry.

Pure PIL/torch — no sim. Pins ``to_image``'s three accepted layouts (with
the newest-3-channels rule for stacks), its rejections, and the exact
canvas sizes ``sheet``/``paired_sheet`` compose (label bars ``_LABEL_H``
plus ``_PAD`` gutters, read from the module rather than hardcoded).
"""

import os

import numpy as np
import pytest
import torch
from PIL import Image

from deepracer_genesis.tools.dr_editor import sheets
from deepracer_genesis.tools.dr_editor.sheets import (
    paired_sheet,
    sheet,
    to_image,
)

_H, _W = 4, 6


# ---------------------------------------------------------------- to_image

def test_to_image_chw_float():
    """A (3, H, W) float frame in [0, 1] maps to exact RGB bytes."""
    frame = torch.zeros(3, _H, _W)
    frame[0] = 1.0                                  # solid red
    img = to_image(frame, scale=1)
    assert img.size == (_W, _H)
    assert img.mode == "RGB"
    assert img.getpixel((0, 0)) == (255, 0, 0)


def test_to_image_stack_shows_newest_three_channels():
    """A (C, H, W) stack with C % 3 == 0 renders the NEWEST 3 channels."""
    stack = torch.ones(12, _H, _W)                  # older groups: solid white
    stack[9:] = 0.0
    stack[10] = 1.0                                 # newest group: solid green
    img = to_image(stack, scale=1)
    assert img.size == (_W, _H)
    assert img.getpixel((0, 0)) == (0, 255, 0)
    assert img.getpixel((_W - 1, _H - 1)) == (0, 255, 0)


def test_to_image_hwc_uint8():
    """An (H, W, 3) uint8 frame passes through byte-exact."""
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, size=(_H, _W, 3), dtype=np.uint8)
    img = to_image(arr, scale=1)
    assert img.size == (_W, _H)
    np.testing.assert_array_equal(np.asarray(img), arr)


def test_to_image_scale_upsizes_nearest():
    """scale=k nearest-neighbour upsizes: k x k blocks replicate one pixel."""
    frame = torch.zeros(3, _H, _W)
    frame[2, 0, 0] = 1.0                            # one blue pixel at (0, 0)
    img = to_image(frame, scale=3)
    assert img.size == (_W * 3, _H * 3)
    for x in range(3):
        for y in range(3):
            assert img.getpixel((x, y)) == (0, 0, 255)
    assert img.getpixel((3, 0)) == (0, 0, 0)


def test_to_image_rejects_two_dim():
    """A 2-dim (H, W) array is not a frame."""
    with pytest.raises(ValueError, match="3 dims"):
        to_image(torch.zeros(_H, _W))


def test_to_image_rejects_bad_channel_count():
    """(5, H, W) is neither RGB nor a channel stack."""
    with pytest.raises(ValueError, match="unrecognized frame layout"):
        to_image(torch.zeros(5, _H, _W))


# ------------------------------------------------------------------- sheet

def _tiles(n):
    """n labeled random (3, H, W) frames."""
    g = torch.Generator().manual_seed(0)
    return [(f"t{i}", torch.rand(3, _H, _W, generator=g)) for i in range(n)]


def test_sheet_filmstrip_canvas_size(tmp_path):
    """Default cols: 3 tiles land in one labeled row (a filmstrip)."""
    scale = 2
    out = str(tmp_path / "strip.png")
    assert sheet(_tiles(3), out, scale=scale) == out
    tile_w, tile_h = _W * scale, _H * scale + sheets._LABEL_H
    with Image.open(out) as img:
        assert img.size == (3 * tile_w + 2 * sheets._PAD, tile_h)


def test_sheet_wraps_at_cols(tmp_path):
    """cols=2 wraps 3 tiles into a 2-tile row plus a 1-tile row."""
    scale = 2
    out = str(tmp_path / "grid.png")
    sheet(_tiles(3), out, cols=2, scale=scale)
    tile_w, tile_h = _W * scale, _H * scale + sheets._LABEL_H
    with Image.open(out) as img:
        assert img.size == (2 * tile_w + sheets._PAD,
                            2 * tile_h + sheets._PAD)


def test_sheet_title_adds_header_bar(tmp_path):
    """A title adds a _LABEL_H + 2 header above the tiles."""
    scale = 1
    out = str(tmp_path / "titled.png")
    sheet(_tiles(2), out, scale=scale, title="hue sweep")
    tile_w, tile_h = _W * scale, _H * scale + sheets._LABEL_H
    with Image.open(out) as img:
        assert img.size == (2 * tile_w + sheets._PAD,
                            tile_h + sheets._LABEL_H + 2)


def test_paired_sheet_writes_grid(tmp_path):
    """Two 2-tile rows compose a 2x2 grid; the path is returned."""
    scale = 1
    row = _tiles(2)
    out = str(tmp_path / "pairs" / "pair.png")
    got = paired_sheet([("onboard", row), ("topdown", row)], out, scale=scale)
    assert got == out
    assert os.path.isfile(out)
    tile_w, tile_h = _W * scale, _H * scale + sheets._LABEL_H
    with Image.open(out) as img:
        assert img.size == (2 * tile_w + sheets._PAD,
                            2 * tile_h + sheets._PAD)
