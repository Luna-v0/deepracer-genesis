"""Contact sheets, filmstrips, and stage strips: tensors -> labeled PNGs.

Pure PIL/torch composition — no sim, no display. Frames arrive as
``(3, H, W)``/``(N, 3, H, W)`` float in [0, 1] (the pipeline convention) or
``(H, W, 3)`` uint8 (the debug-camera convention); everything is normalized,
integer-upscaled for legibility (160x120 policy frames are tiny), labeled,
and tiled.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import torch
from PIL import Image, ImageDraw, ImageFont

_LABEL_H = 14
_PAD = 2


def to_image(frame, *, scale: int = 3) -> Image.Image:
    """Normalize one frame of any editor convention into an upscaled PIL image.

    Args:
        frame: ``(3, H, W)`` float in [0, 1], ``(C, H, W)`` with C a multiple
            of 3 (a channel stack — the NEWEST 3 channels are shown),
            ``(H, W, 3)`` uint8, or a numpy equivalent.
        scale: Integer nearest-neighbour upscale factor.

    Returns:
        The RGB PIL image.

    Raises:
        ValueError: If the frame layout is not recognized.
    """
    t = torch.as_tensor(frame)
    if t.dim() != 3:
        raise ValueError(f"expected one frame (3 dims); got shape {tuple(t.shape)}")
    if t.shape[-1] == 3 and t.dtype == torch.uint8:          # (H, W, 3) uint8
        arr = t.cpu().numpy()
    elif t.shape[0] % 3 == 0:                                # (C, H, W) float
        rgb = t[-3:] if t.shape[0] > 3 else t                # newest stack frame
        arr = (rgb.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    else:
        raise ValueError(f"unrecognized frame layout {tuple(t.shape)}")
    img = Image.fromarray(arr)
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    return img


def _labeled(img: Image.Image, label: str) -> Image.Image:
    """Put a one-line label bar above an image.

    Args:
        img: The tile image.
        label: Label text (empty = no bar).

    Returns:
        The tile with its label bar.
    """
    if not label:
        return img
    out = Image.new("RGB", (img.width, img.height + _LABEL_H), (24, 24, 24))
    out.paste(img, (0, _LABEL_H))
    ImageDraw.Draw(out).text((3, 1), label, fill=(230, 230, 230),
                             font=ImageFont.load_default())
    return out


def sheet(tiles: Sequence[tuple[str, object]], out_path: str, *,
          cols: Optional[int] = None, scale: int = 3,
          title: str = "") -> str:
    """Tile labeled frames into one PNG (filmstrip = one row; grid = cols).

    Args:
        tiles: ``(label, frame)`` pairs in display order.
        out_path: Destination PNG path (parents are created).
        cols: Tiles per row (default: all in one row — a filmstrip).
        scale: Integer upscale per tile.
        title: Optional sheet-level title bar.

    Returns:
        The written path.
    """
    imgs = [_labeled(to_image(f, scale=scale), lab) for lab, f in tiles]
    cols = cols or len(imgs)
    rows = [imgs[i:i + cols] for i in range(0, len(imgs), cols)]
    w = max(sum(t.width for t in r) + _PAD * (len(r) - 1) for r in rows)
    h = sum(max(t.height for t in r) for r in rows) + _PAD * (len(rows) - 1)
    top = _LABEL_H + 2 if title else 0
    canvas = Image.new("RGB", (w, h + top), (12, 12, 12))
    if title:
        ImageDraw.Draw(canvas).text((3, 1), title, fill=(255, 255, 255),
                                    font=ImageFont.load_default())
    y = top
    for r in rows:
        x = 0
        for t in r:
            canvas.paste(t, (x, y))
            x += t.width + _PAD
        y += max(t.height for t in r) + _PAD
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path)
    return out_path


def paired_sheet(rows: Sequence[tuple[str, Sequence[tuple[str, object]]]],
                 out_path: str, *, scale: int = 3, title: str = "") -> str:
    """Multi-row sheet with a row label per row (e.g. onboard/topdown rows).

    Args:
        rows: ``(row_label, [(tile_label, frame), ...])`` pairs; every row
            should have the same tile count.
        out_path: Destination PNG path.
        scale: Integer upscale per tile.
        title: Optional sheet-level title.

    Returns:
        The written path.
    """
    tiles: list[tuple[str, object]] = []
    cols = max(len(r[1]) for r in rows)
    for row_label, row in rows:
        for i, (lab, f) in enumerate(row):
            tiles.append((f"{row_label} {lab}" if i == 0 else lab, f))
    return sheet(tiles, out_path, cols=cols, scale=scale, title=title)
