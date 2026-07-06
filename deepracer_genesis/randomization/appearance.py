"""Scene-level appearance randomization: per-env track colors and textures.

Genesis compiles the scene once, so appearance cannot be resampled per
episode. Instead this module bakes N *visual variants* of a track (same
geometry, different textures/colors) and the env loads them as one
heterogeneous entity — each parallel env renders its own variant, so every
training batch contains the full appearance distribution. This is the same
mechanism (and the same balanced env->morph assignment) as heterogeneous
multi-track training.

Variant recipe, per texture of the track's composite mesh:
- every texture gets a random per-variant RGB tint (`tint` range);
- textures whose filename looks like the ROAD surface can additionally be
  swapped for one of the alternate surface materials that ship with the
  DeepRacer assets (Brick/Carpet/Concrete/Grass `*_DIFF` images), retiled to
  the original resolution (`swap_road_materials`);
- lane-line textures get their own (usually milder) `line_tint` — they are
  the task-relevant visual cue, so they default to subtler variation.

The composite DAEs reference textures RELATIVELY (`textures/x.png`), so a
variant is just a copy of the .dae next to a directory of modified textures —
no XML rewriting. Variants are deterministic in (track, params, seed) and
cached under ~/.cache/deepracer_genesis/appearance/.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

import numpy as np

_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "deepracer_genesis",
                      "appearance")
_ROAD_RE = re.compile(r"road", re.IGNORECASE)
_LINE_RE = re.compile(r"line", re.IGNORECASE)
_ALT_SURFACE_RE = re.compile(r"_DIFF\.(png|jpe?g)$", re.IGNORECASE)


def _tint_image(img, rgb):
    """Multiply an image's RGB channels by an (r, g, b) factor triple.

    Keeps the source's alpha-ness: adding an alpha channel to an opaque
    texture flips Madrona onto its alpha-cutout path, which renders with
    magenta background bleed (known gs-madrona quirk)."""
    from PIL import Image
    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    mode = "RGBA" if has_alpha else "RGB"
    arr = np.asarray(img.convert(mode)).astype(np.float32)
    arr[..., :3] *= np.asarray(rgb, dtype=np.float32)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), mode)


def generate_track_variants(mesh_path: str, n: int, *, seed: int = 0,
                            tint: tuple = (0.6, 1.4),
                            line_tint: tuple = (0.9, 1.1),
                            swap_road_materials: bool = True) -> list[str]:
    """Bake `n` appearance variants of the composite mesh at `mesh_path`.

    Returns the list of variant .dae paths (cached: a second call with the
    same parameters is free). Tints/swaps are drawn from a dedicated RNG so
    the variant set is reproducible across processes and machines.
    """
    from PIL import Image

    src_dir = os.path.dirname(mesh_path)
    tex_dir = os.path.join(src_dir, "textures")
    if not os.path.isdir(tex_dir):
        raise FileNotFoundError(f"no textures/ directory next to {mesh_path}")

    key_payload = json.dumps({"mesh": os.path.basename(mesh_path), "n": n,
                              "seed": seed, "tint": tint, "line_tint": line_tint,
                              "swap": swap_road_materials}, sort_keys=True)
    key = hashlib.sha1(key_payload.encode()).hexdigest()[:10]
    base = os.path.basename(mesh_path)
    root = os.path.join(_CACHE, base.rsplit(".", 1)[0], key)

    paths = [os.path.join(root, f"var_{i:02d}", base) for i in range(n)]
    if all(os.path.exists(p) for p in paths):
        return paths

    rng = np.random.default_rng(seed)
    textures = sorted(os.listdir(tex_dir))
    alternates = [t for t in textures if _ALT_SURFACE_RE.search(t)]

    dae_text = open(mesh_path, encoding="utf-8").read()
    for i, var_dae in enumerate(paths):
        var_dir = os.path.dirname(var_dae)
        var_tex = os.path.join(var_dir, "textures")
        os.makedirs(var_tex, exist_ok=True)

        body_rgb = rng.uniform(*tint, size=3)
        line_rgb = rng.uniform(*line_tint, size=3)
        swap_to = (rng.choice(alternates)
                   if swap_road_materials and alternates and rng.random() < 0.75
                   else None)

        # texture filenames must be UNIQUE per variant: genesis's mesh
        # preprocessing cache dedups byte-identical meshes, so identical DAE
        # copies would all resolve to the first variant's textures
        var_text = dae_text
        for name in textures:
            if _ALT_SURFACE_RE.search(name):
                continue                     # source material, not referenced
            var_name = f"v{i:02d}_{name}"
            var_text = var_text.replace(f"textures/{name}",
                                        f"textures/{var_name}")
            img = Image.open(os.path.join(tex_dir, name))
            if _ROAD_RE.search(name) and swap_to:
                # band-limit hard: Madrona samples without mipmaps, so any
                # high-frequency, high-contrast pattern (brick, grass) aliases
                # into pixel noise at road viewing distances. Downsampling to
                # 32px and bilinear-upscaling keeps the material's color
                # character but kills the frequencies that speckle. The
                # ORIGINAL texture's alpha channel is kept — some alternates
                # carry transparency masks that Madrona bleeds magenta through
                alt = (Image.open(os.path.join(tex_dir, swap_to)).convert("RGB")
                       .resize((32, 32), Image.LANCZOS)
                       .resize(img.size, Image.BILINEAR))
                if img.mode == "RGBA":
                    img = Image.merge("RGBA", (*alt.split(),
                                               img.convert("RGBA").getchannel("A")))
                else:
                    img = alt
            rgb = line_rgb if _LINE_RE.search(name) else body_rgb
            out = _tint_image(img, rgb)
            if name.lower().endswith((".jpg", ".jpeg")):
                out = out.convert("RGB")     # JPEG has no alpha channel
            out.save(os.path.join(var_tex, var_name))
        with open(var_dae, "w", encoding="utf-8") as f:
            f.write(var_text)
    return paths


def generate_field_planes(n: int, *, seed: int = 0, size_m: float = 60.0,
                          base_color: tuple = (0.30, 0.48, 0.32),
                          tint: tuple = (0.5, 1.5)) -> list[str]:
    """Bake `n` ground-plane OBJ quads with per-variant diffuse colors.

    The env's ground plane is what shows through where DAE ground materials
    render transparent under Madrona; a heterogeneous list of these quads
    gives each parallel env its own field color (color lives in the MTL, so
    it survives the batch renderer, unlike per-entity surface colors on
    primitives).
    """
    rng = np.random.default_rng(seed ^ 0x5EED)
    key = hashlib.sha1(json.dumps([n, seed, size_m, base_color, tint],
                                  sort_keys=True).encode()).hexdigest()[:10]
    root = os.path.join(_CACHE, "field_planes", key)
    paths = [os.path.join(root, f"field_{i:02d}.obj") for i in range(n)]
    if all(os.path.exists(p) for p in paths):
        return paths

    os.makedirs(root, exist_ok=True)
    s = size_m / 2
    for i, p in enumerate(paths):
        r, g, b = (np.asarray(base_color) * rng.uniform(*tint, size=3)).clip(0, 1)
        mtl = os.path.basename(p).replace(".obj", ".mtl")
        with open(os.path.join(root, mtl), "w") as f:
            f.write(f"newmtl field\nKd {r:.4f} {g:.4f} {b:.4f}\nKa 0 0 0\nKs 0 0 0\n")
        with open(p, "w") as f:
            f.write(f"mtllib {mtl}\n"
                    f"v -{s} -{s} 0\nv {s} -{s} 0\nv {s} {s} 0\nv -{s} {s} 0\n"
                    "vn 0 0 1\nvn 0 0 1\nvn 0 0 1\nvn 0 0 1\n"
                    "usemtl field\nf 1//1 2//2 3//3\nf 1//1 3//3 4//4\n")
    return paths
