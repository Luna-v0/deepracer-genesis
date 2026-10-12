"""Single home for domain-randomization definitions + the shared search types.

Definitions live here; application stays where it must happen (physics before
stepping, visual in the renderer, action/image DR env-side in the sim step):

- ``spaces``: declarative range/choice *types* (``FloatRange``/``IntRange``/
  ``SymRange``/``Choice``) shared by DR and HPO (Part H).
- ``physics``: per-env physics randomization applied at reset.
- ``visual``: world-color YIQ remap, camera-mount jitter, pixel noise.
- ``image_aug``: pure image-space DR applied env-side to the camera obs.
- ``catalog``: the one table of every knob (name, Space, layer, signals).
- ``appearance``: offline texture bake (not wired into the train path).
"""

from .catalog import CATALOG, Knob
from .physics import randomize_physics
from .spaces import Choice, FloatRange, IntRange, Space, SymRange

__all__ = [
    "CATALOG",
    "Choice",
    "FloatRange",
    "IntRange",
    "Knob",
    "Space",
    "SymRange",
    "randomize_physics",
]
