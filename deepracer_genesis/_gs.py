"""Single source of truth for ``gs.init`` (Part M).

The Genesis backend is a **config parameter** (``sim.backend``), not a CLI flag.
Every entry point routes through :func:`ensure_init` instead of calling
``gs.init(backend=gs.cuda)`` directly, so a run can select CPU or GPU from its
config and the choice cascades to ``gs.device`` (and thus every env tensor).

Genesis allows only one backend per process; :func:`ensure_init` is idempotent
and a second call with a different backend is ignored (the first one wins).
"""

from __future__ import annotations

import os
from typing import Literal

import genesis as gs

Backend = Literal["gpu", "cpu"]

_BACKENDS = {"gpu": lambda: gs.gpu, "cpu": lambda: gs.cpu}


def ensure_init(backend: Backend = "gpu", *, logging_level: str = "warning") -> None:
    """Initialize Genesis on ``backend`` once (idempotent).

    Args:
        backend: ``"gpu"`` (CUDA) or ``"cpu"``.
        logging_level: Genesis log level.

    Raises:
        ValueError: If ``backend`` is not ``"gpu"`` or ``"cpu"``.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be 'gpu' or 'cpu', got {backend!r}")
    # P10: Madrona's default device heap fails init / kills processes when the
    # GPU is shared (e.g. ollama resident). Madrona reads this at scene build,
    # which always happens after ensure_init, so setting it here covers every
    # entry point; an explicit environment value still wins.
    os.environ.setdefault("MADRONA_MWGPU_DEVICE_HEAP_SIZE", str(1 << 30))
    try:
        gs.init(backend=_BACKENDS[backend](), logging_level=logging_level)
    except Exception as e:  # already initialized in this process
        if "initialized" not in str(e).lower():
            raise
