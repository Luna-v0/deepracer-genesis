"""Shared search-space *types* for HPO and domain randomization (Part H).

A user declares a range/choice once and hands the same object to either
sampler — but the two sample at different *levels*, so this is a shared type
hierarchy with two verbs, not one object spanning both samplers:

- ``suggest(trial, name)`` — HPO draws ONE python scalar and freezes it into
  the ``ExperimentSpec`` for the whole trial.
- ``sample(n, device)`` — DR keeps the *range* and resamples an ``(n,)`` CUDA
  tensor every ``reset_idx``.

Not every space implements both honestly: ``Choice`` is HPO-only (a batched
GPU categorical DR does not exist here), and ``SymRange`` is DR-native (its
symmetric magnitude has no single scalar for HPO to freeze). Each space also
exposes ``to_cfg()`` producing the exact value ``builder.sim_cfg`` writes into
``cfg["rand"]`` — a ``(lo, hi)`` tuple for uniform ranges vs a scalar magnitude
for symmetric ones — so the shape ``randomization.domain_rand`` expects is
preserved.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:  # optuna is an HPO-only dependency; keep import-time light
    import optuna


@runtime_checkable
class Space(Protocol):
    """A declared search dimension usable by HPO and/or DR.

    Attributes:
        kind: Discriminator for how ``to_cfg()`` shapes the DR value
            (``"range"`` -> ``(lo, hi)`` tuple, ``"sym"`` -> scalar magnitude,
            ``"choice"`` -> categorical, DR-invalid).
    """

    kind: str

    def suggest(self, trial: optuna.Trial, name: str) -> Any:
        """Draw one python scalar for an HPO trial (frozen for the trial)."""

    def sample(self, n: int, device) -> torch.Tensor:
        """Draw a batched ``(n,)`` DR tensor on ``device`` (resampled each reset)."""

    def to_cfg(self) -> Any:
        """The value ``builder.sim_cfg`` writes into ``cfg["rand"]``."""


@dataclass(frozen=True)
class FloatRange:
    """A continuous ``[lo, hi]`` range; implements BOTH samplers honestly.

    Attributes:
        lo: Lower bound (inclusive).
        hi: Upper bound (inclusive).
        log: Sample on a log scale; requires ``lo > 0``.
    """

    lo: float
    hi: float
    log: bool = False
    kind: str = "range"

    def __post_init__(self):
        if self.hi < self.lo:
            raise ValueError(f"FloatRange hi < lo: ({self.lo}, {self.hi})")
        if self.log and self.lo <= 0:
            raise ValueError(
                f"log=True needs a strictly-positive lower bound; got lo={self.lo}")

    def suggest(self, trial: optuna.Trial, name: str) -> float:
        return trial.suggest_float(name, self.lo, self.hi, log=self.log)

    def sample(self, n: int, device) -> torch.Tensor:
        u = torch.rand(n, device=device)
        if self.log:
            lo, hi = math.log(self.lo), math.log(self.hi)
            return torch.exp(lo + (hi - lo) * u)
        return self.lo + (self.hi - self.lo) * u

    def to_cfg(self) -> tuple[float, float]:
        return (self.lo, self.hi)


@dataclass(frozen=True)
class IntRange:
    """An integer ``[lo, hi]`` range (inclusive); implements BOTH samplers.

    Attributes:
        lo: Lower bound (inclusive).
        hi: Upper bound (inclusive).
    """

    lo: int
    hi: int
    kind: str = "range"

    def __post_init__(self):
        if self.hi < self.lo:
            raise ValueError(f"IntRange hi < lo: ({self.lo}, {self.hi})")

    def suggest(self, trial: optuna.Trial, name: str) -> int:
        return trial.suggest_int(name, self.lo, self.hi)

    def sample(self, n: int, device) -> torch.Tensor:
        return torch.randint(self.lo, self.hi + 1, (n,), device=device)

    def to_cfg(self) -> tuple[int, int]:
        return (self.lo, self.hi)


@dataclass(frozen=True)
class SymRange:
    """A symmetric magnitude: DR samples in ``[-m, m]``, gated on ``m > 0``.

    DR-native (mass/COM shifts, camera jitters): there is no single scalar to
    freeze, so ``suggest`` raises. ``to_cfg`` emits the bare magnitude the DR
    code expects (a scalar, distinct from a ``(lo, hi)`` tuple).

    Attributes:
        m: Non-negative magnitude; ``m == 0`` disables the knob.
    """

    m: float
    kind: str = "sym"

    def __post_init__(self):
        if self.m < 0:
            raise ValueError(f"SymRange magnitude must be >= 0; got {self.m}")

    def suggest(self, trial: optuna.Trial, name: str):
        raise NotImplementedError(
            "SymRange is DR-native (a symmetric [-m, m] magnitude); it has no "
            "single scalar to freeze into an HPO trial. Search the bound with a "
            "FloatRange(0, m_max) and build a SymRange from the drawn value.")

    def sample(self, n: int, device) -> torch.Tensor:
        return -self.m + 2 * self.m * torch.rand(n, device=device)

    def to_cfg(self) -> float:
        return self.m


@dataclass(frozen=True)
class Choice:
    """A categorical list; HPO-only (``sample`` raises).

    There is no batched-GPU categorical DR in this repo (string/tuple choices
    like ``'relu'`` or ``(256, 128)`` cannot be a CUDA tensor), so ``sample``
    is intentionally unimplemented.

    Attributes:
        values: The candidate values HPO chooses among.
    """

    values: Sequence[Any]
    kind: str = "choice"

    def __post_init__(self):
        if len(self.values) == 0:
            raise ValueError("Choice needs at least one value")

    def suggest(self, trial: optuna.Trial, name: str) -> Any:
        return trial.suggest_categorical(name, list(self.values))

    def sample(self, n: int, device) -> torch.Tensor:
        raise NotImplementedError(
            "Choice is HPO-only: a batched-GPU categorical DR (string/tuple "
            "choices) is not supported. Use FloatRange/IntRange/SymRange for DR.")

    def to_cfg(self):
        raise NotImplementedError(
            "Choice has no cfg['rand'] shape (categorical DR is unsupported).")
