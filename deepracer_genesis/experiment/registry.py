"""Name registry + Experiment base class (plan section 1.2).

A registered name is a handle to Python code, not a config file path.
Functions register via @experiment; Experiment subclasses auto-register
under their class name.
"""

from __future__ import annotations

from typing import Callable, Union

REGISTRY: dict[str, Union[Callable, type]] = {}


def experiment(fn=None, *, name: str | None = None):
    """Register an experiment-building function under its (or a given) name."""
    def deco(f):
        key = name or f.__name__
        if key in REGISTRY:
            raise ValueError(f"experiment name {key!r} already registered "
                             f"(by {REGISTRY[key]!r})")
        REGISTRY[key] = f
        return f
    return deco(fn) if fn is not None else deco


class Experiment:
    """Class idiom for authoring experiments — lean on this for ablations.

    Class attributes are the configuration surface; a variant is a subclass
    with one attribute overridden, or an instance with keyword overrides:

        class SafeTransferTight(SafeTransfer): budget = 10.0
        SafeTransfer(budget=10.0, seed=3)

    Subclasses auto-register under their class name (prefix with '_' to opt
    out). Override spec() to run the `>>` chain and .build() the spec.
    """

    def __init__(self, **overrides):
        for key, value in overrides.items():
            if not hasattr(type(self), key):
                raise AttributeError(
                    f"{type(self).__name__} has no attribute {key!r} "
                    f"(available: {sorted(self._config_attrs())})")
            setattr(self, key, value)

    @classmethod
    def _config_attrs(cls):
        return [k for k in vars(cls) if not k.startswith("_") and k != "spec"
                and not callable(getattr(cls, k))]

    def __init_subclass__(cls, register: bool = True, **kwargs):
        super().__init_subclass__(**kwargs)
        if register and not cls.__name__.startswith("_"):
            if cls.__name__ in REGISTRY:
                raise ValueError(f"experiment name {cls.__name__!r} already registered")
            REGISTRY[cls.__name__] = cls

    def spec(self):
        raise NotImplementedError(
            f"{type(self).__name__} must implement spec() -> ExperimentSpec")
