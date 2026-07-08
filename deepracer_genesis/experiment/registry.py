"""Name registry + Experiment base class (plan section 1.2).

A registered name is a handle to Python code, not a config file path.
Functions register via @experiment; Experiment subclasses auto-register
under their class name.
"""

from __future__ import annotations

from typing import Callable, Union

REGISTRY: dict[str, Union[Callable, type]] = {}


def experiment(fn=None, *, name: str | None = None):
    # (decorator) register a zero-arg spec factory under `name` or fn.__name__
    """Register an experiment-building function under its (or a given) name.

    Usable bare (`@experiment`) or parameterized (`@experiment(name="foo")`).

    Args:
        fn: The zero-arg spec factory (filled in automatically in the bare
            form).
        name: Registry key; defaults to fn.__name__.

    Returns:
        The registered function unchanged (bare form) or the decorator.

    Raises:
        ValueError: If the name is already registered.
    """
    def deco(f):
        key = name or f.__name__
        if key in REGISTRY:
            raise ValueError(f"experiment name {key!r} already registered "
                             f"(by {REGISTRY[key]!r})")
        REGISTRY[key] = f
        return f
    return deco(fn) if fn is not None else deco


class Experiment:
    """Class idiom for authoring experiments.

    A single file defines the whole run: hyperparameters as class
    attributes, the env/DR/policy pipeline in pipeline(), and
    `MyExp().run()` executes it.

    A variant is a subclass with one attribute overridden, or an instance
    with keyword overrides:

        class SafeTransferTight(SafeTransfer): budget = 10.0
        SafeTransfer(budget=10.0, seed=3)

    Subclasses auto-register under their class name (prefix with '_' to opt
    out). Implement EITHER pipeline() — returning the `>>` chain, finalized
    with the standard attributes automatically — or spec() for full control.
    Runnable-single-file pattern (see experiments/template.py):

        if __name__ == "__main__":
            MyExperiment().run()

    Attributes:
        seed: Training seed.
        total_env_steps: Total environment steps to train for.
        eval_every_steps: Periodic-evaluation cadence in env steps
            (0 = final eval only).
        ablation_group: Group tag for the reporter's delta tables.
        variant: Variant tag within the group; defaults to the class name.
    """

    # ---- standard training configuration (overridable per subclass) ----
    seed: int = 0
    total_env_steps: int = 5_000_000
    eval_every_steps: int = 0
    ablation_group: str | None = None
    variant: str | None = None

    def __init__(self, **overrides):
        for key, value in overrides.items():
            if not hasattr(type(self), key):
                raise AttributeError(
                    f"{type(self).__name__} has no attribute {key!r} "
                    f"(available: {sorted(self._config_attrs())})")
            setattr(self, key, value)

    @classmethod
    def _config_attrs(cls) -> list[str]:
        attrs = set()
        for klass in cls.__mro__:
            attrs.update(k for k, v in vars(klass).items()
                         if not k.startswith("_") and not callable(v))
        return sorted(attrs)

    def __init_subclass__(cls, register: bool = True, **kwargs):
        super().__init_subclass__(**kwargs)
        if register and not cls.__name__.startswith("_"):
            if cls.__name__ in REGISTRY:
                raise ValueError(f"experiment name {cls.__name__!r} already registered")
            REGISTRY[cls.__name__] = cls

    # ------------------------------------------------------------------
    def pipeline(self) -> "Stage | Pipeline":
        """Return the `>>` stage chain (NOT built).

        The standard attributes (seed, total_env_steps, ...) are applied by
        spec() afterwards.

        Returns:
            The Stage or Pipeline defining the experiment.

        Raises:
            NotImplementedError: If the subclass overrides neither
                pipeline() nor spec().
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement pipeline() or spec()")

    def spec(self) -> ExperimentSpec:
        """Build the final ExperimentSpec.

        The default implementation finalizes pipeline() with the standard
        attributes (seed, total_env_steps, eval_every_steps, ablation_group,
        variant); override for full control over spec construction.

        Returns:
            The validated ExperimentSpec.
        """
        return self.pipeline().build(
            seed=self.seed,
            total_env_steps=self.total_env_steps,
            eval_every_steps=self.eval_every_steps,
            ablation_group=self.ablation_group,
            variant=self.variant or type(self).__name__,
        )

    def run(self, *, root: str = "runs", force: bool = False) -> "EvalRecord":
        """Train this experiment (identity-cached).

        Args:
            root: Runs directory.
            force: Re-train even when a cached record exists.

        Returns:
            The EvalRecord of the finished (or cached) run.
        """
        from .run import run as _run
        return _run(self, root=root, force=force)
