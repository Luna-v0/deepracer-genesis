"""The ``Experiment`` authoring base class (no registry).

Author an experiment as a class: hyperparameters are class attributes, the
pipeline goes in ``pipeline()`` (or override ``spec()`` for full control), and
you run it directly — ``MyExperiment().run()`` or ``run(MyExperiment)``. There
is no name registry: an experiment is referenced by its Python class, not a
string handle.
"""

from __future__ import annotations


class Experiment:
    """Author an experiment as a single file: hyperparameters as class
    attributes, the pipeline in pipeline(), and `MyExp().run()` to execute.

    Attributes:
        seed: Random seed for reproducibility.
        total_env_steps: Number of environment steps to train for.
        eval_every_steps: Evaluation interval in steps (0 disables periodic eval).
        resume: Checkpoint the policy starts from, or None to train from scratch.
        ablation_group: Optional grouping label for ablation studies.
        variant: Optional variant label, defaulting to the subclass name.
    """

    # ---- standard training configuration (overridable per subclass) ----
    seed: int = 0
    total_env_steps: int = 5_000_000
    eval_every_steps: int = 0
    resume: str | None = None
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

    # ------------------------------------------------------------------
    def pipeline(self) -> "Stage | Pipeline":
        """Return the `>>` stage chain (NOT built); spec() finalizes it with
        the standard attributes afterwards.

        Returns:
            The Stage or Pipeline defining the experiment.

        Raises:
            NotImplementedError: If the subclass overrides neither
                pipeline() nor spec().
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement pipeline() or spec()")

    def spec(self) -> ExperimentSpec:
        """Build the final ExperimentSpec by finalizing pipeline() with the
        standard attributes; override for full control over construction.

        Returns:
            The validated ExperimentSpec.
        """
        return self.pipeline().build(
            seed=self.seed,
            total_env_steps=self.total_env_steps,
            eval_every_steps=self.eval_every_steps,
            resume=self.resume,
            ablation_group=self.ablation_group,
            variant=self.variant or type(self).__name__,
        )

    def run(self, *, root: str = "runs") -> "EvalRecord":
        """Train this experiment (always from scratch — no result cache).

        Args:
            root: Runs directory.

        Returns:
            The EvalRecord of the finished run.
        """
        from .run import run as _run
        return _run(self, root=root)
