"""Every reference example is a runnable Experiment class (no sim, no registry).

Guards the examples package against API drift without coupling to the exact
structure of any one example.
"""

import examples
from deepracer_genesis.experiment import Experiment, run
from deepracer_genesis.experiment.spec import ExperimentSpec


def test_examples_are_experiment_subclasses():
    for cls in examples.EXAMPLES:
        assert isinstance(cls, type) and issubclass(cls, Experiment)


def test_every_example_builds_a_valid_spec():
    for cls in examples.EXAMPLES:
        spec = run(cls, build_only=True)      # run dispatches the class directly
        assert isinstance(spec, ExperimentSpec)
        spec.validate()


def test_examples_blend_the_axes():
    """The set should span backends, modalities, renderers, and algorithms."""
    specs = [run(cls, build_only=True) for cls in examples.EXAMPLES]
    backends = {s.env.backend for s in specs}
    modalities = {s.env.modality for s in specs}
    renders = {s.env.render for s in specs}
    def _algo(s):
        if s.algorithm.cls is not None:
            return s.algorithm.cls.__name__
        return "PPOLagrangian" if s.algorithm.lagrangian.get("budget") else "PPO"
    algos = {_algo(s) for s in specs}
    views = {s.env.view for s in specs}
    assert {"cpu", "gpu"} <= backends
    assert {"feature", "camera"} <= modalities
    assert {"madrona", "nyx"} <= renders
    # PPOLagrangian left the set with the safe_rl examples (their TorchRL
    # backend was removed); the reference set is plain PPO until the
    # rsl-rl Lagrangian lands
    assert {"PPO"} <= algos
    assert "gui" in views
