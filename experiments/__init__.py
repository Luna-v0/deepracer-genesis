"""Your experiments live here — this package is your blank canvas.

Author each experiment as an ``Experiment`` subclass (copy one from
``examples/`` as a starting point):

    from deepracer_genesis.experiment import Experiment, FeatureEnvironment, VectorPolicy

    class MyRun(Experiment):
        total_env_steps = 5_000_000
        group = "my_study"

        def pipeline(self):
            return FeatureEnvironment(num_envs=1024) >> VectorPolicy(keys=("state",))

Run it directly — ``MyRun().run()``, ``run(MyRun)``, a ``__main__`` block, or
``python -m deepracer_genesis.experiment experiments.my_run:MyRun``. There is no
name registry: an experiment is referenced by its class.
"""
