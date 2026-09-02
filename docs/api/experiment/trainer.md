# Training backend (rsl-rl)

`run_rsl()` is the training loop: it drives rsl-rl's `OnPolicyRunner` (collection
+ algorithm update), periodic in-loop evaluation, the out-of-loop per-track
holdout eval, checkpointing, and charts. `rsl_supported()` gates which specs
dispatch here; `spec_to_train_cfg()` translates an `ExperimentSpec` into the
runner's config (including a custom `Algo(cls=...)` class).

`ExperimentSpec.resume` loads a checkpoint's **actor and critic weights only**:
the optimizer, iteration counter and RND state stay fresh, so the `lr` and
`schedule` the spec asked for are the ones the run uses.

::: deepracer_genesis.experiment.rsl_backend
