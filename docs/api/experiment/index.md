# `deepracer_genesis.experiment`

The experiment layer is the **config-as-code surface**. You compose a `>>` chain of
stages, `build()` it into a frozen `ExperimentSpec`, and `run()` trains it; the
evaluator measures the result and the report/visualize modules summarize it.

For the conceptual walkthrough of the DSL see [Experiments & the `>>` DSL](../../concepts/experiments.md).
This section is the **API reference**, one page per module.

## Lifecycle

```
stages  ──>>──▶  Pipeline.build()  ──▶  ExperimentSpec  ──▶  run() → run_rsl()
                                                                     │
                                             evaluator ◀────────────┘
                                                  │
                                       report / visualize
```

## Modules

| Module | What it is |
|--------|------------|
| [spec](spec.md) | The frozen `ExperimentSpec` and its typed sub-specs (`EnvSpec`, `ObsDRSpec`, `EncoderSpec`, `PolicySpec`, `ActionDRSpec`, `AlgorithmSpec`), plus `validate()`, `to_dict()`, `id()`. |
| [stages](stages.md) | The `>>` DSL: `Stage`/`Pipeline` and every stage (envs, DR, encoders, policies, algorithms) folded into a spec. |
| [authoring](authoring.md) | The `Experiment` base class (`pipeline`/`spec`/`run`) — subclass it to author an experiment. |
| [run](run.md) | The `build()` and `run()` entry points. |
| [trainer](trainer.md) | The rsl-rl training backend (`run_rsl`, `spec_to_train_cfg`). |
| [evaluator](evaluator.md) | `EvalRecord`, `evaluate_policy`, `aggregate_episodes`. |
| [overrides](overrides.md) | `override()` for building spec variants. |
| [report](report.md) | Aggregate runs into comparison tables (`load_records`, `spec_axes`, `grouped_rows`, `build_report`). |
| [visualize](visualize.md) | `rollout_video`, `dr_preview_video`. |
