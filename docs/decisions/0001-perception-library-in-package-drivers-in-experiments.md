# 0001 — Perception: library code in the package, run drivers in `experiments/`

- Status: accepted
- Date: 2026-09-01

## Context

PR #5 ("Cnn perception") added a top-level `perception/` package of ~1,800 LOC
holding three different kinds of code at once: feature sets the env constructs at
runtime, a supervised dataset reader, and twenty one-off scripts that produced the
numbers in the PR's write-up.

At the repo root, `perception/` fell outside
`[tool.setuptools.packages.find] include = ["deepracer_genesis*"]`, so it shipped in
no wheel. It imported only because `sys.path[0]` is the working directory — the same
mechanism `examples/` relies on, and `tests/test_perception.py` had to insert the
repo root by hand with the comment *"perception/ is not an installed package"*.

Nothing inside `deepracer_genesis/` imported it, so the package was not broken. But
`CNNPerceptionFeatures` and `NoisyPerceptionFeatures` are adapters of the public
`PerceptionFeatures` extension point, and `resolve_feature_set` takes a class object:
a user who installs the wheel could never obtain those classes to pass in. The
deployment story the PR is built around — seven channels from the camera, twenty-two
computed onboard, the same policy weights on either source — is unreachable from an
installed package.

## Decision

Split on one line: **does a trained policy need this code at runtime to drive?**

- Yes → `deepracer_genesis/perception/` — `model.py`, `features.py` (the merged
  `cnn_features.py` + `noisy_features.py`), `dataset.py`, `augment.py`.
- No → `experiments/perception/` — `data_generation.py`, `train_cnn.py`, the two
  `train_policy_with_*.py`, and the `evaluation/` and `visualization/` trees.

The env grew a public `camera_stack` property so the feature set reads a documented
seam instead of the private `_stack_buf`. `CNNPerceptionFeatures` now requires an
explicit `checkpoint` and validates it against the env's `frame_stack` at
construction.

`dataset.py` keeps reading parquet with `pandas`, which is declared as a new
`perception` extra and added to the dev group. A port to `pyarrow` (already a core
dependency) would have removed the dependency entirely, but pandas is the
maintainer's preferred tool for this kind of work and the cost is confined to an
opt-in extra.

## Alternatives

**Leave `perception/` at the repo root.** Zero cost, and it matches the precedent of
`examples/`, `experiments/`, `scripts/` and `benchmarks/`. Rejected: those are all
demo or driver code, whereas the feature sets are runtime adapters of a public port.
The precedent covers the drivers, which is exactly where they now live.

**Move the whole tree into `deepracer_genesis/perception/`.** One `git mv`, no
judgement calls. Rejected: it puts ~1,250 LOC of single-use research scripts into
the wheel, along with their `pandas`/`matplotlib` needs, and freezes hardcoded track
lists and laptop-sized defaults into the distribution.

**Put `dataset.py` in `datasets/` next to `rollout.py`.** The writer of the parquet
format lives there, so the reader arguably should too. Rejected as the weaker pull:
keeping the perception subsystem cohesive in one package beat co-locating the format;
the `t` column is now used to validate stacks, which makes the coupling explicit
without co-location.

## Consequences

- The perception feature sets ship, so a camera policy is runnable from an installed
  package and the ONNX/car path stays reachable.
- `deepracer_genesis.perception.dataset` needs `pandas`, so it is guarded behind
  the `perception` extra. The package `__init__` deliberately exports only
  `model` and `features`, so `import deepracer_genesis.perception` does NOT pull
  pandas — the car and the run-a-trained-CNN path stay free of it. Only the
  dataset reader, used to train a CNN, requires the extra.
- `experiments/perception/` is not importable from an installed wheel. That is
  intended, but it means `tests/test_perception.py` reaching into
  `experiments.perception.train_policy_with_noise` for the holdout-split invariant
  only works from a source checkout. If that becomes a problem, the track split
  should move into the package rather than the test being weakened.
- `CNNPerceptionFeatures` no longer has a default checkpoint. Any caller must pass
  one; a missing file now raises instead of silently serving privileged simulator
  values. Existing invocations must be updated.
- Checkpoints move out of the source tree to `runs/cnn/`, which `.gitignore` already
  covers.
