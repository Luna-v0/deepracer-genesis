"""``spec.seed`` is actually applied by ``run()`` (P14): unit + end-to-end.

The smoke test builds real CPU sims (~20 s); deselect with ``-m "not smoke"``.
"""

import math
import random

import numpy as np
import pytest
import torch

from deepracer_genesis.seeding import seed_everything


def _draws():
    return (random.random(), np.random.rand(), torch.rand(2).tolist())


def test_seed_everything_pins_python_numpy_and_torch():
    seed_everything(123)
    first = _draws()
    seed_everything(123)
    assert _draws() == first
    seed_everything(124)
    assert _draws() != first


@pytest.mark.smoke
def test_run_is_reproducible_under_spec_seed(tmp_path):
    """Same spec + seed => bit-identical eval metrics; different seed differs.

    CPU backend, feature modality, tiny budget — deterministic enough to pin
    exactly (verified in-process AND cross-process when P14 was fixed).
    """
    from deepracer_genesis.experiment import PPO, FeatureEnvironment, VectorPolicy, run

    def one(seed, variant):
        return run(
            FeatureEnvironment(tracks=("reinvent_base",), num_envs=4,
                               backend="cpu")
            >> VectorPolicy() >> PPO(),
            root=str(tmp_path), total_env_steps=96, seed=seed,
            group="seeding", variant=variant)

    a, b, c = one(7, "a"), one(7, "b"), one(11, "c")
    assert a.seed == 7 and c.seed == 11          # the record reports the seed
    diverged = {
        k: (a.metrics[k], b.metrics[k]) for k in a.metrics
        if a.metrics[k] != b.metrics[k]
        and not (math.isnan(a.metrics[k]) and math.isnan(b.metrics[k]))}
    assert not diverged, f"same-seed runs diverged: {diverged}"
    assert any(a.metrics[k] != c.metrics[k] for k in a.metrics
               if not math.isnan(a.metrics[k])), "seed had no effect on the run"
