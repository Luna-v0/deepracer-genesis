"""Reporter unit tests over synthetic EvalRecords (Phase 6)."""

import experiments  # noqa: F401
from deepracer_genesis.experiment import run
from deepracer_genesis.experiment.evaluator import EvalRecord
from deepracer_genesis.experiment.report import (
    delta_rows,
    grouped_rows,
    spec_axes,
)


def _rec(target, variant, group, seed=0, **metrics):
    spec = run(target, build_only=True)
    return EvalRecord(spec_id=spec.id(), spec=spec.to_dict(), seed=seed,
                      ablation_group=group, variant=variant, metrics=metrics)


def test_spec_axes_derivation():
    axes = spec_axes(run("cam_baseline", build_only=True).to_dict())
    assert axes == {"modality": "camera", "render": "madrona",
                    "algorithm": "ppo", "asymmetry": "asymmetric",
                    "encoder": "none", "dr_profile": "full"}
    axes = spec_axes(run("cam_plain", build_only=True).to_dict())
    assert axes["dr_profile"] == "none" and axes["asymmetry"] == "asymmetric"
    axes = spec_axes(run("SafeTransfer", build_only=True).to_dict())
    assert axes["algorithm"] == "ppo_lagrangian"
    assert axes["encoder"] == "frozen_cnn"
    assert axes["dr_profile"] == "obs+action"


def test_grouped_rows_aggregate_over_seeds():
    recs = [_rec("feature_baseline", "feature", "baselines", seed=s,
                 completion_rate=0.9 + 0.02 * s) for s in range(2)]
    rows = grouped_rows(recs)
    assert len(rows) == 1
    mean, std = rows[0]["completion_rate"]
    assert abs(mean - 0.91) < 1e-9 and std > 0
    assert rows[0]["n_runs"] == 2


def test_delta_rows_pick_baseline_and_diff():
    recs = [
        _rec("cam_plain", "no_dr", "dr_effect", completion_rate=0.5),
        _rec("cam_full_dr", "full_dr", "dr_effect", completion_rate=0.8),
    ]
    d = delta_rows(recs)
    assert d["dr_effect"]["baseline"] == "no_dr"
    delta, _ = d["dr_effect"]["deltas"]["full_dr"]["completion_rate"]
    assert abs(delta - 0.3) < 1e-9


def test_single_variant_groups_skipped():
    recs = [_rec("feature_baseline", "feature", "baselines", completion_rate=1.0)]
    assert delta_rows(recs) == {}
