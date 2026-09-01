"""Reporter unit tests over synthetic EvalRecords.

Self-contained: builds the archetype specs it needs inline via the DSL, so it
does not depend on any authored experiments/examples package.
"""

from deepracer_genesis.experiment import (
    AsymmetricCameraPolicy,
    CameraEnvironment,
    DomainRandomizationActions,
    DomainRandomizationCamera,
    DomainRandomizationPhysics,
    DomainRandomizationTrackAppearance,
    FeatureEnvironment,
    FrozenCNNToFeatureVector,
    SafeRLCameraEnvironment,
    VectorPolicy,
)
from deepracer_genesis.experiment.evaluator import EvalRecord
from deepracer_genesis.experiment.report import grouped_rows, spec_axes


def _camera_full_dr():
    return (CameraEnvironment(render="madrona", num_envs=64)
            >> DomainRandomizationTrackAppearance(strength=0.6)
            >> DomainRandomizationCamera(brightness=(0.7, 1.3), camera_jitter=True)
            >> DomainRandomizationPhysics()
            >> AsymmetricCameraPolicy(actor_keys=("camera",),
                                      critic_keys=("camera", "state"))
            >> DomainRandomizationActions(steer_noise=0.02)).build()


def _camera_no_dr():
    return (CameraEnvironment(render="madrona", num_envs=64)
            >> AsymmetricCameraPolicy(actor_keys=("camera",),
                                      critic_keys=("camera", "state"))).build()


def _safe_transfer():
    return (SafeRLCameraEnvironment(render="madrona",
                                    cost="offtrack_or_overspeed", budget=25.0)
            >> DomainRandomizationCamera(brightness=(0.7, 1.3))
            >> FrozenCNNToFeatureVector(checkpoint="x.pt", output_dim=256)
            >> VectorPolicy(keys=("encoded", "state"))
            >> DomainRandomizationActions(steer_noise=0.02)).build()


def _feature():
    return (FeatureEnvironment(num_envs=64) >> VectorPolicy(keys=("state",))).build()


def _rec(spec, variant, group, seed=0, **metrics):
    return EvalRecord(spec_id=spec.id(), spec=spec.to_dict(), seed=seed,
                      group=group, variant=variant, metrics=metrics)


def test_spec_axes_derivation():
    axes = spec_axes(_camera_full_dr().to_dict())
    assert axes == {"modality": "camera", "render": "madrona",
                    "algorithm": "PPO", "asymmetry": "asymmetric",
                    "encoder": "none", "dr_profile": "full"}
    axes = spec_axes(_camera_no_dr().to_dict())
    assert axes["dr_profile"] == "none" and axes["asymmetry"] == "asymmetric"
    axes = spec_axes(_safe_transfer().to_dict())
    assert axes["algorithm"] == "PPOLagrangian"
    assert axes["encoder"] == "frozen_cnn"
    assert axes["dr_profile"] == "obs+action"


def test_grouped_rows_aggregate_over_seeds():
    feat = _feature()
    recs = [_rec(feat, "feature", "baselines", seed=s, completion_rate=0.9 + 0.02 * s)
            for s in range(2)]
    rows = grouped_rows(recs)
    assert len(rows) == 1
    mean, std = rows[0]["completion_rate"]
    assert abs(mean - 0.91) < 1e-9 and std > 0
    assert rows[0]["n_runs"] == 2
