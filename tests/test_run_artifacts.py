"""Run-artifact retention: checkpoint pruning, best tracking, telemetry subset.

The unit slice is pure; the ``smoke`` slice trains tiny real CPU runs (~20 s).
"""

import math
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest
import torch

from deepracer_genesis.analysis.telemetry import TelemetryRecorder, telemetry_env_ids
from deepracer_genesis.experiment import (PPO, EvalConfig, Evaluation,
                                          FeatureEnvironment, VectorPolicy, run)
from deepracer_genesis.experiment.rsl_backend import (_improves,
                                                      _sweep_iter_checkpoints)

# ---------------------------------------------------------------------------
# pure helpers


@pytest.mark.parametrize(("value", "best", "expected"), [
    (1.0, None, True),
    (2.0, 1.0, True),
    (1.0, 1.0, False),            # ties keep the earlier checkpoint
    (0.5, 1.0, False),
    (math.nan, None, False),      # NaN never wins
    (None, None, False),          # eval saw no episode
])
def test_improves(value, best, expected):
    assert _improves(value, best) is expected


def test_sweep_removes_only_iteration_checkpoints(tmp_path):
    keep = ["model.pt", "model_best.pt", "model_last.pt", "model_1x.pt", "cfg.json"]
    drop = ["model_0.pt", "model_100.pt", "model_11384.pt"]
    for name in keep + drop:
        (tmp_path / name).write_bytes(b"x")
    assert _sweep_iter_checkpoints(str(tmp_path)) == len(drop)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(keep)


@pytest.mark.parametrize(("n", "k", "expected"), [
    (8, None, tuple(range(8))),
    (8, 8, tuple(range(8))),
    (8, 100, tuple(range(8))),
    (9, 3, (0, 4, 8)),
    (1024, 1, (0,)),
])
def test_telemetry_env_ids(n, k, expected):
    assert telemetry_env_ids(n, k) == expected


def test_telemetry_env_ids_evenly_cover_large_sims():
    ids = telemetry_env_ids(1024, 64)
    assert len(ids) == 64 and ids[0] == 0 and ids[-1] == 1023
    assert list(ids) == sorted(set(ids))


def _fake_env(n=6):
    """The attributes TelemetryRecorder reads, with env i's values == i."""
    ar = torch.arange(n, dtype=torch.float32)
    track = SimpleNamespace(
        variant_idx=torch.tensor([0, 0, 0, 1, 1, 1][:n]),
        names=["a", "b"], variant_offset=torch.zeros(2, 2),
        total_len_env=10.0 + ar)
    return SimpleNamespace(
        num_envs=n, dt=0.02, track=track,
        base_pos=torch.stack([ar, ar, ar], 1), yaw=ar,
        actions=torch.stack([ar, ar], 1), signals={"v_forward": ar},
        progress_m=ar, lateral=ar, half_width=ar)


def test_recorder_subset_keeps_schema_and_env_ids(tmp_path):
    env = _fake_env()
    rec = TelemetryRecorder(env, envs=(1, 4))
    zeros = torch.zeros(6)
    for _ in range(3):
        rec.step(torch.arange(6.0), torch.zeros(6, dtype=torch.bool),
                 torch.zeros(6, dtype=torch.bool), zeros)
    table = pq.read_table(rec.flush(str(tmp_path / "t.parquet")))
    full = TelemetryRecorder(env)
    assert table.num_rows == 3 * 2
    assert table.column("env").to_pylist() == [1, 4] * 3
    assert table.column("x").to_pylist() == [1.0, 4.0] * 3
    assert table.column("reward").to_pylist() == [1.0, 4.0] * 3
    assert table.column("track").to_pylist() == ["a", "b"] * 3
    assert table.column("track_len").to_pylist() == [11.0, 14.0] * 3
    assert list(full.env_ids) == list(range(6))


# ---------------------------------------------------------------------------
# spec knobs


@pytest.mark.parametrize("kwargs", [
    {"telemetry_envs": 0}, {"telemetry_envs": True}, {"telemetry_envs": 2.5},
    {"keep_checkpoints": "none"}, {"best_metric": "offtrack_rate"},
])
def test_eval_config_rejects_bad_knobs(kwargs):
    with pytest.raises(ValueError):
        EvalConfig(**kwargs)


def test_retention_knobs_are_hash_exempt():
    def spec(**ev):
        return run(FeatureEnvironment(tracks=("reinvent_base",), num_envs=4,
                                      backend="cpu")
                   >> VectorPolicy() >> PPO() >> Evaluation(**ev),
                   build_only=True, total_env_steps=96)

    base = spec().id()
    assert spec(telemetry_envs=None, keep_checkpoints="all",
                best_metric="completion_rate").id() == base
    assert spec(eval_num_envs=8).id() != base      # real eval knobs still hash


# ---------------------------------------------------------------------------
# end-to-end (smoke)


def _train(tmp_path, variant, on_eval=None, **ev):
    """Three eval chunks on 4 CPU envs; returns (record, run dir)."""
    rec = run(FeatureEnvironment(tracks=("reinvent_base",), num_envs=4,
                                 backend="cpu")
              >> VectorPolicy() >> PPO() >> Evaluation(**ev),
              root=str(tmp_path), total_env_steps=288, eval_every_steps=96,
              group="ret", variant=variant, on_eval=on_eval)
    return rec, next(Path(tmp_path, "ret").glob(f"{variant}-*"))


def _ckpts(run_dir):
    return sorted(p.name for p in run_dir.glob("model*.pt"))


@pytest.mark.smoke
def test_best_last_keeps_final_and_best_only(tmp_path):
    rec, run_dir = _train(tmp_path, "bl", telemetry_envs=2)
    assert _ckpts(run_dir) == ["model.pt", "model_best.pt"]
    best = torch.load(run_dir / "model_best.pt", map_location="cpu",
                      weights_only=False)
    assert "optimizer_state_dict" not in best
    assert best["best"]["metric"] == "mean_progress_m"
    assert rec.train["best_checkpoint"] == str(run_dir / "model_best.pt")
    assert rec.train["best_frames"] in {h["frames"] for h in rec.eval_history}
    # periodic telemetry is subsampled, the final one stays full
    periodic = sorted((run_dir / "telemetry").glob("eval_*.parquet"))
    assert periodic
    for p in periodic:
        assert set(pq.read_table(p, columns=["env"]).column("env").to_pylist()) == {0, 3}
    final = pq.read_table(run_dir / "telemetry" / "final.parquet", columns=["env"])
    assert set(final.column("env").to_pylist()) == {0, 1, 2, 3}


@pytest.mark.smoke
def test_best_checkpoint_warm_starts(tmp_path):
    rec, _ = _train(tmp_path, "src")
    child = run(FeatureEnvironment(tracks=("reinvent_base",), num_envs=4,
                                   backend="cpu") >> VectorPolicy() >> PPO(),
                root=str(tmp_path), total_env_steps=96, group="ret",
                variant="child", resume=rec.train["best_checkpoint"])
    assert child.train["checkpoint"]


@pytest.mark.smoke
def test_keep_all_restores_interval_checkpoints(tmp_path):
    _, run_dir = _train(tmp_path, "all", keep_checkpoints="all")
    names = _ckpts(run_dir)
    assert "model.pt" in names and "model_best.pt" in names
    assert any(n not in ("model.pt", "model_best.pt") for n in names)


@pytest.mark.smoke
def test_stopped_run_leaves_last_and_best(tmp_path):
    class Stop(Exception):
        pass

    def on_eval(frames, metrics):
        raise Stop

    with pytest.raises(Stop):
        _train(tmp_path, "pruned", on_eval=on_eval)
    run_dir = next(Path(tmp_path, "ret").glob("pruned-*"))
    assert set(_ckpts(run_dir)) <= {"model_best.pt", "model_last.pt"}
    assert "model_last.pt" in _ckpts(run_dir)
    last = torch.load(run_dir / "model_last.pt", map_location="cpu",
                      weights_only=False)
    assert "optimizer_state_dict" in last          # resumable
