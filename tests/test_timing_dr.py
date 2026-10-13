"""Control timing DR, phase 1 (ADR 0005): control_dt_s, frame_interval_s,
the actuation timing knobs, horizon_s, and the dataset / CNN / model-card parity.

Spec-level checks are GPU-free; the sim checks build small CPU sims the way
test_perception_loop does. Knob-OFF bit-identity is pinned by the golden ids
here and the digest probe in the phase-1 report.
"""

import dataclasses
import io
import json
import math
import subprocess
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from deepracer_genesis.configs.cfgs import DEFAULT_CONTROL_DT_S
from deepracer_genesis.envs.rewards import deepracer
from deepracer_genesis.experiment import (
    PPO,
    AsymmetricCameraPolicy,
    CameraEnvironment,
    DomainRandomizationActions,
    FeatureEnvironment,
    SpecError,
    VectorPolicy,
)
from deepracer_genesis.experiment.builder import Builder
from deepracer_genesis.experiment.rsl_backend import (
    discount_for_horizon,
    spec_to_train_cfg,
)
from deepracer_genesis.experiment.spec import _active_knobs, control_decimation
from deepracer_genesis.perception import dataset as ds
from deepracer_genesis.perception.model import (
    PerceptionCNN,
    load_checkpoint,
    save_checkpoint,
)
from deepracer_genesis.randomization.catalog import BY_NAME, by_group

TIMING = {"latency_ms": (30.0, 90.0), "latency_jitter_ms": 10.0,
          "decision_drop": 0.05}


# ------------------------------------------------------------------ spec layer
def _feature(**env_kw):
    return FeatureEnvironment(num_envs=4, backend="cpu", **env_kw) >> VectorPolicy() >> PPO()


def test_control_dt_derives_an_integer_decimation():
    cfg = Builder(_feature(control_dt_s=0.07).build()).sim_cfg()
    assert cfg["sim"]["dt"] == 0.01 and cfg["sim"]["decimation"] == 7
    assert cfg["reward"]["action_rate_dt_ref"] == DEFAULT_CONTROL_DT_S
    assert control_decimation(0.02) == 2


@pytest.mark.parametrize("bad", [0.015, 0.0, -0.02, 0.0333])
def test_control_dt_refuses_a_non_integer_decimation(bad):
    with pytest.raises(SpecError, match="control_dt_s"):
        _feature(control_dt_s=bad).build()


def test_unset_timing_fields_keep_the_golden_ids():
    base = _feature().build(seed=0)
    assert base.id() == "954bac4294db"                       # phase-1 report
    assert base.env.control_dt_s is None and base.env.frame_interval_s is None
    cam = (CameraEnvironment() >> AsymmetricCameraPolicy() >> PPO()).build()
    assert cam.id() == "115343e98fa5"
    # the DR id stays too: new ActionDRSpec fields are exempt while unset
    dr = (FeatureEnvironment(num_envs=4, backend="cpu") >> VectorPolicy()
          >> DomainRandomizationActions(delay_steps=1, steer_noise=0.02)
          >> PPO()).build(seed=0)
    assert dr.id() == "4ed80d52ee6a"


def test_set_timing_fields_change_the_id():
    base = _feature().build(seed=0)
    assert _feature(control_dt_s=0.07).build(seed=0).id() != base.id()
    on = (FeatureEnvironment(num_envs=4, backend="cpu") >> VectorPolicy()
          >> DomainRandomizationActions(**TIMING) >> PPO()).build(seed=0)
    assert on.id() != base.id()
    assert {k.name for k, _ in _active_knobs(on)} == {
        "latency_ms", "latency_jitter_ms", "decision_drop"}


def test_frame_interval_needs_a_camera_stack_and_a_whole_stride():
    feature = _feature().build()            # the feature stage has no such field
    with pytest.raises(SpecError, match="camera env"):
        dataclasses.replace(feature, env=dataclasses.replace(
            feature.env, frame_interval_s=0.06)).validate()
    with pytest.raises(SpecError, match="frame_stack"):
        (CameraEnvironment(frame_stack=1, frame_interval_s=0.04)
         >> AsymmetricCameraPolicy()).build()
    with pytest.raises(SpecError, match="frame_interval_s"):
        (CameraEnvironment(frame_interval_s=0.03) >> AsymmetricCameraPolicy()).build()
    spec = (CameraEnvironment(control_dt_s=0.03, frame_interval_s=0.06)
            >> AsymmetricCameraPolicy()).build()
    cfg = Builder(spec).sim_cfg()
    assert cfg["sim"]["decimation"] == 3 and cfg["vision"]["frame_stride"] == 2


def test_timing_knobs_refuse_delay_steps_and_discrete_policies():
    with pytest.raises(SpecError, match="delay_steps"):
        (FeatureEnvironment() >> VectorPolicy()
         >> DomainRandomizationActions(delay_steps=1, latency_ms=(30, 90))).build()
    with pytest.raises(SpecError, match="discrete"):
        (FeatureEnvironment() >> VectorPolicy(actions=((-1, 0.5), (1, 0.5)))
         >> DomainRandomizationActions(latency_ms=(30, 90))).build()
    for bad in ({"latency_ms": (90, 30)}, {"latency_ms": (-5, 30)},
                {"latency_jitter_ms": -1}, {"decision_drop": 1.0}):
        with pytest.raises(SpecError):
            (FeatureEnvironment() >> VectorPolicy()
             >> DomainRandomizationActions(**bad)).build()


def test_builder_emits_the_timing_keys_only_when_set():
    off = Builder(_feature().build()).sim_cfg()
    assert "action_dr" not in off
    legacy = Builder((FeatureEnvironment() >> VectorPolicy()
                      >> DomainRandomizationActions(steer_noise=0.02)).build()).sim_cfg()
    assert set(legacy["action_dr"]) == {"steer_noise", "speed_noise", "delay_steps"}
    on = Builder((FeatureEnvironment() >> VectorPolicy()
                  >> DomainRandomizationActions(**TIMING)).build(seed=5)).sim_cfg()
    assert on["action_dr"]["latency_ms"] == (30.0, 90.0)
    assert on["action_dr"]["latency_jitter_ms"] == 10.0
    assert on["action_dr"]["decision_drop"] == 0.05
    assert on["sim"]["seed"] == 5


def test_catalog_groups_the_timing_knobs_as_dynamics():
    assert {k.name for k in by_group("dynamics")} >= {
        "latency_ms", "latency_jitter_ms", "decision_drop", "friction"}
    assert BY_NAME["track_width_scale"].group == "task"
    assert all(k.group == "observation" for k in [BY_NAME["blur"], BY_NAME["world_color"]])
    assert BY_NAME["latency_ms"].layer == "actuation"


# ------------------------------------------------------------------ horizon_s
def test_horizon_s_sets_gamma_from_the_control_period():
    cfg = spec_to_train_cfg((FeatureEnvironment(control_dt_s=0.07) >> VectorPolicy()
                             >> PPO(horizon_s=2.0, lam_horizon_s=1.0)).build())
    assert cfg["algorithm"]["gamma"] == pytest.approx(math.exp(-0.07 / 2.0))
    assert cfg["algorithm"]["lam"] == pytest.approx(math.exp(-0.07 / 1.0))
    assert discount_for_horizon(2.0, 0.02) == pytest.approx(0.99005, abs=1e-5)
    plain = spec_to_train_cfg((FeatureEnvironment() >> VectorPolicy() >> PPO()).build())
    assert plain["algorithm"]["gamma"] == 0.99 and plain["algorithm"]["lam"] == 0.95


def test_horizon_s_is_hash_exempt_when_unset_and_validated_when_set():
    off = (FeatureEnvironment() >> VectorPolicy() >> PPO()).build()
    assert "horizon_s" not in off.algorithm.ppo
    on = (FeatureEnvironment() >> VectorPolicy() >> PPO(horizon_s=2.0)).build()
    assert on.id() != off.id()
    with pytest.raises(SpecError, match="horizon_s"):
        (FeatureEnvironment() >> VectorPolicy() >> PPO(horizon_s=0.0)).build()


# ------------------------------------------------------------ action_rate
class _Stub:
    def __init__(self, dt, ref):
        self.dt = dt
        self.cfg = {"termination": {"wheel_margin": 0.08},
                    "action": {"max_speed": 3.0},
                    "reward": {"action_rate_dt_ref": ref}}
        self.lateral = torch.zeros(2)
        self.half_width = torch.full((2,), 0.35)
        self.d_progress = torch.zeros(2)
        self.v_forward = torch.ones(2)
        self.heading_err = torch.zeros(2)
        self.actions = torch.tensor([[0.0, 0.5], [0.6, 0.5]])
        self.last_actions = torch.zeros(2, 2)


def test_action_rate_is_scaled_to_the_reference_period_only_when_set():
    raw = -((_Stub(0.07, None).actions - torch.zeros(2, 2)) ** 2).sum(1)
    assert torch.allclose(deepracer(_Stub(0.07, None))["action_rate"], raw * 0.07)
    scaled = deepracer(_Stub(0.07, 0.02))["action_rate"]
    assert torch.allclose(scaled, raw * 0.07 * (0.02 / 0.07) ** 2)
    assert torch.allclose(deepracer(_Stub(0.02, 0.02))["action_rate"], raw * 0.02)


# ------------------------------------------------------------- CNN interval
def _tiny_cnn(path, stack, hw, interval):
    torch.manual_seed(0)
    net = PerceptionCNN(in_channels=3 * stack, n_targets=7, input_hw=hw,
                        channels=(4,), kernels=(5,), strides=(4,), head=8)
    save_checkpoint(net, path, frame_interval_s=interval)
    return path


def test_checkpoints_carry_and_reload_the_frame_interval(tmp_path):
    p = _tiny_cnn(tmp_path / "cnn.pt", 3, (48, 64), 0.06)
    assert load_checkpoint(p).frame_interval_s == 0.06
    q = tmp_path / "bare.pt"
    torch.save({"arch": dict(load_checkpoint(p).arch),
                "state_dict": load_checkpoint(p).state_dict()}, q)
    assert load_checkpoint(q).frame_interval_s is None    # pre-field checkpoint


# ------------------------------------------------------------- dataset stride
HW = (24, 32)
FRAMES = 8
N_STATE = 10
SLICE = (2, 9)


def _png(rng):
    img = Image.fromarray(rng.integers(0, 255, (*HW, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _make_track(root, name, seed, control_dt=0.02):
    import pandas as pd
    rng = np.random.default_rng(seed)
    d = root / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps(
        {"cnn_target_slice": list(SLICE), "control_dt": control_dt}))
    pd.DataFrame({
        "image": [_png(rng) for _ in range(FRAMES)],
        "state": [rng.uniform(-0.5, 0.5, N_STATE).astype(np.float32)
                  for _ in range(FRAMES)],
        "env": np.zeros(FRAMES, np.int32),
        "episode": np.zeros(FRAMES, np.int32),
        "t": np.arange(FRAMES, dtype=np.int64),
    }).to_parquet(d / "rollout_000.parquet")


@pytest.fixture()
def data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(ds, "CACHE", tmp_path / "cache")
    _make_track(tmp_path, "alpha", seed=1)
    _make_track(tmp_path, "slow", seed=2, control_dt=0.07)
    return tmp_path


def test_stride_stacks_every_sth_row_and_records_the_interval(data_root):
    consecutive = ds.RolloutDataset(tracks=("alpha",), k=3)
    strided = ds.RolloutDataset(tracks=("alpha",), k=3, stride=2)
    assert len(consecutive) == FRAMES - 2 and len(strided) == FRAMES - 4
    frames, target = strided[1]                       # rows 1, 3, 5
    single = ds.RolloutDataset(tracks=("alpha",), k=1)
    assert torch.equal(frames[0:3], single[1][0])
    assert torch.equal(frames[3:6], single[3][0])
    assert torch.equal(frames[6:9], single[5][0])
    assert torch.equal(target, single[5][1])
    assert consecutive.frame_interval_s == pytest.approx(0.02)
    assert strided.frame_interval_s == pytest.approx(0.04)
    assert strided.row_interval_s == pytest.approx(0.02)


def test_mixed_collection_rates_are_refused(data_root):
    with pytest.raises(ValueError, match="different control periods"):
        ds.RolloutDataset(tracks=("alpha", "slow"), k=2)
    with pytest.raises(ValueError, match="stride"):
        ds.RolloutDataset(tracks=("alpha",), k=2, stride=0)


# ------------------------------------------------------------------ CPU sims
RES = (64, 48)
DRIVE = (0.3, 0.6)


def _sim(pipeline, seed=0):
    torch.manual_seed(0)
    return Builder((pipeline >> VectorPolicy(keys=("state",)) >> PPO()).build(seed=seed)).sim()


def _feature_env(**kw):
    return FeatureEnvironment(backend="cpu", num_envs=2, tracks=("reinvent_base",),
                              random_start=False, **kw)


def _drive(sim, value=DRIVE):
    return sim.step(torch.tensor([value] * sim.num_envs))


def test_control_dt_sim_keeps_the_episode_cap_in_seconds_and_scales_telemetry(tmp_path):
    from deepracer_genesis.analysis.telemetry import episode_agg, load_telemetry
    from deepracer_genesis.experiment.evaluator import evaluate_policy

    sim = _sim(_feature_env(control_dt_s=0.07, episode_length_s=0.7))
    assert sim.dt == pytest.approx(0.07)
    assert sim.cfg["sim"]["decimation"] == 7
    assert sim.max_episode_length == math.ceil(0.7 / 0.07)
    assert sim._timing is None and sim.applied_actions is sim.actions

    def actor(td):
        td.set("action", torch.tensor([DRIVE] * sim.num_envs))
        return td

    path = tmp_path / "tel.parquet"
    metrics = evaluate_policy(sim, actor, steps=25, telemetry=str(path))
    assert metrics["episodes"] >= 2
    assert metrics["mean_episode_s"] == pytest.approx(0.7)
    df = load_telemetry(str(path))
    assert set(df["dt"].unique()) == {np.float32(0.07)}
    agg = episode_agg(df)
    assert agg["time_s"].iloc[0] == pytest.approx(0.7, abs=1e-6)


def test_timing_on_keeps_issued_and_applied_commands_apart():
    sim = _sim(_feature_env(control_dt_s=0.05)
               >> DomainRandomizationActions(latency_ms=(30.0, 30.0)))
    assert sim._timing is not None
    issued = torch.tensor([[0.4, 0.2], [-0.4, 0.8]])
    obs, _, _, _ = sim.step(issued)
    assert torch.equal(sim.actions, issued)            # the features read this
    assert torch.equal(sim.applied_actions, issued)    # 30 ms < 50 ms: landed
    assert torch.allclose(obs["state"][:, 6:8], issued)
    sim2 = _sim(_feature_env(control_dt_s=0.02)
                >> DomainRandomizationActions(latency_ms=(30.0, 30.0)))
    obs, _, _, _ = sim2.step(issued)
    assert torch.equal(sim2.actions, issued)
    assert torch.equal(sim2.applied_actions, torch.zeros(2, 2))   # still pending
    assert torch.allclose(obs["state"][:, 6:8], issued)           # issued, not applied
    sim2.step(issued)
    assert torch.equal(sim2.applied_actions, issued)              # lands step 2


def test_timing_on_leaves_spawns_identical_and_resets_only_respawned_envs():
    off = _sim(_feature_env())
    on = _sim(_feature_env() >> DomainRandomizationActions(**TIMING))
    assert torch.equal(off.base_pos, on.base_pos) and torch.equal(off.yaw, on.yaw)
    latency = on._timing.latency_s.clone()
    assert latency[0] != latency[1]                               # per-env draws
    on.step(torch.tensor([DRIVE] * 2))
    on.episode_length_buf[0] = on.max_episode_length - 1          # env 0 times out
    on.step(torch.tensor([DRIVE] * 2))
    assert on.step_info["time_out"].tolist() == [True, False]
    assert on._timing.latency_s[1] == latency[1]
    assert on._timing.pending()[0] == 0


def test_strided_stack_holds_every_sth_frame_and_reprimes_on_respawn(tmp_path):
    from deepracer_genesis.perception.features import CNNPerceptionFeatures

    k, s = 3, 3
    good = _tiny_cnn(tmp_path / "good.pt", k, (RES[1], RES[0]), 0.06)
    cam = CameraEnvironment(backend="cpu", resolution=RES, num_envs=2,
                            tracks=("reinvent_base",), random_start=False,
                            frame_stack=k, frame_interval_s=0.06,
                            feature_set=CNNPerceptionFeatures,
                            feature_params={"checkpoint": str(good)})
    sim = _sim(cam)
    assert sim.frame_stride == s and sim.frame_interval_s == pytest.approx(0.06)
    frames = [sim.obs_image_buf.clone()]
    for _ in range((k - 1) * s + 2):
        _drive(sim)
        frames.append(sim.obs_image_buf.clone())
    t = len(frames) - 1
    stack = sim.camera_stack
    for i in range(k):                      # slot i is (k-1-i)*s steps old
        assert torch.equal(stack[:, 3 * i:3 * i + 3], frames[t - (k - 1 - i) * s])
    # the dataset's stride-s stack over the per-step frames is the env's stack
    rows = torch.stack(frames)              # (T, N, 3, H, W): one row per step
    rebuilt = torch.cat([rows[t - (k - 1 - i) * s] for i in range(k)], dim=1)
    assert torch.equal(rebuilt, stack)
    sim.episode_length_buf[0] = sim.max_episode_length - 1
    obs = _drive(sim)[0]
    fresh = sim.obs_image_buf[0]
    for i in range(k):
        assert torch.equal(obs["camera"][0, 3 * i:3 * i + 3], fresh)
    assert not torch.equal(obs["camera"][1, 0:3], obs["camera"][1, 3:6])


def test_the_cnn_refuses_a_mismatched_frame_interval(tmp_path):
    from deepracer_genesis.perception.features import CNNPerceptionFeatures

    bad = _tiny_cnn(tmp_path / "bad.pt", 3, (RES[1], RES[0]), 0.02)
    cam = CameraEnvironment(backend="cpu", resolution=RES, num_envs=2,
                            tracks=("reinvent_base",), random_start=False,
                            frame_stack=3, frame_interval_s=0.06,
                            feature_set=CNNPerceptionFeatures,
                            feature_params={"checkpoint": str(bad)})
    with pytest.raises(ValueError, match="0.02 s apart"):
        _sim(cam)


_DIGEST = r"""
import hashlib, sys, warnings
warnings.simplefilter("ignore")
import torch
from deepracer_genesis.experiment import PPO, DomainRandomizationActions, FeatureEnvironment, VectorPolicy
from deepracer_genesis.experiment.builder import Builder
from deepracer_genesis.seeding import seed_everything
spec = (FeatureEnvironment(num_envs=4, backend="cpu", control_dt_s=0.05)
        >> VectorPolicy()
        >> DomainRandomizationActions(latency_ms=(30.0, 90.0), latency_jitter_ms=10.0,
                                      decision_drop=0.05)
        >> PPO()).build(seed=int(sys.argv[1]))
seed_everything(spec.seed)
sim = Builder(spec).sim()
torch.manual_seed(0)
h = hashlib.sha256()
for _ in range(40):
    obs, rew, done, _ = sim.step(torch.rand(4, 2) * 2 - 1)
    h.update(obs["state"].numpy().tobytes() + rew.numpy().tobytes() + done.numpy().tobytes())
    h.update(sim.applied_actions.numpy().tobytes())
print("DIGEST:" + h.hexdigest())
"""


def _digest(seed):
    proc = subprocess.run([sys.executable, "-c", _DIGEST, str(seed)],
                          capture_output=True, text=True, check=False,
                          env={"CUDA_VISIBLE_DEVICES": "", **_clean_env()})
    assert proc.returncode == 0, proc.stderr[-2000:]
    return next(ln for ln in proc.stdout.splitlines() if ln.startswith("DIGEST:"))


def _clean_env():
    import os
    return {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}


@pytest.mark.smoke
def test_knobs_on_are_deterministic_across_processes():
    assert _digest(0) == _digest(0)
    assert _digest(0) != _digest(1)


@pytest.mark.smoke
def test_rsl_smoke_with_the_timing_knobs_on(tmp_path):
    from deepracer_genesis.experiment import run

    record = run(FeatureEnvironment(tracks=("reinvent_base",), num_envs=4,
                                    backend="cpu", control_dt_s=0.07)
                 >> VectorPolicy()
                 >> DomainRandomizationActions(**TIMING)
                 >> PPO(horizon_s=2.0),
                 root=str(tmp_path), total_env_steps=96, seed=3,
                 group="timing", variant="smoke")
    assert all(math.isfinite(v) for v in record.metrics.values()
               if isinstance(v, float) and not math.isnan(v))
    assert record.spec["env"]["control_dt_s"] == 0.07
    assert record.spec["action_dr"]["latency_ms"] == [30.0, 90.0]


def test_spec_record_round_trips_the_timing_fields():
    from deepracer_genesis.deploy.onnx import _spec_from_record

    spec = (FeatureEnvironment(control_dt_s=0.07) >> VectorPolicy()
            >> DomainRandomizationActions(**TIMING) >> PPO(horizon_s=2.0)).build(seed=2)
    back = _spec_from_record({"spec": spec.to_dict(), "seed": 2, "group": None,
                              "variant": None})
    assert back.id() == spec.id()
    assert back.action_dr.latency_ms == (30.0, 90.0)
    assert dataclasses.asdict(back.env)["control_dt_s"] == 0.07
