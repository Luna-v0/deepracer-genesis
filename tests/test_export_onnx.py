"""ONNX export contract tests (deploy/onnx.py).

The export runs in a SUBPROCESS: genesis and onnxruntime crash when imported
together (clashing LLVM symbols), and the pytest process may have genesis
loaded from other tests. The child does everything that touches onnx/ort and
prints a JSON report; the parent only asserts on it.
"""

import json
import os
import subprocess
import sys

import pytest

_DRIVER = r"""
import json, os, sys

import numpy as np
import torch
from tensordict import TensorDict

from deepracer_genesis.deploy.onnx import (
    ACTION_OUTPUT, CAMERA_INPUT, NUM_ACTIONS, export_policy,
)
from deepracer_genesis.experiment import AsymmetricCameraPolicy, CameraEnvironment
from deepracer_genesis.experiment.run import build
from deepracer_genesis.experiment.rsl_backend import spec_to_train_cfg
from rsl_rl.models import CNNModel

out = sys.argv[1]
frame_stack = int(sys.argv[2])
env_kwargs = json.loads(sys.argv[3])        # extra CameraEnvironment kwargs

pipe = (CameraEnvironment(render="madrona", resolution=(160, 120),
                          frame_stack=frame_stack, **env_kwargs)
        >> AsymmetricCameraPolicy(actor_keys=("camera",),
                                  critic_keys=("camera", "state")))
spec = build(pipe)

# Fabricate an OnPolicyRunner-style checkpoint with random (but trained-shaped)
# actor weights: the export contract is architecture + I/O, not reward.
cfg = spec_to_train_cfg(spec)
actor_cfg = dict(cfg["actor"])
actor_cfg.pop("class_name")
w, h = spec.env.resolution
channels = 3 * frame_stack
obs = TensorDict({"camera": torch.zeros(1, channels, h, w)}, batch_size=[1])
torch.manual_seed(0)
model = CNNModel(obs, cfg["obs_groups"], "actor", NUM_ACTIONS, **actor_cfg)
ckpt = os.path.join(out, "model.pt")
torch.save({"actor_state_dict": model.state_dict(), "critic_state_dict": {},
            "optimizer_state_dict": {}, "iter": 0, "infos": None}, ckpt)

export_dir = export_policy(pipe, ckpt=ckpt, out=out, bundle_name="testbundle")

import onnx
import onnxruntime as ort

onnx_path = os.path.join(export_dir, "policy.onnx")
graph = onnx.load(onnx_path)
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
(inp,) = sess.get_inputs()
(outp,) = sess.get_outputs()

# Independent parity check on fresh inputs (the exporter verified already;
# this catches the exporter lying about verification).
model.eval()
max_diff = 0.0
for i in range(4):
    cam = torch.rand(1, channels, h, w)
    td = TensorDict({"camera": cam}, batch_size=[1])
    with torch.no_grad():
        ref = model(td).numpy()
    (got,) = sess.run(None, {CAMERA_INPUT: cam.numpy()})
    max_diff = max(max_diff, float(np.abs(ref - got).max()))

card = json.load(open(os.path.join(export_dir, "model_card.json")))
meta = json.load(open(os.path.join(export_dir, "model_metadata.json")))

print("REPORT:" + json.dumps({
    "input_name": inp.name, "input_shape": inp.shape, "input_type": inp.type,
    "output_name": outp.name, "output_shape": outp.shape,
    "opset": max(op.version for op in graph.opset_import if op.domain in ("", "ai.onnx")),
    "verified": card["policy"]["verified_against_torch"],
    "max_diff": max_diff,
    "metadata": meta,
    "card_camera": card["observations"]["FRONT_FACING_CAMERA"],
    "card_action": card["action_space"]["normalized_to_physical"],
    "bundle_exists": os.path.exists(os.path.join(export_dir, "testbundle.tar.gz")),
}))
"""


_FEATURE_DRIVER = r"""
import json, os, sys

import numpy as np
import torch
from tensordict import TensorDict

from deepracer_genesis.deploy.onnx import (
    ACTION_OUTPUT, NUM_ACTIONS, STATE_INPUT, export_policy, state_dim,
)
from deepracer_genesis.experiment import FeatureEnvironment, VectorPolicy
from deepracer_genesis.experiment.run import build
from deepracer_genesis.experiment.rsl_backend import spec_to_train_cfg
from rsl_rl.models import MLPModel

out = sys.argv[1]
pipe = FeatureEnvironment() >> VectorPolicy()
spec = build(pipe)

cfg = spec_to_train_cfg(spec)
actor_cfg = dict(cfg["actor"])
actor_cfg.pop("class_name")
dim = state_dim(spec)
obs = TensorDict({"state": torch.zeros(1, dim)}, batch_size=[1])
torch.manual_seed(0)
model = MLPModel(obs, cfg["obs_groups"], "actor", NUM_ACTIONS, **actor_cfg)
# Give the normalizer NON-TRIVIAL stats: parity below then proves the
# EmpiricalNormalization really is baked into the exported graph (P6).
with torch.no_grad():
    model.obs_normalizer._mean.add_(torch.randn(1, dim) * 0.5)
    model.obs_normalizer._std.copy_(torch.rand(1, dim) + 0.5)
ckpt = os.path.join(out, "model.pt")
torch.save({"actor_state_dict": model.state_dict(), "critic_state_dict": {},
            "optimizer_state_dict": {}, "iter": 0, "infos": None}, ckpt)

export_dir = export_policy(pipe, ckpt=ckpt, out=out, bundle_name="featbundle")

import onnxruntime as ort

sess = ort.InferenceSession(os.path.join(export_dir, "policy.onnx"),
                            providers=["CPUExecutionProvider"])
(inp,) = sess.get_inputs()
(outp,) = sess.get_outputs()

model.eval()
max_diff = 0.0
for _ in range(4):
    state = torch.randn(1, dim) * 3.0     # far from 0-mean: exercises the norm
    with torch.no_grad():
        ref = model(TensorDict({"state": state}, batch_size=[1])).numpy()
    (got,) = sess.run(None, {STATE_INPUT: state.numpy()})
    max_diff = max(max_diff, float(np.abs(ref - got).max()))

card = json.load(open(os.path.join(export_dir, "model_card.json")))
meta = json.load(open(os.path.join(export_dir, "model_metadata.json")))
print("REPORT:" + json.dumps({
    "input_name": inp.name, "input_shape": inp.shape,
    "output_name": outp.name, "output_shape": outp.shape,
    "verified": card["policy"]["verified_against_torch"],
    "max_diff": max_diff,
    "card_state": card["observations"]["STATE"],
    "sensor": meta["sensor"],
    "bundle_exists": os.path.exists(os.path.join(export_dir, "featbundle.tar.gz")),
}))
"""


def _run_driver(tmp_path_factory, frame_stack, env_kwargs=None):
    out = tmp_path_factory.mktemp(f"export_k{frame_stack}")
    argv = [sys.executable, "-c", _DRIVER, str(out), str(frame_stack),
            json.dumps(env_kwargs or {})]
    proc = subprocess.run(
        argv, capture_output=True, text=True, check=False,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert proc.returncode == 0, f"export driver failed:\n{proc.stdout}\n{proc.stderr}"
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("REPORT:"))
    return json.loads(line[len("REPORT:"):])


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    return _run_driver(tmp_path_factory, frame_stack=1)


@pytest.fixture(scope="module")
def report_stacked(tmp_path_factory):
    return _run_driver(tmp_path_factory, frame_stack=4)


def test_model_io_contract(report):
    # The car node dispatches sensor data by input NAME and the docs pin the
    # output name — both are contract, not implementation detail.
    assert report["input_name"] == "FRONT_FACING_CAMERA"
    assert report["input_shape"] == [1, 3, 120, 160]
    assert "float" in report["input_type"]
    assert report["output_name"] == "action"
    assert report["output_shape"] == [1, 2]


def test_stacked_model_io_and_card(report_stacked):
    # frame_stack=4 => 12 input channels, and the model card must state the
    # stack contract the node reproduces (order + priming).
    assert report_stacked["input_shape"] == [1, 12, 120, 160]
    assert report_stacked["verified"] is True
    assert report_stacked["max_diff"] < 1e-5
    cam = report_stacked["card_camera"]
    assert cam["frame_stack"] == 4
    assert cam["stack_order"].startswith("oldest_first")
    assert cam["stack_priming"].startswith("repeat_first")
    # timing contract (ADR 0005) at the 50 Hz default: one decision per frame
    assert cam["control_dt_s"] == 0.02
    assert cam["frame_interval_s"] == 0.02
    assert cam["frame_stride"] == 1
    assert cam["stack_source"] == "policy_decisions"


@pytest.fixture(scope="module")
def report_timed(tmp_path_factory):
    return _run_driver(tmp_path_factory, frame_stack=4,
                       env_kwargs={"control_dt_s": 0.07, "frame_interval_s": 0.14})


def test_model_card_states_the_trained_control_period_and_frame_interval(report_timed):
    cam = report_timed["card_camera"]
    assert report_timed["input_shape"] == [1, 12, 120, 160]
    assert cam["control_dt_s"] == 0.07
    assert cam["frame_interval_s"] == 0.14
    assert cam["frame_stride"] == 2
    assert cam["stack_source"] == "policy_decisions"
    assert "frame_stride-th" in cam["stack_source_note"]


def test_opset_is_car_compatible(report):
    # OpenVINO 2021.1 on the car predates newer opsets; 11 is the proven ceiling.
    assert report["opset"] <= 11


# ------------------------------------------------------- feature policies (P6)
@pytest.fixture(scope="module")
def feature_report(tmp_path_factory):
    out = tmp_path_factory.mktemp("export_feature")
    proc = subprocess.run(
        [sys.executable, "-c", _FEATURE_DRIVER, str(out)],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert proc.returncode == 0, f"feature driver failed:\n{proc.stdout}\n{proc.stderr}"
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("REPORT:"))
    return json.loads(line[len("REPORT:"):])


def test_feature_model_io_contract(feature_report):
    assert feature_report["input_name"] == "STATE"
    assert feature_report["input_shape"] == [1, 28]     # 8 + 2*lookahead_k
    assert feature_report["output_name"] == "action"
    assert feature_report["output_shape"] == [1, 2]
    assert feature_report["sensor"] == ["STATE"]
    assert feature_report["bundle_exists"]


def test_feature_export_bakes_the_normalization(feature_report):
    # the driver perturbed the normalizer's mean/std and fed inputs far from
    # zero-mean; parity within 1e-5 proves the normalization is in the graph
    assert feature_report["verified"] is True
    assert feature_report["max_diff"] < 1e-5
    card = feature_report["card_state"]
    assert "BAKED" in card["range"]
    assert card["layout"].startswith("v_forward")
    assert card["control_dt_s"] == 0.02           # ADR 0005: the trained period


def test_export_guards_name_the_unsupported_actor_shapes():
    from deepracer_genesis.deploy.onnx import _rebuild_actor
    from deepracer_genesis.experiment import FeatureEnvironment, VectorPolicy
    from deepracer_genesis.experiment.run import build

    routed = build(FeatureEnvironment() >> VectorPolicy())
    import dataclasses
    routed = dataclasses.replace(
        routed, policy=dataclasses.replace(routed.policy,
                                           actor_keys=("obs_actor",)))
    with pytest.raises(NotImplementedError, match="routed"):
        _rebuild_actor(routed, {})

    rnn = build(FeatureEnvironment()
                >> VectorPolicy(mlp={"rnn": {"type": "lstm"}}))
    with pytest.raises(NotImplementedError, match="recurrent"):
        _rebuild_actor(rnn, {"actor_state_dict": {}})


def test_onnx_matches_torch(report):
    assert report["verified"] is True
    assert report["max_diff"] < 1e-5


def test_model_metadata_is_aws_continuous(report):
    meta = report["metadata"]
    # Exactly the keys/values the stock navigation + webserver nodes parse.
    assert meta["action_space_type"] == "continuous"
    assert meta["action_space"]["steering_angle"] == {"high": 30.0, "low": -30.0}
    assert meta["action_space"]["speed"] == {"high": 4.0, "low": 0.1}
    assert meta["sensor"] == ["FRONT_FACING_CAMERA"]
    assert meta["training_algorithm"] == "clipped_ppo"


def test_bundle_written(report):
    assert report["bundle_exists"]


# --- the exported speed ceiling must follow the env's action cap -------------
# PR #5 added EnvSpec.max_speed but left model_metadata on the physics constant,
# so a policy trained at max_speed=2.0 exported metadata claiming 4.0 and the
# stock navigation node would rescale [-1, 1] into twice the trained top speed.

def _camera_spec(**env_kwargs):
    """Build the driver's camera spec (frame stack aside) with extra env kwargs."""
    from deepracer_genesis.experiment import AsymmetricCameraPolicy, CameraEnvironment

    return (CameraEnvironment(render="madrona", resolution=(160, 120),
                              **env_kwargs)
            >> AsymmetricCameraPolicy(actor_keys=("camera",),
                                      critic_keys=("camera", "state"))).build()


def test_metadata_speed_ceiling_follows_the_env_action_cap():
    from deepracer_genesis.deploy.onnx import model_metadata

    meta = model_metadata(_camera_spec(max_speed=2.0))
    assert meta["action_space"]["speed"] == {"high": 2.0, "low": 0.1}


def test_metadata_speed_ceiling_defaults_to_the_physics_limit():
    from deepracer_genesis.deploy.onnx import model_metadata
    from deepracer_genesis.physics.limits import MAX_SPEED

    meta = model_metadata(_camera_spec())
    assert meta["action_space"]["speed"]["high"] == MAX_SPEED


def test_action_physical_tracks_the_same_cap():
    """The model card's normalized->physical table must not disagree with it."""
    from deepracer_genesis.deploy.onnx import action_physical, model_metadata

    spec = _camera_spec(max_speed=1.5)
    assert (action_physical(spec)["speed"]["high"]
            == model_metadata(spec)["action_space"]["speed"]["high"] == 1.5)


# --- the written caps must be the caps the training sim was built with ------
# The exporter once hard-coded MIN_SPEED/MAX_STEERING_DEG while Builder set the
# sim's caps on its own, so a knob threaded into one side only (PR #5's
# max_speed) shipped wrong units. Both now read spec.action_caps; this compares
# the WRITTEN files against Builder.sim_cfg, so a future min_speed knob is
# covered by adding a case with it in env_kwargs.

_CAPPED = {"max_speed": 2.0}


@pytest.fixture(scope="module")
def report_capped(tmp_path_factory):
    return _run_driver(tmp_path_factory, frame_stack=1, env_kwargs=_CAPPED)


@pytest.mark.parametrize(("fixture", "env_kwargs", "top_speed"),
                         [("report", {}, 4.0), ("report_capped", _CAPPED, 2.0)],
                         ids=["default", "max_speed"])
def test_exported_card_and_metadata_carry_the_training_sim_caps(
        request, fixture, env_kwargs, top_speed):
    from deepracer_genesis.experiment.builder import Builder

    rep = request.getfixturevalue(fixture)
    act = Builder(_camera_spec(**env_kwargs)).sim_cfg()["action"]
    steer = {"low": -act["max_steering_deg"], "high": act["max_steering_deg"]}
    speed = {"low": act["min_speed"], "high": act["max_speed"]}
    card = {ch: {"low": rng["low"], "high": rng["high"]}
            for ch, rng in rep["card_action"].items()}
    meta = rep["metadata"]["action_space"]
    assert card["steering"] == meta["steering_angle"] == steer
    assert card["speed"] == meta["speed"] == speed
    assert speed["high"] == top_speed    # the capped case is not vacuous
