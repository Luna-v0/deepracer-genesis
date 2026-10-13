"""ONNX export of the two CNN-perception halves, checked against torch.

Exports run in a genesis-free subprocess (as `test_export_onnx.py` does) and
synthesize their own checkpoints, so nothing here needs a trained model.
"""

import json
import os
import subprocess
import sys

import pytest
import torch

from deepracer_genesis.deploy.perception_onnx import (
    ACTOR_INPUT,
    ACTOR_OUTPUT,
    PERCEPTION_INPUT,
    PERCEPTION_OUTPUT,
    export_perception_cnn,
    speed_cmd_max_speed_from_eval_record,
    speed_cmd_max_speed_from_run_dir,
)
from deepracer_genesis.perception.model import CHANNEL_NAMES, PerceptionCNN
from deepracer_genesis.physics.limits import MAX_SPEED, MAX_STEERING_DEG, MIN_SPEED

pytest.importorskip("onnxruntime", reason="needs the [export] extra")

_DRIVER = r"""
import json, os, sys
import numpy as np
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from deepracer_genesis.deploy.perception_onnx import (
    ACTOR_INPUT, PERCEPTION_INPUT, export_feature_actor, export_perception_cnn)
from deepracer_genesis.perception.model import PerceptionCNN

out = sys.argv[1]
torch.manual_seed(0)

per_ckpt = os.path.join(out, "perception.pt")
torch.save(PerceptionCNN().state_dict(), per_ckpt)

obs = TensorDict({ACTOR_INPUT: torch.zeros(1, 29)}, batch_size=[1])
groups = {"actor": [ACTOR_INPUT], "critic": [ACTOR_INPUT]}
cfg = dict(hidden_dims=[256, 128, 64], activation="elu", obs_normalization=True,
           distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0})
model = MLPModel(obs, groups, "actor", 2, **cfg).eval()
act_ckpt = os.path.join(out, "model.pt")
torch.save({"actor_state_dict": model.state_dict(), "critic_state_dict": {},
            "optimizer_state_dict": {}, "iter": 7, "infos": None}, act_ckpt)

per_path = export_perception_cnn(per_ckpt, os.path.join(out, "p"))
act_path = export_feature_actor(act_ckpt, os.path.join(out, "a"), max_speed=2.0)
# default speed_cmd_max_speed (no provenance confirmed): must fall back to
# the physics constant, not silently inherit max_speed_mps above
act_path_default_caps = export_feature_actor(
    act_ckpt, os.path.join(out, "a_default_caps"), max_speed=2.0)
# explicit, caller-confirmed speed_cmd_max_speed (post-373b5de provenance)
act_path_explicit_caps = export_feature_actor(
    act_ckpt, os.path.join(out, "a_explicit_caps"), max_speed=2.0,
    speed_cmd_max_speed=2.0)

# run_dir wiring (review finding 1): a run dir whose own eval_record.json
# names a resolved max_speed must be read automatically, with no explicit
# speed_cmd_max_speed from the caller
run_dir_with_record = os.path.join(out, "fake_run_with_record")
os.makedirs(run_dir_with_record, exist_ok=True)
with open(os.path.join(run_dir_with_record, "eval_record.json"), "w") as f:
    json.dump({"spec": {"env": {"modality": "feature", "max_speed": 2.5}}}, f)
act_path_from_run_dir = export_feature_actor(
    act_ckpt, os.path.join(out, "a_from_run_dir"), max_speed=2.0,
    run_dir=run_dir_with_record)

# an explicit override still wins over run_dir
act_path_override_wins = export_feature_actor(
    act_ckpt, os.path.join(out, "a_override_wins"), max_speed=2.0,
    speed_cmd_max_speed=9.9, run_dir=run_dir_with_record)

# a run_dir with no eval_record.json at all: falls back to the physics
# constant exactly like no run_dir
run_dir_without_record = os.path.join(out, "fake_run_without_record")
os.makedirs(run_dir_without_record, exist_ok=True)
act_path_run_dir_no_record = export_feature_actor(
    act_ckpt, os.path.join(out, "a_run_dir_no_record"), max_speed=2.0,
    run_dir=run_dir_without_record)

import onnxruntime as ort
per = ort.InferenceSession(per_path, providers=["CPUExecutionProvider"])
act = ort.InferenceSession(act_path, providers=["CPUExecutionProvider"])

# the normalizer must have travelled with the graph: a deliberately
# un-normalized vector has to give the same action as the torch model
raw = torch.randn(1, 29) * 3.0 + 1.0
with torch.no_grad():
    reference = model(TensorDict({ACTOR_INPUT: raw}, batch_size=[1]))
(got,) = act.run(None, {ACTOR_INPUT: raw.numpy()})
normalizer_ok = bool(np.abs(got - reference.numpy()).max() < 1e-4)

# and the two graphs must chain: perception's 7 are the actor's first 7
(seven,) = per.run(None, {PERCEPTION_INPUT: np.zeros((1, 12, 120, 160), np.float32)})
state = np.concatenate([seven, np.zeros((1, 22), np.float32)], 1).astype(np.float32)
(action,) = act.run(None, {ACTOR_INPUT: state})

bad = os.path.join(out, "bad.pt")
torch.save({"weights": {}}, bad)
try:
    export_feature_actor(bad, os.path.join(out, "bad"))
    rejects_bad_ckpt = False
except KeyError:
    rejects_bad_ckpt = True

# a searched (non-stock) architecture with its own input size must export
# from the arch the checkpoint carries — no shape arguments involved
from deepracer_genesis.perception.model import save_checkpoint
custom = PerceptionCNN(in_channels=6, n_targets=5, input_hw=(60, 80),
                       channels=(8, 16), kernels=(5, 3), strides=(4, 2),
                       head=32).eval()
custom_ckpt = os.path.join(out, "custom.pt")
save_checkpoint(custom, custom_ckpt)
custom_path = export_perception_cnn(custom_ckpt, os.path.join(out, "c"))
cust = ort.InferenceSession(custom_path, providers=["CPUExecutionProvider"])
sample = torch.rand(1, 6, 60, 80)
with torch.no_grad():
    cref = custom(sample).numpy()
(cgot,) = cust.run(None, {PERCEPTION_INPUT: sample.numpy()})
custom_parity = float(np.abs(cgot - cref).max())

print("REPORT:" + json.dumps({
    "custom_inputs": [(i.name, list(i.shape)) for i in cust.get_inputs()],
    "custom_parity": custom_parity,
    "custom_card_arch": json.load(
        open(os.path.join(out, "c", "perception_card.json")))["arch"],
    "per_inputs": [(i.name, list(i.shape)) for i in per.get_inputs()],
    "per_outputs": [(o.name, list(o.shape)) for o in per.get_outputs()],
    "act_inputs": [(i.name, list(i.shape)) for i in act.get_inputs()],
    "act_outputs": [(o.name, list(o.shape)) for o in act.get_outputs()],
    "normalizer_ok": normalizer_ok,
    "seven_shape": list(seven.shape),
    "action": action[0].tolist(),
    "rejects_bad_ckpt": rejects_bad_ckpt,
    "per_card": json.load(open(os.path.join(out, "p", "perception_card.json"))),
    "act_card": json.load(open(os.path.join(out, "a", "actor_card.json"))),
    "act_card_default_caps": json.load(
        open(os.path.join(out, "a_default_caps", "actor_card.json"))),
    "act_card_explicit_caps": json.load(
        open(os.path.join(out, "a_explicit_caps", "actor_card.json"))),
    "act_card_from_run_dir": json.load(
        open(os.path.join(out, "a_from_run_dir", "actor_card.json"))),
    "act_card_override_wins": json.load(
        open(os.path.join(out, "a_override_wins", "actor_card.json"))),
    "act_card_run_dir_no_record": json.load(
        open(os.path.join(out, "a_run_dir_no_record", "actor_card.json"))),
}))
"""


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    """Run both exports in a subprocess that never imports genesis."""
    out = tmp_path_factory.mktemp("perception_onnx")
    proc = subprocess.run(
        [sys.executable, "-c", _DRIVER, str(out)], capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert proc.returncode == 0, f"driver failed:\n{proc.stdout}\n{proc.stderr}"
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("REPORT:"))
    return json.loads(line[len("REPORT:"):])


def test_custom_arch_exports_from_its_own_payload(report):
    """A searched arch (6ch, 60x80, non-stock trunk) must export using the
    arch the checkpoint carries and match torch numerically."""
    assert report["custom_inputs"] == [[PERCEPTION_INPUT, [1, 6, 60, 80]]]
    assert report["custom_parity"] < 1e-5
    arch = report["custom_card_arch"]
    assert arch["channels"] == [8, 16] and arch["head"] == 32
    assert arch["input_hw"] == [60, 80]


def test_perception_graph_io(report):
    assert report["per_inputs"] == [[PERCEPTION_INPUT, [1, 12, 120, 160]]]
    assert report["per_outputs"] == [[PERCEPTION_OUTPUT, [1, len(CHANNEL_NAMES)]]]
    card = report["per_card"]
    assert card["kind"] == "perception_cnn"
    assert card["outputs"][PERCEPTION_OUTPUT]["channels"] == list(CHANNEL_NAMES)
    assert card["parameters"] == sum(p.numel() for p in PerceptionCNN().parameters())
    assert card["onnx_vs_torch_max_abs_diff"] < 1e-4


def test_actor_graph_io(report):
    assert report["act_inputs"] == [[ACTOR_INPUT, [1, 29]]]
    assert report["act_outputs"] == [[ACTOR_OUTPUT, [1, 2]]]
    card = report["act_card"]
    assert card["action_mapping"]["max_speed_mps"] == 2.0
    assert card["iteration"] == 7
    assert "UNBOUNDED" in card["outputs"][ACTOR_OUTPUT]["meaning"]
    assert card["onnx_vs_torch_max_abs_diff"] < 1e-4


def test_actor_card_carries_onboard_feature_caps(report):
    """New, additive card fields for the 22 on-board features' caps.

    max_steering_deg/speed_cmd_min_speed/speed_err_norm never vary by
    checkpoint provenance, so they are always the physics constants.
    speed_cmd_max_speed is the one caller-supplied, provenance-dependent
    value: unconfirmed (default) must fall back to the physics constant
    rather than silently inheriting action_mapping.max_speed_mps, while an
    explicit, caller-confirmed value is recorded as given."""
    default_caps = report["act_card_default_caps"]["onboard_feature_caps"]
    assert default_caps["max_steering_deg"] == MAX_STEERING_DEG
    assert default_caps["speed_cmd_min_speed_mps"] == MIN_SPEED
    assert default_caps["speed_err_norm_mps"] == MAX_SPEED
    assert default_caps["speed_cmd_max_speed_mps"] == MAX_SPEED   # fallback, NOT 2.0

    explicit_caps = report["act_card_explicit_caps"]["onboard_feature_caps"]
    assert explicit_caps["speed_cmd_max_speed_mps"] == 2.0
    assert explicit_caps["speed_err_norm_mps"] == MAX_SPEED   # unaffected


def test_actor_card_reads_speed_cmd_max_speed_from_run_dir(report):
    """Review finding 1: with no explicit speed_cmd_max_speed, the exporter
    reads the checkpoint's own run_dir/eval_record.json instead of falling
    back to the physics constant (or inheriting action_mapping's max_speed)."""
    from_run_dir = report["act_card_from_run_dir"]["onboard_feature_caps"]
    assert from_run_dir["speed_cmd_max_speed_mps"] == 2.5

    # an explicit caller override still takes priority over run_dir
    override_wins = report["act_card_override_wins"]["onboard_feature_caps"]
    assert override_wins["speed_cmd_max_speed_mps"] == 9.9

    # a run_dir with no eval_record.json at all: physics-constant fallback,
    # exactly like passing no run_dir
    no_record = report["act_card_run_dir_no_record"]["onboard_feature_caps"]
    assert no_record["speed_cmd_max_speed_mps"] == MAX_SPEED


def test_actor_export_folds_the_normalizer_in(report):
    """Exported without it, raw features would silently give a wrong action."""
    assert report["normalizer_ok"]


def test_the_two_graphs_chain(report):
    """perception's 7 outputs are the first 7 of the actor's 29 inputs."""
    assert report["seven_shape"] == [1, 7]
    assert len(report["action"]) == 2
    assert all(v == v for v in report["action"])       # not NaN


def test_export_refuses_a_checkpoint_without_an_actor(report):
    assert report["rejects_bad_ckpt"]


def test_export_refuses_when_genesis_is_loaded(tmp_path, monkeypatch):
    """genesis + onnxruntime bundle clashing LLVM symbols and crash together."""
    ckpt = tmp_path / "perception.pt"
    torch.save(PerceptionCNN().state_dict(), ckpt)
    monkeypatch.setitem(sys.modules, "genesis", object())
    with pytest.raises(RuntimeError, match="NOT imported genesis"):
        export_perception_cnn(str(ckpt), str(tmp_path / "out"))


# ------------------------------------------- speed_cmd provenance (review finding 1)
# These are pure JSON-dict functions — no torch/onnx/genesis involved — so
# they run directly, no subprocess/genesis-free dance needed.

def _record(env):
    return {"spec": {"env": env}}


def test_speed_cmd_max_speed_pre_373b5de_key_absent():
    """No "max_speed" key at all: the dumped spec predates commit 373b5de
    (which is also the commit that added the field to EnvSpec at all), so
    speed_cmd always used the global physics constant regardless of any
    other value in this dict."""
    assert speed_cmd_max_speed_from_eval_record(
        _record({"modality": "feature"})) == MAX_SPEED


def test_speed_cmd_max_speed_post_373b5de_key_present_but_null():
    """Key present but null: the spec left max_speed at the physics
    default, resolved by action_caps() to the same constant — but via the
    post-373b5de resolved-cap code path, not the pre-373b5de hardcoding."""
    assert speed_cmd_max_speed_from_eval_record(
        _record({"modality": "feature", "max_speed": None})) == MAX_SPEED


def test_speed_cmd_max_speed_post_373b5de_key_set():
    """Key present with a real cap: speed_cmd followed it during training."""
    assert speed_cmd_max_speed_from_eval_record(
        _record({"modality": "feature", "max_speed": 2.0})) == 2.0


def test_speed_cmd_max_speed_prefers_the_stamped_block_when_present():
    """A record carrying the newer, explicit onboard_feature_caps (stamped
    by rsl_backend.py at train time) is read directly — no key-presence
    inference needed, and it wins even if spec.env disagreed."""
    record = _record({"modality": "feature", "max_speed": 2.0})
    record["onboard_feature_caps"] = {"speed_cmd_max_speed_mps": 3.5}
    assert speed_cmd_max_speed_from_eval_record(record) == 3.5


def test_speed_cmd_max_speed_from_the_real_pre_373b5de_run_on_disk():
    """The actual run dir the reviewer found on disk: no "max_speed" key at
    all in its eval_record.json, independently confirming (with no git
    history consulted here) that this run predates commit 373b5de. Reads
    the real file in place; never copied into the repo."""
    run_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "runs", "best_camera_completion", "best_camera_completion",
        "final_d1-0-75f935de3139")
    assert os.path.isfile(os.path.join(run_dir, "eval_record.json")), (
        "the real run dir this test reads is expected to still be on disk")
    assert speed_cmd_max_speed_from_run_dir(run_dir) == MAX_SPEED


def test_speed_cmd_max_speed_from_run_dir_without_a_record_returns_none():
    assert speed_cmd_max_speed_from_run_dir("/nonexistent/path/xyz") is None
