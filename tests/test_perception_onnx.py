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

from deepracer_genesis.deploy.perception_onnx import (ACTOR_INPUT, ACTOR_OUTPUT,
                                                      PERCEPTION_INPUT,
                                                      PERCEPTION_OUTPUT,
                                                      export_perception_cnn)
from deepracer_genesis.perception.model import CHANNEL_NAMES, PerceptionCNN

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

print("REPORT:" + json.dumps({
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
