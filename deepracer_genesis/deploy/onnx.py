"""Export trained policies to ONNX plus a JSON model card, rebuilt on CPU.

Runs without genesis (it shares clashing LLVM symbols with onnxruntime): run as
its own process, e.g. ``python -m deepracer_genesis.deploy.onnx <target>``.

The export contract (see dissertation docs/inference-node-plan.md, decisions
D3/D5/D6): opset 11 and a static batch of 1 because the deployment target is
OpenVINO 2021.1's ONNX importer on the physical car, which predates newer
opsets and dynamic shapes; one input named after the physical sensor
(``FRONT_FACING_CAMERA``) because the AWS device stack dispatches sensor data
to model inputs by name substring; output ``action`` is the raw Gaussian mean,
UNBOUNDED — consumers must clip to [-1, 1] before rescaling, exactly as the
training env does (envs/base_env.py clips before mdp.map_action).
"""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
from typing import Optional

from ..experiment.spec import ExperimentSpec
from ..physics.limits import MAX_SPEED, MAX_STEERING_DEG, MIN_SPEED

def action_physical(spec: ExperimentSpec) -> dict:
    """Physical meaning of the normalized action channels for one spec.

    The speed ceiling follows the env's action cap when set, not the physics
    constant, so the car rescales into the range the policy actually trained on.

    Args:
        spec: The experiment the exported model was trained from.

    Returns:
        Per-channel ``low``/``high``/``unit`` ranges.
    """
    high = MAX_SPEED if spec.env.max_speed is None else spec.env.max_speed
    return {
        "steering": {"low": -MAX_STEERING_DEG, "high": MAX_STEERING_DEG,
                     "unit": "deg"},
        "speed": {"low": MIN_SPEED, "high": high, "unit": "m/s"},
    }

#: ONNX input/output names. The camera input is named after the sensor (not
#: "camera") so a name-substring dispatch like the AWS inference node's can
#: route EvoSensorMsg.images[0] to it; "action" (singular) matches the model
#: card and all prior deployment docs.
CAMERA_INPUT = "FRONT_FACING_CAMERA"
ACTION_OUTPUT = "action"

#: number of continuous action channels [steer, speed] (envs/base_env.py).
NUM_ACTIONS = 2


def state_dim(spec: ExperimentSpec) -> int:
    """Width of the state vector (delegates to the spec's feature set).

    NOTE: importing ``envs.features`` triggers ``envs/__init__`` which loads
    genesis — never call this on the camera-only export path (the guard in
    ``export_policy`` would already have passed, and onnxruntime would then
    crash the process).
    """
    from ..envs.features import feature_dim
    return feature_dim(spec.env.feature_set, lookahead_k=spec.env.lookahead_k,
                       params=spec.env.feature_params)


def state_layout(spec: ExperimentSpec) -> str:
    """Channel-by-channel description of the state vector."""
    from ..envs.features import feature_layout
    return feature_layout(spec.env.feature_set,
                          lookahead_k=spec.env.lookahead_k,
                          params=spec.env.feature_params)


def model_metadata(spec: ExperimentSpec) -> dict:
    """AWS-device-compatible ``model_metadata.json`` content for the spec.

    The device webserver/ctrl/navigation nodes parse this file to drive the
    load flow; the continuous ranges below are what the stock navigation node
    rescales the model's [-1, 1] outputs into, so they MUST equal the training
    env's action caps or the car drives with silently wrong units.
    """
    if spec.policy.actions is not None:
        raise NotImplementedError(
            "discrete action-space metadata not implemented: the rsl-rl "
            "backend cannot train discrete policies (rsl_supported), so an "
            "exported discrete model cannot exist yet")
    speed = action_physical(spec)["speed"]
    return {
        "action_space": {
            "steering_angle": {"high": MAX_STEERING_DEG, "low": -MAX_STEERING_DEG},
            "speed": {"high": speed["high"], "low": speed["low"]},
        },
        "action_space_type": "continuous",
        "sensor": [CAMERA_INPUT],
        "neural_network": "DEEP_CONVOLUTIONAL_NETWORK_SHALLOW",
        "training_algorithm": "clipped_ppo",
        "version": "5",
    }


def _rebuild_actor(spec: ExperimentSpec, ckpt_payload: dict):
    """Rebuild the trained actor on CPU and wrap it for single-input export.

    Uses rsl-rl's own ``CNNModel`` with the SAME cfg mapping training uses
    (``spec_to_train_cfg``), so exporter and trainer cannot disagree on
    architecture; then loads ``actor_state_dict`` strictly so any mismatch
    fails here, not on the car.

    Returns:
        ``(export_actor, reference_model)`` — the export wrapper (camera tensor
        in, action out) and the full rsl-rl model used as the parity reference.
    """
    import copy

    import torch
    from rsl_rl.models import CNNModel
    from tensordict import TensorDict
    from torch import nn

    from ..experiment.rsl_backend import spec_to_train_cfg

    if spec.env.modality != "camera" or tuple(spec.policy.actor_keys) != ("camera",):
        raise NotImplementedError(
            f"export supports camera-only actors (actor_keys=('camera',)); got "
            f"modality={spec.env.modality!r}, actor_keys={spec.policy.actor_keys!r}. "
            "Feature policies bake an EmpiricalNormalization whose export is "
            "not wired up yet.")

    train_cfg = spec_to_train_cfg(spec)
    actor_cfg = dict(train_cfg["actor"])
    class_name = actor_cfg.pop("class_name")
    if class_name != "CNNModel":
        raise NotImplementedError(f"unexpected actor class {class_name!r}")

    w, h = spec.env.resolution
    channels = 3 * getattr(spec.env, "frame_stack", 1)
    # Only the actor's obs groups are read (rsl-rl _get_obs_dim indexes
    # obs_groups["actor"]), so no "state" entry: computing its width would
    # import envs.features -> envs/__init__ -> genesis, poisoning this
    # deliberately genesis-free process.
    dummy_obs = TensorDict({"camera": torch.zeros(1, channels, h, w)},
                           batch_size=[1])
    model = CNNModel(dummy_obs, train_cfg["obs_groups"], "actor", NUM_ACTIONS,
                     **actor_cfg)
    model.load_state_dict(ckpt_payload["actor_state_dict"], strict=True)
    model.eval()

    class _ExportActor(nn.Module):
        """camera (1,3,H,W) float32 in [0,1] -> action (1,2), no sampling."""

        def __init__(self, m: CNNModel):
            super().__init__()
            self.cnn = copy.deepcopy(m.cnns["camera"])
            self.mlp = copy.deepcopy(m.mlp)
            # GaussianDistribution's deterministic output is the identity on
            # the MLP output (the mean); kept explicit so a future squashing
            # distribution exports correctly instead of silently diverging.
            self.head = (m.distribution.as_deterministic_output_module()
                         if m.distribution is not None else nn.Identity())

        def forward(self, camera):
            return self.head(self.mlp(self.cnn(camera)))

    actor = _ExportActor(model).eval()

    # Wiring self-check: the wrapper must reproduce the full model's
    # deterministic forward bit-for-bit before we trust the ONNX of it.
    with torch.no_grad():
        cam = torch.rand(1, channels, h, w)
        td = TensorDict({"camera": cam}, batch_size=[1])
        ref = model(td)
        got = actor(cam)
        if not torch.equal(ref, got):
            raise RuntimeError(
                f"export wrapper diverges from CNNModel forward "
                f"(max abs diff {(ref - got).abs().max().item():.3e})")
    return actor, model


def export_policy(target, *, root: str = "runs", ckpt: Optional[str] = None,
                  out: Optional[str] = None, opset: int = 11,
                  bundle_name: Optional[str] = None, **overrides) -> str:
    """Export ``target``'s trained actor to ONNX + model card + car bundle.

    Args:
        target: Any experiment handle accepted by ``experiment.run.build``.
        root: Runs directory the run dir resolves under.
        ckpt: Checkpoint path; defaults to ``model.pt`` in the run dir.
        out: Output directory; defaults to ``<run_dir>/export``.
        opset: ONNX opset. Default 11 — the newest the car's OpenVINO 2021.1
            ONNX importer is known to accept (proven by the dr-gym gates).
        bundle_name: Name for the car bundle tar.gz; defaults to the variant
            or "agent".
        **overrides: Keyword overrides forwarded to ``build(target)``.

    Returns:
        The export directory (contains ``policy.onnx``, ``model_card.json``,
        ``model_metadata.json`` and ``<bundle_name>.tar.gz``).
    """
    import sys
    if "genesis" in sys.modules:
        raise RuntimeError(
            "export_policy must run in a process that has NOT imported "
            "genesis: genesis and onnxruntime bundle clashing LLVM symbols "
            "and crash together. Run the export as its own step/script.")

    import torch

    from ..experiment.run import build

    spec: ExperimentSpec = build(target, **overrides)
    run_dir = spec.run_dir(root)
    ckpt = ckpt or os.path.join(run_dir, "model.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"no checkpoint at {ckpt} — train first")
    out = out or os.path.join(run_dir, "export")
    os.makedirs(out, exist_ok=True)

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "actor_state_dict" not in payload:
        raise KeyError(
            f"checkpoint {ckpt} has keys {sorted(payload)} — expected an "
            "rsl-rl OnPolicyRunner save with 'actor_state_dict'")
    actor, _model = _rebuild_actor(spec, payload)

    w, h = spec.env.resolution
    frame_stack = getattr(spec.env, "frame_stack", 1)
    dummy = torch.zeros(1, 3 * frame_stack, h, w)

    onnx_path = os.path.join(out, "policy.onnx")
    # Static batch 1 on purpose: the car infers one frame at a time and
    # OpenVINO 2021.1 handles static shapes most reliably; parity tests loop
    # per-sample instead of batching.
    torch.onnx.export(
        actor, (dummy,), onnx_path, opset_version=opset,
        input_names=[CAMERA_INPUT], output_names=[ACTION_OUTPUT],
        dynamo=False,
    )

    verified = _verify_onnx(actor, onnx_path, dummy)

    card = {
        "policy": {
            "file": "policy.onnx",
            "sha256": hashlib.sha256(open(onnx_path, "rb").read()).hexdigest(),
            "opset": opset,
            "batch": 1,
            "verified_against_torch": verified,
        },
        "action_space": {
            "type": "continuous",
            "output": f"{ACTION_OUTPUT} (1, {NUM_ACTIONS}) = [steer, speed], raw "
                      "Gaussian mean, UNBOUNDED — clip to [-1, 1] before use "
                      "(training env clips before mapping)",
            "normalized_to_physical": action_physical(spec),
        },
        "observations": {
            CAMERA_INPUT: {
                "shape": [1, 3 * frame_stack, h, w],
                "dtype": "float32",
                "range": "[0, 1] (= uint8 RGB / 255, no mean/std)",
                "layout": "NCHW, RGB",
                # The stack contract the node must reproduce exactly:
                "frame_stack": frame_stack,
                "stack_order": "oldest_first (newest frame = last 3 channels)",
                "stack_priming": "repeat_first_frame (never zeros)",
                "fov_deg": spec.env.fov,
                "camera": "front RGB, native 160x120 in sim (no train-time resize)",
            },
        },
        "training": {
            "spec": spec.to_dict(),
            "spec_id": spec.id(),
            "checkpoint": os.path.abspath(ckpt),
            "metrics": _load_metrics(os.path.dirname(ckpt)) or _load_metrics(run_dir),
        },
    }
    with open(os.path.join(out, "model_card.json"), "w") as f:
        json.dump(card, f, indent=2)

    meta = model_metadata(spec)
    with open(os.path.join(out, "model_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    bundle = _write_bundle(out, bundle_name or spec.variant or "agent")
    print(f"[export] policy.onnx + model_card.json + {os.path.basename(bundle)} -> {out}"
          + ("" if verified else "  (onnxruntime missing: NOT verified)"))
    return out


def _verify_onnx(actor, onnx_path: str, dummy) -> bool:
    """Check the ONNX graph reproduces the torch actor on random inputs.

    Returns False (with a loud warning) only when onnxruntime is missing;
    a numeric mismatch raises — an unverified export must never reach the car
    silently.
    """
    import torch
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("[export] WARNING: onnxruntime not installed — graph NOT "
              "verified against torch. Install the 'export' extra.")
        return False
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    for _ in range(8):
        rand = torch.rand_like(dummy)
        (onnx_out,) = sess.run(None, {CAMERA_INPUT: rand.numpy()})
        with torch.no_grad():
            torch_out = actor(rand).numpy()
        # 1e-5 is comfortably above fp32 kernel-order noise for this graph
        # size and far below the 1e-4 workstation parity gate.
        if not np.allclose(onnx_out, torch_out, atol=1e-5):
            raise RuntimeError(
                f"ONNX/torch mismatch: max abs diff "
                f"{np.abs(onnx_out - torch_out).max():.3e}")
    return True


def _write_bundle(out_dir: str, name: str) -> str:
    """Pack the AWS-style car bundle: metadata + model at the archive root.

    Layout mirrors what the device console extracts into
    /opt/aws/deepracer/artifacts/<name>/ — flat, with model_metadata.json next
    to the model file, because both the (patched) model optimizer and the
    navigation node resolve siblings of the model artifact path.
    """
    bundle_path = os.path.join(out_dir, f"{name}.tar.gz")
    with tarfile.open(bundle_path, "w:gz") as tar:
        tar.add(os.path.join(out_dir, "model_metadata.json"),
                arcname="model_metadata.json")
        tar.add(os.path.join(out_dir, "policy.onnx"), arcname="model.onnx")
        tar.add(os.path.join(out_dir, "model_card.json"),
                arcname="model_card.json")
    return bundle_path


def _load_metrics(run_dir: str) -> dict:
    p = os.path.join(run_dir, "eval_record.json")
    if os.path.exists(p):
        return json.load(open(p)).get("metrics", {})
    return {}


def main(argv=None) -> None:
    """CLI: ``python -m deepracer_genesis.deploy.onnx <target> [options]``."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=export_policy.__doc__)
    parser.add_argument("target", help="experiment as module:ClassName")
    parser.add_argument("--root", default="runs")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--opset", type=int, default=11)
    parser.add_argument("--bundle-name", default=None)
    args = parser.parse_args(argv)

    # build() takes classes, not names — resolve the CLI's module:ClassName.
    # Targets like examples.camera live under the project root (= cwd).
    import importlib
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    mod_name, cls_name = args.target.split(":")
    target = getattr(importlib.import_module(mod_name), cls_name)
    export_policy(target, root=args.root, ckpt=args.ckpt, out=args.out,
                  opset=args.opset, bundle_name=args.bundle_name)


if __name__ == "__main__":
    main()
