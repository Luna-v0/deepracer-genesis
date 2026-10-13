"""ONNX export for the two halves of the CNN-perception stack.

The perception CNN (camera to physical channels) and the feature actor (state
vector to action) export separately, so either can be run on its own.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Sequence

from ..physics.limits import MAX_SPEED, MAX_STEERING_DEG, MIN_SPEED

logger = logging.getLogger(__name__)

#: Named after the sensor so a name-substring dispatch (as the AWS inference
#: node does) can route the camera image to it.
PERCEPTION_INPUT = "FRONT_FACING_CAMERA"
PERCEPTION_OUTPUT = "perception"
ACTOR_INPUT = "state"
ACTOR_OUTPUT = "action"
NUM_ACTIONS = 2

#: Newest opset the car's OpenVINO 2021.1 ONNX importer is known to accept.
DEFAULT_OPSET = 11


def _require_genesis_free() -> None:
    """Refuse to export from a process that has imported genesis.

    Raises:
        RuntimeError: If ``genesis`` is already in ``sys.modules``.
    """
    if "genesis" in sys.modules:
        raise RuntimeError(
            "run the export in a process that has NOT imported genesis: "
            "genesis and onnxruntime bundle clashing LLVM symbols and crash "
            "together. Run the export as its own step/script.")


def speed_cmd_max_speed_from_eval_record(eval_record: dict) -> float:
    """Derive the checkpoint's true speed_cmd cap from its own eval_record.json.

    Review finding 1 (2026-10-13-review-speed-caps.md): commit 373b5de is
    *simultaneously* the commit that made ``envs/features.py``'s
    ``speed_cmd`` follow the run's resolved ``max_speed`` AND the commit that
    added the ``max_speed`` field to ``EnvSpec`` at all — before it, the field
    could not exist in a dumped spec, so its mere PRESENCE in
    ``eval_record["spec"]["env"]`` is already a 100%-reliable pre/post-373b5de
    signal, with no git history to consult:

    * Key absent -> the run predates 373b5de -> speed_cmd always used the
      global physics constant -> return it.
    * Key present (including an explicit ``null``) -> the run is at or after
      373b5de -> speed_cmd followed the resolved cap, which itself defaults
      to the physics constant when the spec value is ``None`` -> return
      ``env["max_speed"] or MAX_SPEED``.

    A record written by a run that *also* carries the newer, explicit
    ``onboard_feature_caps`` block (added in the same change as this
    function) is read from there instead — direct, no inference needed.

    Args:
        eval_record: A loaded ``eval_record.json`` dict (``EvalRecord.save``'s
            payload shape).

    Returns:
        The max_speed ``speed_cmd`` was actually computed with for this run.
    """
    record = eval_record or {}
    caps = record.get("onboard_feature_caps") or {}
    stamped = caps.get("speed_cmd_max_speed_mps")
    if stamped is not None:
        return stamped
    env = record.get("spec", {}).get("env") or {}
    return env.get("max_speed") or MAX_SPEED


def speed_cmd_max_speed_from_run_dir(run_dir: str) -> float | None:
    """`speed_cmd_max_speed_from_eval_record`, loading `<run_dir>/eval_record.json`.

    Args:
        run_dir: A run directory (typically the checkpoint's own parent dir).

    Returns:
        The derived value, or None if `run_dir` has no `eval_record.json`
        (e.g. a checkpoint saved outside the run-dir convention) — callers
        should treat None the same as "no record available".
    """
    path = os.path.join(run_dir, "eval_record.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return speed_cmd_max_speed_from_eval_record(json.load(f))


def _verify(onnx_path: str, sample, reference, name: str, tol: float = 1e-4) -> float:
    """Check the exported graph reproduces the torch module on one sample.

    Args:
        onnx_path: Path to the written graph.
        sample: The input tensor the reference was evaluated on.
        reference: The torch module's output for that input.
        name: The graph's input name.
        tol: Largest tolerated absolute difference.

    Returns:
        The observed maximum absolute difference.

    Raises:
        RuntimeError: If the graph diverges from torch beyond ``tol``.
    """
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    (got,) = session.run(None, {name: sample.numpy()})
    diff = float(np.abs(got - reference.numpy()).max())
    if diff > tol:
        raise RuntimeError(
            f"{onnx_path} diverges from torch by {diff:.3e} (> {tol:.0e})")
    return diff


def export_perception_cnn(checkpoint: str, out_dir: str, *,
                          frame_stack: int = 4,
                          input_hw: tuple[int, int] = (120, 160),
                          opset: int = DEFAULT_OPSET) -> str:
    """Export the frozen perception CNN to ONNX plus a JSON card.

    Args:
        checkpoint: Path to a ``PerceptionCNN`` state dict.
        out_dir: Directory the graph and card are written to.
        frame_stack: Camera frames stacked along the channel axis.
        input_hw: Frame height and width a BARE state dict was trained for;
            arch-carrying checkpoints record their own and ignore this.
        opset: ONNX opset to emit.

    Returns:
        Path to the written ``.onnx`` file.

    Raises:
        RuntimeError: If genesis is loaded, or the graph diverges from torch.
    """
    _require_genesis_free()
    import torch

    from deepracer_genesis.perception.model import CHANNEL_NAMES, SIGMA, load_checkpoint

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" not in state:
        # bare stock state dict: the head was sized for the input_hw ARGUMENT
        state = {"arch": {"in_channels": state["features.0.weight"].shape[1],
                          "n_targets": state["head.3.weight"].shape[0],
                          "input_hw": tuple(input_hw)},
                 "state_dict": state}
    net = load_checkpoint(state)
    in_channels = net.arch["in_channels"]
    n_targets = net.arch["n_targets"]
    input_hw = tuple(net.arch["input_hw"])   # arch-carrying payloads win

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "perception.onnx")
    h, w = input_hw
    dummy = torch.zeros(1, in_channels, h, w)
    torch.onnx.export(net, dummy, path, opset_version=opset,
                      input_names=[PERCEPTION_INPUT],
                      output_names=[PERCEPTION_OUTPUT],
                      dynamo=False)
    with torch.no_grad():
        reference = net(dummy)
    diff = _verify(path, dummy, reference, PERCEPTION_INPUT)

    card = {
        "kind": "perception_cnn",
        "source_checkpoint": os.path.basename(checkpoint),
        "opset": opset,
        "arch": {k: list(v) if isinstance(v, tuple) else v
                 for k, v in net.arch.items()},
        "parameters": sum(p.numel() for p in net.parameters()),
        "inputs": {PERCEPTION_INPUT: {
            "shape": [1, in_channels, h, w], "dtype": "float32",
            "range": "[0, 1]",
            "layout": f"NCHW; {in_channels // 3} stacked RGB frames, OLDEST "
                      "FIRST (newest = last 3 channels)",
            "priming": "a fresh episode repeats the first frame; zeros never "
                       "occur on a real camera",
            # seconds between stacked frames at training (ADR 0005); None
            # for checkpoints that predate the field
            "frame_interval_s": getattr(net, "frame_interval_s", None)}},
        "outputs": {PERCEPTION_OUTPUT: {
            "shape": [1, n_targets], "dtype": "float32",
            "channels": list(CHANNEL_NAMES[:n_targets]),
            "normalization": "lateral_offset, heading_err/pi, speed/4, "
                             "yaw_rate/5, sideslip_beta/0.5, curvature/2.5",
            "validation_rmse": list(SIGMA[:n_targets])}},
        "onnx_vs_torch_max_abs_diff": diff,
        "notes": "Predicts the channels of PerceptionFeatures a camera can "
                 "recover. Not a driving model: it emits no action.",
    }
    with open(os.path.join(out_dir, "perception_card.json"), "w") as f:
        json.dump(card, f, indent=2)
    logger.info("perception CNN -> %s (parity %.2e)", path, diff)
    return path


def export_feature_actor(checkpoint: str, out_dir: str, *,
                         obs_dim: int = 29,
                         hidden_dims: Sequence[int] = (256, 128, 64),
                         activation: str = "elu",
                         num_actions: int = NUM_ACTIONS,
                         max_speed: float | None = None,
                         speed_cmd_max_speed: float | None = None,
                         run_dir: str | None = None,
                         opset: int = DEFAULT_OPSET) -> str:
    """Export an MLP actor over the feature vector, normalizer folded in.

    The observation normalizer is part of the graph, so callers feed raw
    features exactly as the env produces them.

    Args:
        checkpoint: Path to an rsl-rl ``OnPolicyRunner`` save.
        out_dir: Directory the graph and card are written to.
        obs_dim: Width of the feature vector.
        hidden_dims: Actor hidden layer widths.
        activation: Actor activation name.
        num_actions: Action dimension.
        max_speed: The env's action cap in m/s, recorded in the card as
            ``action_mapping.max_speed_mps`` (the actor's own declared action
            range; unchanged, pre-existing field).
        speed_cmd_max_speed: The max_speed ``envs/features.py``'s
            ``PerceptionFeatures`` actually used to compute ``speed_cmd`` for
            the 22 on-board features THIS checkpoint was trained with.
            **This is not always the same as `max_speed` above.** Before
            commit 373b5de (2026-08-30), ``speed_cmd`` always used the global
            physics constant (4.0) no matter what any spec said; from that
            commit on, it follows the run's resolved ``max_speed``. Takes
            priority over `run_dir` below when both are given. Pass this
            explicitly only when you have independently confirmed the value
            (e.g. by hand, from the run's git history) — `run_dir` is the
            preferred, automatic source.
        run_dir: The checkpoint's run directory, if it has one (typically its
            parent dir). When `speed_cmd_max_speed` is not given, this
            function reads `<run_dir>/eval_record.json` (review finding 1,
            `speed_cmd_max_speed_from_run_dir`) and uses ITS answer — reliable
            because 373b5de is simultaneously the commit that made
            `speed_cmd` follow the resolved cap and the commit that added the
            `max_speed` field to `EnvSpec` at all, so the field's mere
            presence already tells the two code-version cases apart with no
            git history to consult, and a record written by this same change
            carries the answer directly in its own `onboard_feature_caps`
            block. If `run_dir` has no `eval_record.json` (or is None), the
            physics constant (today's exact default) is recorded, matching
            dr-onnx's current behavior exactly.
        opset: ONNX opset to emit.

    Returns:
        Path to the written ``.onnx`` file.

    Raises:
        KeyError: If the checkpoint has no ``actor_state_dict``.
        RuntimeError: If genesis is loaded, or the graph diverges from torch.
    """
    _require_genesis_free()
    if speed_cmd_max_speed is None and run_dir is not None:
        speed_cmd_max_speed = speed_cmd_max_speed_from_run_dir(run_dir)
    import copy

    import torch
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict
    from torch import nn

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "actor_state_dict" not in payload:
        raise KeyError(
            f"{checkpoint} has keys {sorted(payload)} — expected an rsl-rl "
            "OnPolicyRunner save with 'actor_state_dict'")

    obs = TensorDict({ACTOR_INPUT: torch.zeros(1, obs_dim)}, batch_size=[1])
    groups = {"actor": [ACTOR_INPUT], "critic": [ACTOR_INPUT]}
    model = MLPModel(obs, groups, "actor", num_actions,
                     hidden_dims=list(hidden_dims), activation=activation,
                     obs_normalization=True,
                     distribution_cfg={"class_name": "GaussianDistribution",
                                       "init_std": 1.0})
    model.load_state_dict(payload["actor_state_dict"], strict=True)
    model.eval()

    class _ExportActor(nn.Module):
        """state (1, obs_dim) in -> action (1, num_actions), no sampling."""

        def __init__(self, m: MLPModel) -> None:
            super().__init__()
            self.normalizer = copy.deepcopy(m.obs_normalizer)
            self.mlp = copy.deepcopy(m.mlp)
            self.head = (m.distribution.as_deterministic_output_module()
                         if m.distribution is not None else nn.Identity())

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            return self.head(self.mlp(self.normalizer(state)))

    actor = _ExportActor(model).eval()

    # the wrapper must reproduce the full model bit-for-bit before its ONNX is
    # trusted — the normalizer is the easy thing to fold in wrong
    sample = torch.rand(1, obs_dim)
    with torch.no_grad():
        reference = model(TensorDict({ACTOR_INPUT: sample}, batch_size=[1]))
        if not torch.equal(reference, actor(sample)):
            raise RuntimeError(
                "export wrapper diverges from MLPModel forward "
                f"(max abs diff {(reference - actor(sample)).abs().max():.3e})")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "actor.onnx")
    torch.onnx.export(actor, sample, path, opset_version=opset,
                      input_names=[ACTOR_INPUT], output_names=[ACTOR_OUTPUT],
                      dynamo=False)
    diff = _verify(path, sample, reference, ACTOR_INPUT)

    card = {
        "kind": "feature_actor",
        "source_checkpoint": os.path.basename(checkpoint),
        "opset": opset,
        "iteration": payload.get("iter"),
        "parameters": sum(p.numel() for p in actor.parameters()),
        "inputs": {ACTOR_INPUT: {
            "shape": [1, obs_dim], "dtype": "float32",
            "layout": "CNN targets[7]: lateral_offset, heading_err/pi, speed/4, "
                      "yaw_rate/5, sideslip_beta/0.5, curvature@[1.0, 3.0]m/2.5 "
                      "| policy-only: prev_action[2x2], speed_error[2], "
                      "steer_response_error[8], yaw_error[8]",
            "normalization": "folded into the graph; feed RAW features"}},
        "outputs": {ACTOR_OUTPUT: {
            "shape": [1, num_actions], "dtype": "float32",
            "meaning": "[steer, speed], raw Gaussian mean, UNBOUNDED — clip to "
                       "[-1, 1] before use (the training env clips before "
                       "mapping)"}},
        "action_mapping": {
            "steer": "steer_rad = a[0] * radians(max_steering_deg)",
            "speed": "speed_mps = min_speed + (a[1] + 1) * 0.5 * "
                     "(max_speed - min_speed)",
            "max_speed_mps": max_speed},
        # caps the 22 on-board features (dr_onnx/policy_features.py) were
        # computed with — separate from action_mapping above because they can
        # genuinely diverge from it (see export_feature_actor's docstring on
        # speed_cmd_max_speed). max_steering_deg, speed_cmd_min_speed and
        # speed_err_norm are provenance-independent (no spec knob has ever
        # changed them, pre- or post-373b5de) so they are always the physics
        # constants; speed_cmd_max_speed is the one caller-supplied,
        # provenance-dependent value, defaulting to the same physics constant
        # so an un-updated/naive export never claims a non-default cap.
        "onboard_feature_caps": {
            "max_steering_deg": MAX_STEERING_DEG,
            "speed_cmd_min_speed_mps": MIN_SPEED,
            "speed_cmd_max_speed_mps": (
                MAX_SPEED if speed_cmd_max_speed is None else speed_cmd_max_speed),
            "speed_err_norm_mps": MAX_SPEED,
        },
        "onnx_vs_torch_max_abs_diff": diff,
        "notes": "Only 7 of the input channels come from the camera (via "
                 "perception.onnx); the other 22 are computed onboard from the "
                 "action history and perception.onnx's speed and yaw_rate "
                 "outputs, so this graph is not a stock-AWS drop-in.",
    }
    with open(os.path.join(out_dir, "actor_card.json"), "w") as f:
        json.dump(card, f, indent=2)
    logger.info("feature actor -> %s (parity %.2e)", path, diff)
    return path
