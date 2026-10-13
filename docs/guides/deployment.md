# Deployment (ONNX)

Export a trained actor to ONNX (plus a `model_card.json`) so it can run on the
physical DeepRacer or any ONNX runtime. The exporter is
`deepracer_genesis/deploy/onnx.py`.

> Mental model in one sentence: `export_policy` rebuilds the actor on CPU from a
> checkpoint, traces it to ONNX with a single named input
> (`FRONT_FACING_CAMERA` for camera actors — the name the car's inference
> stack expects — or `STATE` for feature actors, with the
> `EmpiricalNormalization` baked into the graph), and writes the car bundle:
> `policy.onnx` + `model_metadata.json` + `model_card.json` + a
> `<name>.tar.gz` to upload.

!!! warning "Run the export in a genesis-free process"
    genesis and onnxruntime bundle clashing LLVM symbols and crash the
    process when both are loaded. `export_policy` refuses to run if genesis
    was already imported — call it from a fresh process (its own script or
    subprocess), never from a session that built a sim. The `best_camera`
    notebooks show the subprocess pattern.

---

## Exporting

```python
from deepracer_genesis.deploy.onnx import export_policy

# pass your Experiment subclass (uses the run dir's model.pt):
export_policy(FeatureBaseline)

# or an explicit chain + checkpoint:
export_policy(
    FeatureEnvironment(num_envs=64) >> VectorPolicy() >> PPO(),
    ckpt="runs/feature_ppo_abc123/model.pt",
    out="export/my_model")
```

The default `opset=11` is deliberate: it is the newest opset the physical
car's OpenVINO 2021.1 ONNX importer is known to accept. Raise it only for
non-car runtimes.

`export_policy(target, *, root, ckpt, out, opset, **overrides)`:

1. builds the spec, loads the run dir's `model.pt` (or an explicit `ckpt`),
2. rebuilds the actor on CPU with rsl-rl's own model classes via the SAME cfg
   mapping training used (`spec_to_train_cfg`), loading weights strictly,
3. traces it with a static batch-1 dummy input (the car infers one frame at a
   time; OpenVINO 2021.1 prefers static shapes),
4. verifies the graph against torch (`atol=1e-5`) if `onnxruntime` is
   installed — a numeric mismatch raises, a missing runtime only warns.

## Graph inputs / outputs

One input, named for the actor kind:

| Actor | Input | Shape | Notes |
|-------|-------|-------|-------|
| camera (`actor_keys=("camera",)`) | `FRONT_FACING_CAMERA` | `(1, 3·stack, H, W)` | float32 in `[0, 1]`; stack contract in the model card |
| feature (`actor_keys=("state",)`) | `STATE` | `(1, state_dim)` | **raw** feature vector — the `EmpiricalNormalization` is baked into the graph, do NOT normalize again; layout in the model card |

Output: `action` `(1, 2) = [steer, speed]` — the raw Gaussian mean,
**unbounded**: clip to `[-1, 1]` before use (the training env clips before
mapping). Recurrent actors, routed (`obs_actor`) actors, and discrete action
tables are not exportable (the guards name each case).

A deployed feature policy needs the state vector assembled onboard — stock
AWS nodes only feed camera models; see the perception pipeline
(`docs/concepts/perception.md`) for the onboard path.

## Action mapping in the model card

The card records the normalized→physical mapping so the deployed controller can
denormalize:

```json
"normalized_to_physical": {
  "steering": {"low": -30.0, "high": 30.0, "unit": "deg"},
  "speed":    {"low": 0.1,   "high": 4.0,  "unit": "m/s"}
}
```

This matches the training-time action map (see [Rewards & actions](../concepts/rewards-actions.md)).
The bounds are the run's own action caps from `action_caps(spec.env)`
(`experiment/spec.py`), the same function the training sim's config is built
from: `speed.high` follows `EnvSpec.max_speed` (2.0 for a `max_speed=2.0` run),
while `speed.low` and the steering cap are the `physics/limits.py` defaults
because no spec knob sets them. `model_metadata.json` carries the same ranges,
which is what the stock navigation node rescales `[-1, 1]` into.

The card also records the spec, `spec_id`, checkpoint path, opset, SHA256, and
whether the graph was verified against torch.

## Timing contract in the model card

The camera entry of `observations` states the timing the policy trained at
(ADR 0005); the car node must reproduce it exactly:

```json
"FRONT_FACING_CAMERA": {
  "frame_stack": 4,
  "stack_order": "oldest_first (newest frame = last 3 channels)",
  "stack_priming": "repeat_first_frame (never zeros)",
  "control_dt_s": 0.07,
  "frame_interval_s": 0.07,
  "frame_stride": 1,
  "stack_source": "policy_decisions",
  "stack_source_note": "the stack holds the frames observed at policy decisions ..."
}
```

- `control_dt_s` — seconds between policy decisions in training (the node
  runs the model once per `control_dt_s`).
- `frame_interval_s` — seconds between the stacked frames; `frame_stride =
  frame_interval_s / control_dt_s` decisions apart.
- `stack_source = "policy_decisions"` — the stack is fed by the frames the
  policy observed at its decisions (every `frame_stride`-th one), not by
  every camera frame; a skipped decision leaves a gap, never a duplicate.

The `STATE` entry of a feature policy carries `control_dt_s` too.
