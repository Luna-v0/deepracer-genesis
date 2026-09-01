# Deployment (ONNX)

Export a trained actor to ONNX (plus a `model_card.json`) so it can run on the
physical DeepRacer or any ONNX runtime. The exporter is
`deepracer_genesis/deploy/onnx.py`.

> Mental model in one sentence: `export_policy` rebuilds the actor on CPU from a
> checkpoint, traces it to ONNX with the exact observation keys as named inputs
> (`FRONT_FACING_CAMERA` for camera actors — the name the car's inference
> stack expects), and writes the car bundle: `policy.onnx` +
> `model_metadata.json` + `model_card.json` + a `<name>.tar.gz` to upload.

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
2. rebuilds the actor on CPU mirroring `Builder.actor()` (CNN + MLP) with the loaded
   weights,
3. creates dummy inputs from `spec.policy.actor_keys`,
4. `torch.onnx.export(...)` with `dynamic_axes` on the batch dim,
5. verifies against torch (`atol=1e-4`) if `onnxruntime` is installed.

## Graph inputs / outputs

Inputs mirror the actor's observation keys:

| Key | Shape | Notes |
|-----|-------|-------|
| `camera` | `(batch, 3, H, W)` | float32 in `[0, 1]`; `H, W = spec.env.resolution` |
| `state` | `(batch, state_dim)` | layout in the model card (`state_layout`) |
| encoder out | `(batch, output_dim)` | if a `FrozenCNNToFeatureVector` encoder is used |

Outputs:

- **Continuous**: `action` `(batch, 2) = [steer, speed]` in `[-1, 1]`, deterministic
  tanh head (no sampling at deployment).
- **Discrete**: `logits` `(batch, num_actions)`; argmax indexes the action table.

## Action mapping in the model card

The card records the normalized→physical mapping so the deployed controller can
denormalize:

```json
"normalized_to_physical": {
  "steering": {"low": -30.0, "high": 30.0, "unit": "deg"},
  "speed":    {"low": 0.1,   "high": 4.0,  "unit": "m/s"}
}
```

This matches the training-time action map (see [Rewards & actions](../concepts/rewards-actions.md));
the physical bounds come from `physics/limits.py`. The card also records the spec,
`spec_id`, checkpoint path, opset, SHA256, and whether the graph was verified
against torch.
