# Dataset collection

For perception / sim-to-real work you often want `(frame, feature-vector)` datasets
rather than an RL loop. Two collectors live in `deepracer_genesis/datasets/`.

> Mental model in one sentence: `collect_rollout_dataset` drives a privileged expert
> under domain randomization and records temporally-contiguous camera + state
> rollouts to Parquet; `collect_camera_dataset` teleports the car over a pose grid
> for geometric coverage without dynamics.

---

## Rollout collection

`collect_rollout_dataset(target, *, out, steps, num_envs, agent, shard_steps, seed,
compress)` (`datasets/rollout.py`):

- `target` — any `>>` chain that builds a **camera** env (the policy stage is
  optional and ignored during collection).
- `agent` — a `PrivilegedAgent`; the default `NoisyExpert` steers from privileged
  track state with Ornstein-Uhlenbeck noise, so trajectories drift off the
  centerline and *do* go off-track sometimes — exactly the recovery data a
  frame-stacking CNN needs.
- Output: `rollout_XXXX.parquet` shards (zstd) with columns `env, t, episode, done,
  image` (PNG bytes), `state`, `action`, `pose = [x, y, yaw, progress_m]`, sorted
  `(env, t)`.
- A `meta.json` records `feature_set`, `state_layout`, `cnn_target_slice` (the
  channels a CNN must predict — see [Feature vectors](../concepts/features.md)),
  resolution, `control_dt` (the seconds between rows), `frame_stride` /
  `frame_interval_s` (the env's own stack spacing while collecting), and the
  DR configs used. `RolloutDataset(stride=s)` stacks rows `s` control steps
  apart, so its `frame_interval_s` is `s × control_dt` (ADR 0005).

```python
from deepracer_genesis.datasets.rollout import collect_rollout_dataset
from deepracer_genesis.experiment import CameraEnvironment
from deepracer_genesis.experiment.stages import (
    DomainRandomizationTrackAppearance, DomainRandomizationCamera, DomainRandomizationPhysics)

collect_rollout_dataset(
    CameraEnvironment(resolution=(160, 120), num_envs=64, tracks=("reinvent_base",),
                      random_direction=True)
    >> DomainRandomizationTrackAppearance(strength=0.7)
    >> DomainRandomizationCamera(brightness=(0.6, 1.4), hue=0.08, blur=0.5,
                                 cutout=0.1, noise=0.02, camera_jitter=True)
    >> DomainRandomizationPhysics(),
    out="datasets/rollouts", steps=2048, seed=0)
```

## Pose-grid (teleport) collection

`collect_camera_dataset(track, *, lateral_fracs, yaw_offsets, waypoint_stride,
resolution, num_envs, render, ...)` (`datasets/sweep.py`) sweeps a track on a grid of
lateral offsets × yaw offsets × waypoints, rendering frame + state at each pose with
**no policy and no dynamics** — geometric coverage of track views. Output is `.npz`
shards (`image`, `state`, `pose`) plus `meta.json`. One call = one track (Genesis
builds one scene per process).

## Train / test track splits

`splits.py` provides `TrackDataset(names, holdout, test_fraction, seed)` — a
deterministic by-name split (hash each name with the seed, so the split is stable
when you add or remove other tracks). `holdout` defaults to the physically-printed
tracks. Use it to keep evaluation tracks unseen during training.

```python
from deepracer_genesis.datasets.splits import TrackDataset
ds = TrackDataset(test_fraction=0.2, seed=0)
train_tracks, test_tracks = ds.train, ds.test
```
