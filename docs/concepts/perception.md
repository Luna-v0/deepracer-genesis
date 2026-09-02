# Perception

Driving the policy from a camera instead of the simulator's exact state.

`PerceptionFeatures` gives the policy 29 channels. Seven of them describe where
the car is and what is ahead — the ones a real DeepRacer has no sensor for. A CNN
is trained to recover those seven from the camera, frozen, and put in the loop,
so the policy is driven by estimates instead of ground truth. The remaining 22
channels (past actions, command deltas) stay computed onboard, exactly as they
would be on the car.

> Mental model in one sentence: train `PerceptionCNN` on camera frames paired
> with the seven privileged channels, freeze it, then swap `PerceptionFeatures`
> for `CNNPerceptionFeatures` — the observation layout is unchanged, so the same
> policy weights run on either source.

---

## This is not the `FrozenCNNToFeatureVector` stage

The repo has a `FrozenCNNToFeatureVector` stage
([Experiments](experiments.md)), which turns images into a 256-dimension
embedding under an `encoded` key. This is not that, and does not use it. The CNN
here predicts seven named physical quantities, so the observation vector keeps
its exact layout. That is what makes the comparison possible: swap where the
seven numbers come from, change nothing else.

## Where the code lives

Library code, importable and tested, in `deepracer_genesis.perception` (see the
[API page](../api/perception.md)):

| module | what it is |
|---|---|
| `perception.model` | `PerceptionCNN` — 4 convs, 2 dense layers, 496 k parameters |
| `perception.features` | `CNNPerceptionFeatures` (the frozen CNN in the env loop) and `NoisyPerceptionFeatures` (exact values plus noise the size of the CNN's error, no renderer) |
| `perception.dataset` | `RolloutDataset` — frame stacks served from a flat memmap cache — plus the track split (`TRAINING_TRACKS`, `HOLDOUT_TRACKS`) |
| `perception.augment` | camera jitter: exposure, gamma, contrast, white balance, noise |

Run drivers, under `experiments/perception/`. Every command below is run from
the repository root:

```bash
uv run python -m experiments.perception.<module>
```

## Pipeline

```
   data_generation  ->  train_cnn  ->  train_policy_with_cnn  ->  evaluation
      camera frames      the CNN        the policy, fine-tuned      does it cost
      + exact targets                   through the frozen CNN      anything?
```

**1. Collect** — one track at a time, 32 envs, each with its own randomised
track appearance and starting point. Writes
`data/<track>_v2/rollout_*.parquet`.

```bash
uv run python -m experiments.perception.data_generation Monaco
```

**2. Train the CNN** — 4 stacked frames in, 7 scalars out. Saves the best
checkpoint by validation loss under `runs/cnn/`.

```bash
uv run python -m experiments.perception.train_cnn
```

`JITTER` at the top of `train_cnn.py` switches camera jitter on for the training
frames. It is what stands between a CNN that reads a clean render and one that
survives a real camera, and it costs about 3x per frame loaded.

Validation is never jittered, so the loss stays comparable across runs. The two
settings write to different checkpoints — `runs/cnn/perception_jittered.pt` and
`runs/cnn/perception.pt` — so neither run can overwrite the other.

Measured, on the same clean validation set: jitter costs nothing in accuracy
(MSE 0.00977 against 0.00972, mean R2 0.796 against 0.799). On deliberately
degraded frames the jittered network loses 3% against the clean one's 6%, so it
is steadier, but only slightly.

The reading is that the network was already photometrically invariant before any
jitter, for two reasons: collection already randomises the track appearance over
32 palettes, and the seven targets are geometric — an edge stays an edge when
gain or gamma move. The sim-to-real gap this leaves is structural, not
photometric: lens distortion, mount pose, motion blur.

**3. Fine-tune the policy through it** — same env, same policy, same reward;
only the source of the seven channels changes. Starts from
`runs/cnn/reference_policy.pt`, a policy already trained on the exact values.

```bash
uv run python -m experiments.perception.train_policy_with_cnn
```

## Putting the CNN in the loop

```python
from deepracer_genesis.perception.features import CNNPerceptionFeatures

CameraEnvironment(
    feature_set=CNNPerceptionFeatures,
    feature_params={"checkpoint": "runs/cnn/perception_jittered.pt"},
    resolution=(160, 120),
    frame_stack=4,
    tracks=tracks,
)
```

**`checkpoint` is required and has no default.** A missing one used to fall
through to plain `PerceptionFeatures`, which silently served the simulator's
privileged values — a perception run that was quietly a perfect-perception run.
It now raises `ValueError` at construction, as does a camera-less env.

The checkpoint is also checked against the env before the first step: its input
width must be `3 * frame_stack`, and it must predict exactly as many channels as
the feature set's `cnn_target_slice` spans. A stale checkpoint fails at build
time instead of skewing a measurement. `feature_params["cnn_device"]` overrides
where the CNN runs; by default it follows the env's device.

`NoisyPerceptionFeatures` is the same swap without a renderer: it perturbs the
same slice with Gaussian noise scaled to the CNN's measured per-channel error
(`SIGMA`), at `feature_params["noise"]` strength, optionally restricted to named
`noise_channels`. It trains at feature-env speed, which is what makes the
ablation affordable — the caveat being that its noise is white, where the real
CNN's error is time-correlated.

```bash
uv run python -m experiments.perception.train_policy_with_noise
```

## Evaluation

| module | what it answers |
|---|---|
| `experiments.perception.evaluation.compare_perception` | same policy, same track — what does the CNN cost? |
| `experiments.perception.evaluation.sweep_tracks` | where does it fail, and what geometry do those tracks share? |
| `experiments.perception.evaluation.ablation` | which channels carry the loss: "where am I" or "what is ahead"? |
| `experiments.perception.evaluation.compare_noise` | the two ends of the ablation on their own |
| `experiments.perception.evaluation.evaluate_with_cnn` | a trained policy under the CNN, no training |

```bash
uv run python -m experiments.perception.evaluation.compare_perception <model.pt> [track ...] [--video] [--cnn <cnn.pt>]
uv run python -m experiments.perception.evaluation.sweep_tracks <model.pt> [track count]
uv run python -m experiments.perception.evaluation.evaluate_with_cnn <model.pt>
```

## Visualization

`experiments.perception.visualization.showcase` runs the measurements, picks two
tracks from the numbers, and stitches the side-by-side videos. The rest are
single-purpose: fleet videos, policy paths, dataset paths and frames, track
catalogue, top-down renders.

```bash
uv run python -m experiments.perception.visualization.showcase <model.pt>
```

## Portability

Nothing here is platform-specific, but several defaults in the run drivers were
sized for the laptop the work was done on — an Apple Silicon machine with no
CUDA and ten cores. Each one is a parameter, marked `# mac:` at its line where
the choice is not obvious. Raise or switch them to match the machine you are on:

| where | as written | on a CUDA machine |
|---|---|---|
| `data_generation.py` | `backend="cpu"` | `"gpu"` — Madrona batches the camera on the GPU, far faster than the CPU rasterizer |
| `train_policy_with_cnn.py` | `backend="cpu"` | `"gpu"`, same reason |
| `train_policy_with_cnn.py` | `num_envs=64` | raise it; 64 was sized to one core out of ten |
| `train_cnn.py` | `TRAIN_WORKERS`, `VAL_WORKERS` | match the core count |
| `visualization/render_track_topdown.py` | `"rasterizer"` | drop the line to use the default renderer |

Chosen automatically, nothing to change: the CNN trains on `cuda` when it is
there and `cpu` otherwise, and `CNNPerceptionFeatures` follows the env's device
unless `cnn_device` says otherwise. The banner font in `showcase.py` falls back
to PIL's default when the macOS font path is missing.

On macOS, prefixing a long run with `caffeinate -i` stops the machine sleeping
partway through. It is a macOS command and is not part of any command line here.

The one-scene-per-process rule below is a limitation of the pyrender rasterizer,
which is the path taken because Madrona needs CUDA.

## What the measurements say

Three things were expected to matter and were measured instead. All numbers come
from the ten held-out tracks or from `evaluation/compare_perception.py`.

| change | expected | measured |
|---|---|---|
| exact simulator values -> frozen CNN | a real cost to driving | 0-1% on tracks the policy can drive |
| camera jitter on the CNN's training frames | robustness worth having | mean R2 0.796 against 0.799; 3% steadier on degraded frames |
| policy trained on 10 tracks -> 50 | fixes the hard tracks | off-track 0.31 -> 0.29, progress 42.9 m -> 44.9 m |

What does predict failure is corner severity: over the ten held-out tracks the
off-track rate follows the curvature spread of the track, rank correlation +0.82
— and identically for both policies, +0.82 against +0.83. Training on five times
the tracks, including one well past the hardest the policy had ever seen, did not
change that relationship at all.

So the limit is not the perception, and not the breadth of the training set.
Sharper corners are simply harder, and what is left to try sits in the policy:
its capacity, how far ahead it is allowed to look (two curvatures, at 1 m and
3 m), and how much the reward makes leaving the track cost.

## Notes

**One camera scene per process.** pyrender's OpenGL context is process-global:
destroying a previous scene tears down the live one, and the next render fails
on `glBindFramebuffer: invalid operation`. Every script that touches more than
one scene forks a subprocess per scene.

**Where the outputs go.** Datasets land in `data/`, which is git-ignored;
`DEEPRACER_DATA_ROOT` moves that root outside the source tree. Results go to
`runs/`, which *is* version-controlled — run records (`spec.json`,
`eval_record.json`) are committed. What is ignored there is only the bulky and
regenerable: `runs/cnn/` (checkpoints), `runs/videos/`, `runs/showcase/`,
`runs/figures/` and `runs/sweep_tracks.json`.
