# Perception

Driving the policy from a camera instead of the simulator's exact state.

`PerceptionFeatures` gives the policy 29 channels. Seven of them describe where
the car is and what is ahead — the ones a real DeepRacer has no sensor for. This
directory trains a CNN to recover those seven from the camera, freezes it, and
puts it in the loop so the policy is driven by estimates instead of ground
truth. The remaining 22 channels (past actions, command deltas) stay computed
onboard, exactly as they would be on the car.

The repo already has a `FrozenCNNToFeatureVector` stage, which turns images into
a 256 dimension embedding. This is not that. The CNN here predicts seven named
physical quantities, so the observation vector keeps its exact layout and the
same policy weights run on either source. That is what makes the comparison
possible: swap where the seven numbers come from, change nothing else.

Run everything from the repository root:

```bash
.venv/bin/python -m perception.<module>
```

## Pipeline

```
   data_generation  ->  train_cnn  ->  train_policy_with_cnn  ->  evaluation
      camera frames      the CNN        the policy, fine-tuned      does it cost
      + exact targets                   through the frozen CNN      anything?
```

**1. Collect** — one track at a time, 32 envs, each with its own randomised
track appearance and starting point.

```bash
python -m perception.data_generation Monaco
```

**2. Train the CNN** — 4 stacked frames in, 7 scalars out. Saves the best
checkpoint by validation loss.

```bash
python -m perception.train_cnn
```

`AUGMENT` at the top of `train_cnn.py` switches camera jitter on for the
training frames. It is what stands between a CNN that reads a clean render and
one that survives a real camera, and it costs about 3x per frame loaded.

Validation is never augmented, so the loss stays comparable across runs. The two
settings write to different checkpoints — `perception_augmented.pt` and
`perception.pt` — so neither run can overwrite the other. Point a policy at one
with `feature_params={"checkpoint": "perception/perception_augmented.pt"}`.

Measured, on the same clean validation set: augmenting costs nothing in
accuracy (MSE 0.00977 against 0.00972, mean R2 0.796 against 0.799). On
deliberately degraded frames the augmented network loses 3% against the clean
one's 6%, so it is steadier, but only slightly.

The reading is that the network was already photometrically invariant before
any augmentation, for two reasons: collection already randomises the track
appearance over 32 palettes, and the seven targets are geometric -- an edge
stays an edge when gain or gamma move. The sim-to-real gap this leaves is
structural, not photometric: lens distortion, mount pose, motion blur.

**3. Fine-tune the policy through it** — same env, same policy, same reward;
only the source of the seven channels changes. Starts from
`perception/reference_policy.pt`, a policy already trained on the exact values.

```bash
caffeinate -i .venv/bin/python -m perception.train_policy_with_cnn
```

## Files

| file | what it is |
|---|---|
| `model.py` | `PerceptionCNN` — 4 convs, 2 dense layers, 496 k parameters |
| `dataset.py` | frame stacks served from a flat memmap cache |
| `augment.py` | camera jitter: exposure, gamma, contrast, white balance, noise |
| `data_generation.py` | collects one track's rollouts |
| `train_cnn.py` | trains the CNN |
| `cnn_features.py` | `CNNPerceptionFeatures` — the frozen CNN in the env loop |
| `noisy_features.py` | exact values plus noise the size of the CNN's error, no renderer |
| `train_policy_with_cnn.py` | the policy fine-tuned through the real CNN |
| `train_policy_with_noise.py` | the policy trained on simulated CNN error |

### `evaluation/`

| file | what it answers |
|---|---|
| `compare_perception.py` | same policy, same track — what does the CNN cost? |
| `sweep_tracks.py` | where does it fail, and what geometry do those tracks share? |
| `ablation.py` | which channels carry the loss: "where am I" or "what is ahead"? |
| `compare_noise.py` | the two ends of the ablation on their own |
| `evaluate_with_cnn.py` | a trained policy under the CNN, no training |

### `visualization/`

`showcase.py` runs the measurements, picks two tracks from the numbers, and
stitches the side-by-side videos. The rest are single-purpose: fleet videos,
policy paths, dataset paths and frames, track catalogue, top-down renders.

## Running off a Mac

Nothing here is macOS-only, but several defaults were sized for an Apple Silicon
laptop. Every one of them is a parameter, marked `# mac:` at its line.

| where | now | on a CUDA machine |
|---|---|---|
| `data_generation.py` | `backend="cpu"` | `"gpu"` — Madrona batches the camera on the GPU, far faster than the CPU rasterizer |
| `train_policy_with_cnn.py` | `backend="cpu"` | `"gpu"`, same reason |
| `train_policy_with_cnn.py` | `num_envs=64` | raise it; 64 was sized to one core out of ten |
| `train_cnn.py` | `num_workers=8` | match the core count |
| `visualization/render_track_topdown.py` | `"rasterizer"` | drop the line to use the default renderer |

Chosen automatically, nothing to change: the CNN's device (`cuda`, else `mps`,
else `cpu`) in `train_cnn.py`, and in `cnn_features.py` MPS when it is there and
the env's own device otherwise. The banner font in `showcase.py` falls back to
PIL's default when the macOS path is missing.

`caffeinate -i` in the commands is macOS-only: it stops the machine sleeping
during a long run. Drop it elsewhere.

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
-- and identically for both policies, +0.82 against +0.83. Training on five times
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

**Outputs are not versioned.** Datasets land in `data/`, results and videos in
`runs/`; both are ignored by git.
