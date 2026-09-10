# Feature vector vs camera training

Two observation modalities, one simulator. The choice decides your training
speed, your debugging experience, and whether the result can drive the
physical car. Every number below was measured in this repo on an
RTX 4060 Ti.

## The comparison

| | **Feature vector** | **Camera** |
|---|---|---|
| Observation | state vector: track-relative pose, velocities, last action, K look-ahead waypoints ([feature sets](features.md)) | stacked RGB frames at the car's native 160×120 ([renderers](renderers.md)) |
| Pipeline stage | `FeatureEnvironment() >> VectorPolicy()` | `CameraEnvironment() >> AsymmetricCameraPolicy()` |
| Typical env count | 1024+ | 64–128 (rollout storage of stacked images is the VRAM limit) |
| Training throughput | **50k–100k env-steps/s** | **~3k env-steps/s** |
| Time to a lap-completing policy | ~90 s (100% completion) | hours: first-ever laps took 30M steps ≈ 3 h |
| Training stability | reliable | **bistable takeoff**: identical config + seed can park or fly (GPU nondeterminism); plan for redraws |
| Hardware | runs CPU-only (`backend="cpu"`) | NVIDIA GPU in practice (the [CPU rasterizer](renderers.md) trains end-to-end at ~90 env-steps/s — fine for CI, ~30× too slow for real runs) |
| Needs a renderer | no (`NullRenderer`) | yes — and the [choice costs 11×](renderers.md) |
| Reward sensitivity | forgiving | decisive: reward design moved held-out completion from ~8% to ~78% |
| Deploys to the physical car | [exports](../guides/deployment.md) (`STATE` input, normalization baked) — but the car must assemble the state onboard via the [perception pipeline](perception.md); stock AWS nodes can't | **yes**: the primary [ONNX export](../guides/deployment.md) target (`FRONT_FACING_CAMERA` input) |

## Why feature vector exists (even though it can't drive the stock car)

It is the **fast half of every workflow**:

- **Prototype rewards, algorithms, and tracks** at 100k steps/s before
  paying the 30× camera cost. A reward idea that fails on privileged state
  will not be saved by pixels.
- **Prove the task**: feature policies complete 100% of laps here, which is
  how we know any camera failure is a *perception/learning* problem, not an
  env or reward bug.
- **CI and CPU-only machines**: `FeatureEnvironment(backend="cpu")` trains
  without a GPU.

## Why camera training is the destination

The physical DeepRacer sees exactly one thing: a front camera. A directly
deployable policy must map those pixels to steering and throttle (a feature
policy exports too, but needs the perception pipeline to feed it onboard) —
which is why everything
hard in this repo (renderer choice, the [DR catalog](../reference/dr-catalog.md),
the [track zoo](../guides/track-zoo.md), reward design) ultimately serves
the camera modality.

What the repo's camera experiments established (the `experiments/best_camera*`
notebooks record the full story):

- **`frame_stack=1` with an asymmetric critic beat stacking**: the critic
  privately reads `("camera", "state")` so it sees velocities; the actor
  needs only the current frame.
- **Cap the action std** (`distribution={"std_range": (0.1, 1.0)}`) — the
  std dynamics are otherwise unstable (explosion or collapse).
- **Lower lr than feature training** (≈1.5e-4 vs 3e-4 defaults) and
  `gamma=0.995`.
- **Expect the takeoff lottery**: judge configs by whether they *ever* take
  off, kill flat runs early (the TB signature: mean reward pinned negative
  with short or parked episodes past ~5M steps), and redraw.
- **The reward is the strongest lever** — see the
  [reward story](rewards-actions.md#what-the-reward-taught-us-a-short-story).

## The bridge: asymmetric critics

`AsymmetricCameraPolicy(actor_keys=("camera",), critic_keys=("camera", "state"))`
is the standard way to use both modalities at once: the **actor** sees only
what the car will see, the **critic** — which exists only during training —
also reads the privileged state vector. You get feature-grade value
estimates guiding pixel-grade actors, at no deployment cost.

## Recommended path

1. Iterate your idea (reward, tracks, algorithm) on feature vector until it
   laps reliably.
2. Switch the pipeline to `CameraEnvironment` + `AsymmetricCameraPolicy`,
   keep the asymmetric critic on `("camera", "state")`.
3. Preview what the cars will actually see with
   [`view_zoo`](renderers.md) before spending GPU-hours.
4. Add [domain randomization](domain-randomization.md) and
   [zoo tracks](../guides/track-zoo.md) for generalization, then
   [export](../guides/deployment.md).
