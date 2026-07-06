# DeepRacer-Genesis — Experiment Framework & Evaluation Plan

A design + phased build plan for: (1) a composable, **config-as-code** experiment system driven by a
`>>` builder DSL and authored as Python functions/classes, (2) running the camera / feature /
asymmetric / Safe-RL experiments on top of TorchRL, and (3) a combination-grouped evaluation report
with before/after (baseline vs treatment) comparisons that doubles as your future ablation harness.

**Authoring is Python, not YAML and not CLI flags.** Experiments are functions or classes in an
`experiments/` package. The `>>` DSL builds an in-memory `ExperimentSpec`; that spec is
content-hashed for identity and may be *dumped* to a dict/JSON as a run record, but it is never
authored or loaded back from a file. An ablation axis is a Python comprehension or a subclass —
refactorable, autocompleted, and validated at build time.

---

## 0. The core idea

Three hard separations. Everything else follows from these.

1. **Declaration is build-time, pure Python.** An experiment is a function (or an `Experiment`
   class) that runs the `>>` chain once, imports no torch/genesis, and returns a frozen
   `ExperimentSpec` — plain data. `>>` composes a *spec*, it is not a per-step pipe.

2. **Instantiation turns a spec into live TorchRL objects.** A `Builder` reads the spec and
   constructs the Genesis-backed env, obs/action transforms, actor/critic modules, collector,
   buffer, and loss. This is the only layer that imports the heavy stuff.

3. **Execution runs and measures.** A `Trainer` runs the loop and writes an `EvalRecord`; a
   `Reporter` aggregates many records into the grouped report.

The payoff: an experiment's identity *is* its configuration (a content hash of the spec). Re-runs
are cache hits, results are keyed by config, and an ablation is just "many specs that differ in one
field." Before/after and ablation become the same machinery.

```
  Python authoring        ExperimentSpec           Builder                Trainer / Evaluator
  (functions/classes  --> (frozen data,     -->   (spec -> TorchRL  -->   (train loop,
   + >> DSL, no torch)     sha1 id;                 env/policy/loss)        EvalRecord)  -->  Reporter
                           dict dumped as                                                    (grouped report
   run(name | fn | obj)    a run *record*)                                                    + before/after)
```

The dict/JSON dump is an **output** for provenance only; nothing in the authoring path reads it back.

---

## 1. The `>>` builder DSL

### 1.1 Mechanism

`>>` is `__rshift__`. Each **Stage** knows how to fold itself into the spec via `apply(spec) -> spec`.
Composition builds a `Pipeline`; `Pipeline.build()` left-folds the stages over an empty spec, then
validates. The spec is a frozen dataclass, so every `apply` uses `dataclasses.replace` — stages never
mutate. Build order (`env -> obs-DR -> encoder -> policy -> action-DR`) mirrors the conceptual
dataflow; it's still build-time only.

### 1.2 Authoring idioms

**Function idiom** — one function per experiment, registered by name via `@experiment`.
**Class idiom** — an `Experiment` subclass; class attributes are the config surface; a variant is a
subclass (`class SafeTransferTight(SafeTransfer): budget = 10.0`) or constructor kwargs.
**Running** — `run(target, **overrides)` accepts a registered name, a function, an `Experiment`
class/instance, a `Pipeline`, or a raw spec. Name lookup replaces "config file path".

### 1.3 Stage taxonomy

| Stage kind | Concrete stages | Fills |
|---|---|---|
| **Environment** (source, must be first) | `CameraEnvironment(render, resolution, fov, ...)`, `FeatureEnvironment(features, lookahead_k, ...)`, `SafeRLCameraEnvironment(..., cost, budget)`, `SafeRLFeatureEnvironment(...)` | `spec.env` |
| **Obs DR** | `DomainRandomizationCamera(brightness, contrast, hue, blur, cutout, noise, camera_jitter)`, `DomainRandomizationPhysics(friction, mass, com, gains)` | `spec.obs_dr` |
| **Encoder** (optional) | `FrozenCNNToFeatureVector(checkpoint, output_dim, layer)` | `spec.encoder` |
| **Policy** (exactly one) | `AsymmetricCameraPolicy(actor_keys, critic_keys, cnn, mlp)`, `VectorPolicy(keys, mlp)`, `AsymmetricVectorPolicy(actor_keys, critic_keys, mlp)` | `spec.policy` |
| **Action DR** | `DomainRandomizationActions(steer_noise, speed_noise, delay_steps)` | `spec.action_dr` |
| **Algorithm** (optional terminal, usually inferred) | `PPO(...)`, `PPOLagrangian(budget, pid, ...)` | `spec.algorithm` |

SafeRL* envs set `emits_cost=True`; image aug becomes TorchRL transforms; physics DR is applied
env-side at reset; the frozen CNN is an obs transform; asymmetry = `critic_keys ⊇ actor_keys`.

### 1.4 Validation rules (fail at build, not at runtime)

- First stage is an `Environment`; exactly one `Policy` stage.
- `FrozenCNNToFeatureVector` requires an upstream **camera** env and a downstream **vector** policy.
- Asymmetric policy: `critic_keys ⊇ actor_keys`; every referenced key must be produced by the env
  or the encoder.
- Cost-emitting env => algorithm must be Lagrangian (auto-added, or explicit). Plain PPO on a cost
  env is a warning.
- `render="nyx"` => no heterogeneous track DR (repo constraint: heterogeneous morphs are
  Madrona-only).

### 1.5 The two reference pipelines

**Env 1** (`cam_baseline`) — end-to-end camera, asymmetric, full DR, unconstrained PPO.
**Env 2** (`SafeTransfer`) — Safe-RL camera env, DR, **frozen-CNN -> vector transfer**, vector
policy, PPO-Lagrangian. Representation transfer + safe RL: isolates "do frozen visual features
suffice vs. end-to-end vision?" and "does the constraint hold on transferred representations?"

---

## 2. The ExperimentSpec

Frozen dataclass tree (`EnvSpec`, `ObsDRSpec`, `EncoderSpec`, `PolicySpec`, `ActionDRSpec`,
`AlgorithmSpec`, plus seed/ablation_group/variant). `id()` is a content hash (sha1 of sorted-key
JSON, 12 chars) so identity == configuration; `to_dict()` exists for the hash and the run record
only — **no `from_yaml`/`from_dict` in the authoring path.**

Output directory convention: `runs/{ablation_group}/{variant}-{seed}-{id}/` holds checkpoints,
`spec.json` (record only), TB logs, and `eval_record.json`. A config change produces a new directory
automatically; identical configs collide (intentional cache).

---

## 3. Instantiation layer (spec -> TorchRL)

One `Builder`: `base_env()` (DeepRacerEnv as TorchRL `EnvBase`), `env()` (+ obs-DR transforms +
frozen-CNN encoder + action-DR), `actor()` (in_keys = actor_keys), `critic()` (in_keys =
critic_keys ⊇ actor_keys), `loss()` (ClipPPOLoss | PPOLagrangianLoss), `collector()`, `buffer()`,
`optimizers()`. Physics DR is *not* a transform — the builder passes it into the env for
per-episode resampling at reset via `envs_idx`.

---

## 4. Safe RL — PPO-Lagrangian + PID

Built by hand in the TorchRL loop. ~50 lines of delta over PPO:
1. **Cost signal** `c_t` emitted by the `SafeRL*` env (offtrack/crash pulled OUT of the reward —
   declare "violate at most `budget`" instead of hand-tuning a penalty weight).
2. **Cost critic** `V_c` — a second `ValueOperator`, trained by a second GAE on the cost stream.
3. **Lagrange multiplier** λ >= 0 — updated by a **PID controller** on `(J_cost - budget)`
   (Stooke et al.'s PID-Lagrangian) rather than naive dual ascent.
4. **Surrogate**: maximize `A_reward - λ·A_cost` (optionally scaled by `1/(1+λ)`), clipped as usual.

Develop on the **feature env first** (no rendering => ~10x faster iteration), then camera. The DSL
infers all of this from the SafeRL* env stage.

---

## 5. Evaluation & the combination report

**Axes**: modality {camera, feature}, render {madrona, nyx, none}, algorithm {ppo,
ppo_lagrangian_pid}, asymmetry {symmetric, asymmetric}, encoder {none, frozen_cnn}, dr_profile
{none, obs, action, physics, full}.

**Metrics** — task: completion_rate, lap_time, mean_progress, offtrack_rate, mean_return;
efficiency: train steps/s, sample_efficiency (env-steps to X% completion); safety: mean_cost,
cost_violation_rate, budget_satisfied, lambda_final; robustness: evaluation under a **held-out DR
profile** (the metric that justifies DR and asymmetric critics).

Eval protocol: fixed eval envs + held-out seeds, deterministic policy (mean action), K seeds per
spec -> mean ± std.

**Before/after = ablation pairs**: specs tagged with a shared `ablation_group` and different
`variant`; the reporter groups and prints per-metric deltas. Examples: reward_penalty vs
cost_budget; no_dr vs full_dr; end2end vs frozen_cnn; madrona vs nyx. Step-0 vs final checkpoints
covered by the same EvalRecord mechanism.

**Report output**: grouped table (rows = combination cells, cols = metrics, mean ± std over seeds) +
delta table per ablation_group + `report.md`/`report.csv`, regenerable from stored
`eval_record.json`s without re-training.

---

## 6. Ablation patterns

- **Axis sweep**: `sweep(base_fn, "field", [values])` -> specs differing in one field, auto-tagged
  into one ablation_group; or subclasses; or comprehensions over instances.
- **Grid**: Cartesian product; invalid combos dropped by validate() with a logged reason.
- **Seeds**: spec x range(K) -> mean ± std.
- **Reproducibility**: content-hash id => deterministic output dir; re-runs are cache hits.

---

## 7. Implementation roadmap (phased)

- **Phase 0 — Declaration core.** Spec tree + hash; Stage/Pipeline/`>>`; inference; validation;
  registry + Experiment + run(). Unit tests. No torch.
- **Phase 1 — Feature PPO end-to-end.** TorchRL EnvBase wrapper (feature modality); Builder;
  Trainer + logging + checkpoint + EvalRecord. Run `feature_baseline`. **The baseline.**
- **Phase 2 — Camera + asymmetry + both renderers.** CNN actor, asymmetric critic, Madrona + Nyx.
  Run `cam_baseline` and `cam_nyx`.
- **Phase 3 — DR stages.** Obs-DR transforms, action-DR, physics-DR. Run full Env 1.
- **Phase 4 — Safe RL.** Cost stream, cost critic + cost GAE + PID-λ + combined surrogate.
  Feature first, then camera.
- **Phase 5 — Encoder/transfer + full Env 2.** FrozenCNNToFeatureVector from a Phase-2 checkpoint.
  Run `SafeTransfer`.
- **Phase 6 — Ablation + reporting.** sweep/grid/pairing; Reporter; report.md/csv from records.

Each phase produces at least one runnable experiment and its EvalRecord.

---

## 8. Module / directory layout

```
deepracer_genesis/
  experiment/        # the framework
    spec.py stages.py registry.py run.py builder.py trainer.py evaluator.py ablation.py report.py
  algorithms/
    ppo_lagrangian.py
  envs/ randomization/   # existing
experiments/         # AUTHORED experiments (the config surface)
  feature.py camera.py safe.py
runs/                # {ablation_group}/{variant}-{seed}-{id}/
```

---

## 9. Settled decisions

- Config-as-code; no YAML/CLI authoring.
- "Before/after" unified with ablation pairing (step-0 vs final also supported via EvalRecord).
- SafeRL modeled as env flavor with the Lagrangian algorithm inferred (explicit stage allowed as
  override).
- Frozen CNN realized as an obs transform writing an "encoded" key.
- TorchRL is the single stack for all experiment types (apples-to-apples comparisons).
