# Rewards & actions

This page covers the two RL-facing transforms: how a normalized action becomes a
physical car command, and how the built-in reward is shaped.

> Mental model in one sentence: the policy emits `[steer, speed] ∈ [-1, 1]²`, which
> maps to radians + m/s (with Ackermann split across the front wheels), and the
> reward is a weighted sum of small per-step terms read from the same env state the
> [feature vector](features.md) uses.

---

## Actions: normalized → physical

The policy's normalized action is clipped to `[-1, 1]` and mapped in `base_env.py`:

```python
steer = actions[:, 0] * radians(max_steering_deg)                       # ±30° default
speed = min_speed + (actions[:, 1] + 1) * 0.5 * (max_speed - min_speed) # 0.1–4.0 m/s
```

Throttle is **unidirectional** (no reverse): `-1 → min_speed`, `+1 → max_speed`. A
discrete `action_table` of `(steer, speed)` pairs is supported — a scalar action
indexes it before mapping.

### Ackermann front steering

The commanded center angle `δ` is split into per-wheel angles so the inner wheel
turns more sharply (`ackermann_angles` in `physics/limits.py`):

```python
left  = atan2(L·tan δ, L − t/2·tan δ)
right = atan2(L·tan δ, L + t/2·tan δ)
#   L = WHEELBASE_M   = 0.163974 m
#   t = FRONT_TRACK_M = 0.159202 m
```

`Car.steer_targets` (`envs/entities.py`) chooses `parallel` (`δ.repeat(1, 2)`,
legacy) or `ackermann` per the `steering_model` config. `atan2` keeps it finite at
`δ = 0` (both wheels → 0). Constants live in `physics/limits.py` (the single source
of truth for `MAX_STEERING_DEG`, `MIN_SPEED`/`MAX_SPEED`, and the normalization
divisors).

## Rewards

A reward function is a plain callable, passed as a parameter (no registry):

```python
RewardFn = Callable[[DeepRacerEnv], dict[str, torch.Tensor] | torch.Tensor]
```

It maps the env to named `(N,)` per-step terms weighted by `reward_scales` and
summed — or to a bare `(N,)` tensor that *is* the step reward (see
[Custom rewards](#custom-rewards)). The built-in `deepracer` reward
(`envs/rewards.py`):

| Term | Formula | Intent |
|------|---------|--------|
| `progress` | `d_progress` | forward arc-length gained this step (the core signal) |
| `speed` | `clamp(v_forward, 0, max_speed)·dt` | sustain speed |
| `centered` | `exp(−(lateral/half_width)²)·dt` | stay near centerline |
| `heading` | `−|heading_err|·dt` | align with track tangent |
| `steering` | `−|steer_action|·dt` | discourage needless steering |
| `action_rate` | `−‖aₜ − aₜ₋₁‖²·dt` | smooth control |
| `off_track` | `−(lateral outside half_width − wheel_margin)·dt` | penalize edge-riding |

All terms scale by control `dt`, so weights are timestep-independent. Per-term sums
are tracked for logging. The full term/scale table with defaults lives in the
[reward parameters reference](../reference/reward-parameters.md).

The weighted sum is **not** the whole reward: when an episode ends by going
off track or flipping (not by timeout), `check_termination` adds a one-time
**crash penalty** (default −10.0) on top — the largest single-step magnitude
in the system. It appears in the TensorBoard breakdown as
`Episode/rew_crash_penalty` (so the per-term rows sum to what the learner
saw), and is searchable via `RewardShaping(crash_penalty=...)` /
`EnvSpec.crash_penalty`. Under a cost-emitting (safe-RL) env the penalty is
**not** applied — crashes become the constrained cost instead — so the same
reward fn trains against a different effective objective there.

!!! note "Signs live in the terms"
    Penalty terms are negative and `reward_scales` stay positive. This is
    load-bearing: until 2026-08 the `off_track` term was accidentally
    positive, so the default reward *paid* +2·dt per step for riding the
    track edge — 4× the centering bonus — and camera policies dutifully
    learned to crash. `tests/test_rewards.py` pins every term's sign.

### What the reward taught us (a short story)

Three generations of camera policy, same network, same PPO settings:

1. **Progress + "faster is better" speed bonus** — fast, crash-prone
   driving: ~1.0 off-track rate, under 40% lap completion even on training
   tracks.
2. **Progress gated on-track, no speed term** — reward-hacked itself: with
   nothing paying for motion, standing still on the centerline farms the
   `centered` bonus forever. The cars parked.
3. **Target-pace speed term** (`−|v − 1.6 m/s|·dt`) — the winner of a
   reward-design search scored on ground-truth lap completion. Roughly
   doubled completion everywhere, including 78% on a track the policy had
   never seen.

Moral: the reward is the strongest lever in this repo, it is searchable
(functions are parameters), and it must be *evaluated* on a metric it
cannot inflate.

### Custom rewards

Pass your own callable via `RewardShaping` (see [Experiments](experiments.md)):

```python
def my_reward(env):
    return {"progress": env.d_progress,
            "smooth": -(env.actions - env.last_actions).pow(2).sum(1)}

... >> RewardShaping(fn=my_reward, scales={"progress": 10.0, "smooth": 0.1})
```

A reward doesn't have to be a weighted sum. Return a bare `(N,)` tensor and
that tensor **is** the step reward — no term names, no scales (leave `scales`
empty; it logs to TensorBoard as `Episode/rew_total`):

```python
def paced(env):
    pace = -(env.v_forward - env.reward_params["target"]).abs() * env.dt
    return 10.0 * env.d_progress + pace

... >> RewardShaping(fn=paced, params={"target": 1.6})
```

Three supporting pieces make the monolithic form a first-class citizen:

- **`params`** — constants the fn reads via `env.reward_params`. Unlike a
  closure constant (`make_paced(1.6)`), they are part of the spec's content
  hash and land in `eval_record.json`, so two runs with different targets get
  different run dirs — and HPO can search them like any other spec field.
- **Diagnostic channels** — a term whose name starts with `_` is accumulated
  into the per-term TensorBoard breakdown (`Episode/diag_<name>`) but never
  summed into the reward, so a monolithic reward keeps its decomposition.
- **`total`** — returning `{"total": <reward>, "_pace": ..., "_progress": ...}`
  with no scales combines the two: `total` is the reward verbatim, the `_`
  terms are its logged breakdown. (With scales set, term names are ordinary
  and `total` has no special meaning.)

Misconfigurations fail loudly at the first step: a bare tensor alongside
`scales`, named terms with no `scales`, an unscaled non-`_` term next to
`total`, or a scale referencing a `_` term all raise `ValueError`.

Declare what your fn reads with `@reads(...)` (`envs/rewards.py`) so the
build-time learnability check can verify the critic sees those signals —
`spec.validate()` warns if a custom reward leaves it undeclared.

The env fields available to a reward (`v_forward`, `lateral`, `half_width`,
`heading_err`, `d_progress`, `actions`, `last_actions`, ...) are the same ones the
feature vector reads — see [Feature vectors](features.md) for the full palette, and
`envs/signals.py` for the shared **signal bus** that unifies
features, reward, and cost over one vocabulary (e.g. `off_track` as a reward term in
plain RL and a cost term under safe RL).
