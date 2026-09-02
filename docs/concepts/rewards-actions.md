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
RewardFn = Callable[[DeepRacerEnv], dict[str, torch.Tensor]]
```

It maps the env to named `(N,)` per-step terms; the env weights them by
`reward_scales` and sums. The built-in `deepracer` reward (`envs/rewards.py:19-42`):

| Term | Formula | Intent |
|------|---------|--------|
| `progress` | `d_progress` | forward arc-length gained this step (the core signal) |
| `speed` | `clamp(v_forward, 0, max_speed)·dt` | sustain speed |
| `centered` | `exp(−(lateral/half_width)²)·dt` | stay near centerline |
| `heading` | `−|heading_err|·dt` | align with track tangent |
| `steering` | `−|steer_action|·dt` | discourage needless steering |
| `action_rate` | `−‖aₜ − aₜ₋₁‖²·dt` | smooth control |
| `off_track` | `−(lateral outside half_width − wheel_margin)·dt` | penalize leaving the road |

All terms scale by control `dt`, so weights are timestep-independent. Per-term sums
are tracked for logging. The `reward_scales` are positive magnitudes — each term
carries its own sign, so every penalty row above is already negative-valued.

### Custom rewards

Pass your own callable via `RewardShaping` (see [Experiments](experiments.md)):

```python
def my_reward(env):
    return {"progress": env.d_progress,
            "smooth": -(env.actions - env.last_actions).pow(2).sum(1)}

... >> RewardShaping(fn=my_reward, scales={"progress": 10.0, "smooth": 0.1})
```

The env fields available to a reward (`v_forward`, `lateral`, `half_width`,
`heading_err`, `d_progress`, `actions`, `last_actions`, ...) are the same ones the
feature vector reads — see [Feature vectors](features.md) for the full palette, and
`REFACTOR_PLAN.md` Part K for the planned shared **signal bus** that unifies
features, reward, and cost over one vocabulary (e.g. `off_track` as a reward term in
plain RL and a cost term under safe RL).
