# Reward parameters

The complete list of the built-in reward's terms, their default scales, and
the contract for writing your own reward function. Source of truth:
`deepracer_genesis/envs/rewards.py` (terms) and `configs/cfgs.py`
(default scales).

## How the reward is computed

A reward function maps the live env to named per-step `(N,)` terms; the env
multiplies each term by its `reward_scales` entry and sums:

```python
RewardFn = Callable[[DeepRacerEnv], dict[str, torch.Tensor]]
reward = sum(scale[name] * terms[name] for name in reward_scales)
```

Two conventions keep this sane:

- **Signs live in the terms, scales stay positive.** A penalty term is
  negative; making a scale negative to flip a term's meaning is a bug
  waiting to happen. (Historical note: until 2026-08 the `off_track` term
  was accidentally positive — the default reward *paid* for edge-riding at
  4× the centering bonus. `tests/test_rewards.py` now pins every sign.)
- **Every continuous term multiplies by `dt`**, so scales are
  timestep-independent.

Per-term episode sums are logged to TensorBoard, so you can see *which*
term moved when behavior changes.

## Built-in terms (`deepracer` reward)

| Term | Default scale | Formula | Sign | Encourages |
|------|--------------|---------|------|------------|
| `progress` | 10.0 | `d_progress` (arc-length gained, lap-wrap-corrected) | + | moving along the track — the core signal |
| `speed` | 0.5 | `clamp(v_forward, 0, max_speed) · dt` | + | going faster (see the warning below) |
| `centered` | 0.5 | `exp(−(lateral / half_width)²) · dt` | + | staying near the centerline |
| `heading` | 0.5 | `−\|heading_err\| · dt` | − | aligning with the track tangent |
| `steering` | 0.3 | `−\|steer_action\| · dt` | − | not steering needlessly |
| `action_rate` | 0.05 | `−‖aₜ − aₜ₋₁‖² · dt` | − | smooth control |
| `off_track` | 2.0 | `−(any wheel outside half_width − wheel_margin) · dt` | − | keeping all wheels on the road |

On top of the per-step terms, `mdp.check_termination` adds a one-time
`crash_penalty` (**−10.0**, `configs/cfgs.py`) when an episode ends by
going off track or flipping (not on timeouts).

!!! warning "The `speed` term trains qualifying laps, not race finishers"
    "Faster is better" plus dense progress reward reliably produces fast,
    crash-prone driving. The round-3 reward study found that replacing the
    speed term with a **target-pace** term — `−|v_forward − target| · dt`
    with `target ≈ 1.6 m/s` — roughly doubled lap completion on held-out
    tracks. See the storyline in [Rewards & actions](../concepts/rewards-actions.md).

## Overriding scales

`reward_scales` is a spec field — set it per experiment without touching
the reward function (scales are part of the run's content hash):

```python
... >> RewardShaping(scales={"off_track": 6.0, "speed": 0.1})
```

Referencing a term the reward function does not produce raises a
`KeyError` at the first step — a typo cannot silently do nothing.

## Custom reward functions

Pass any callable as a parameter (explicit dependency injection — there is
no registry):

```python
def my_reward(env):
    on = env.lateral.abs() < env.half_width
    return {"progress": env.d_progress * on.float(),
            "off_track": -(~on).float() * env.dt}

... >> RewardShaping(fn=my_reward, scales={"progress": 10.0, "off_track": 2.0})
```

Env state available to a reward (the same signal vocabulary the feature
vector reads): `d_progress`, `progress_m`, `v_forward`, `lateral`,
`half_width`, `heading_err`, `actions`, `last_actions`, `dt`,
`track.total_len_env`, and `cfg` (e.g. `cfg["termination"]["wheel_margin"]`).

Optionally declare what you read with `@reads(...)`
(`envs/rewards.py`) so the build-time learnability check can verify the
critic sees those signals; an undeclared custom reward simply skips that
check.

!!! tip "Reward designs are searchable — and hackable"
    Reward functions are ordinary values, so they can go **into** an HPO
    search (round 3 searched four families). Score such a study on a
    ground-truth metric the shaping cannot inflate (lap completion at a
    deterministic eval), never on the training reward itself — one studied
    family learned to park on the centerline and farm the `centered`
    bonus forever.
