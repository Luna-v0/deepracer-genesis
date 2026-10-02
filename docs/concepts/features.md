# Features: how the state vector is measured, derived, and extended

This is the "where does every number in the observation come from, and how do I
add my own" map for the feature vector. Every claim is anchored to a `file:line`
you can open.

> Mental model in one sentence: `_post_physics` reads a handful of raw sensors
> from Genesis (position, orientation quaternion, linear + angular velocity),
> localizes the car against the waypoint polyline, and stamps ~15 state fields
> onto the env — then the **selected `FeatureSet` is the only thing that turns
> those fields into the policy's vector**, so adding a feature means writing a
> `compute()` that reads `env.<field>` and returns one more column.

---

## 1. One call site: the feature set *is* the state vector

Every control step the env runs `_post_physics` (`envs/base_env.py:319`), which
ends by delegating the entire vector to whichever feature set is plugged in
(`base_env.py:360`):

```python
self.state_buf = self.feature_set.compute()
```

The set is chosen once at construction (`base_env.py:224`) from config:

```python
self.feature_set = resolve_feature_set(obs_cfg["feature_set"])(self, dict(obs_cfg["feature_params"]))
```

`resolve_feature_set(None)` → `ClassicFeatures`; passing a class picks that one
(`features.py:22`). `ClassicFeatures` and `PerceptionFeatures` are **mutually
exclusive alternatives**, never joined — the env uses exactly one, and its
`compute()` return *is* the whole state vector. (Composing/joining vectors is a
planned extension — see the signal bus, `REFACTOR_PLAN.md` Part K.)

Right after, optional Gaussian sensor noise is added to the actor's copy
(`base_env.py:361-362`); this is the *weakest* form of actor/critic asymmetry —
the stronger form (critic reads a richer signal set) is Part K.3.

---

## 2. The measurement stack — where the numbers come from

### 2.1 The bottom of the stack: raw Genesis rigid-body state

`car.kinematics()` (`envs/entities.py:148-156`) is a thin passthrough to the
Genesis physics entity (the `Car.__getattr__` at `entities.py:54` forwards any
unknown attribute straight to the underlying entity):

```python
def kinematics(self):
    e = self.entity
    return e.get_pos(), e.get_quat(), e.get_vel(), e.get_ang()
```

The crucial fact: **`get_vel()`/`get_ang()` are NOT finite-differenced from
positions.** Genesis is a full rigid-body integrator — linear and angular
velocity are primary state variables the solver carries (it integrates velocity
from forces, then position from velocity). You *read* velocity; you don't
reconstruct it. Reading gives the instantaneous world-frame state at 50 Hz
(`dt=0.01` × `decimation=2`).

### 2.2 Three tiers of derivation

| Tier | How it's obtained | Fields |
|------|-------------------|--------|
| **Measured** (sim state) | read directly | `base_pos`, `yaw`←quat, linear `vel`, `yaw_rate`=`ang[:,2]`, `up_z`←quat |
| **Same-step projection** (math on measured, no time delta) | rotate/geometry | `v_forward`/`v_lateral`, `lateral`, `half_width`, `heading_err`, `wp_idx` |
| **Finite-differenced** (this step vs last) | Δ across steps | `d_progress = new_progress − progress_m`, `laps` |

The only "delta from where it was" quantity is **arc-length progress**
(`base_env.py:348-352`) — because progress isn't a physical state variable, it's
track-relative bookkeeping you *must* difference (with wrap-around handling so
crossing the finish line doesn't read as teleporting backward). Velocity is never
differenced — the engine already integrates it as first-class state.

### 2.3 `yaw` is born from the quaternion

Genesis reports orientation as a unit quaternion `q = (w, x, y, z)`. `yaw` isn't
stored — it's extracted every step (`rules.py:13-24`):

```python
w, x, y, z = q.unbind(dim=1)
yaw = torch.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))   # = atan2(R₁₀, R₀₀)
```

That's the standard quaternion→Euler yaw: the heading of the car's forward axis
about the world vertical. `up_z` (`rules.py:40-52`) is the *same* quaternion,
a different matrix entry — `1 − 2(x²+y²) = R₂₂`, the world-z component of the
body-up axis (`1` upright, `<0.3` flipped). Pitch/roll live in `up_z`; `yaw`
ignores them. **In `ClassicFeatures`, the quaternion enters *only* through `yaw`;
`up_z` is not a feature — it exists solely for flip termination (`rules.is_flipped`).**

### 2.4 `yaw`'s one job: rotate world ↔ body frame

`yaw` is an angle, so nothing uses it directly — the code immediately takes
`cos`/`sin` to build a 2D rotation. Every use is a variation of the world→body
rotation:

```
[ v_forward ]   [  cos(yaw)  sin(yaw) ] [ vx ]
[ v_lateral ] = [ -sin(yaw)  cos(yaw) ] [ vy ]
```

- **velocity** into the body frame (`base_env.py:333-335`),
- **look-ahead waypoint offsets** into the body frame (`features.py:130,135-137`)
  — this is why `ClassicFeatures` reports "2 m ahead, 0.3 m left" instead of raw
  world XY,
- **heading error** by comparing `yaw` to the track tangent
  (`heading_err = wrap(yaw − track_yaw − rev·π)`, `base_env.py:347`),
- and *inverted* at spawn — a chosen yaw is written back into a quaternion
  (`qpos[3]=cos(yaw/2), qpos[6]=sin(yaw/2)`, `base_env.py:394-395`).

### 2.5 `dir_sign`: the companion convention

At reset a coin-flip may reverse the driving direction: `yaw += π` **and**
`dir_sign = −1` (`base_env.py:386-387`). From then on every track-relative
quantity is direction-corrected so a reversed car still sees `heading_err ≈ 0` and
accumulates *positive* progress: `heading_err` subtracts `rev·π`; `lateral` is
multiplied by `dir_sign` (`features.py:144,217`); look-ahead walks backwards
(`track.py:234`); curvature probes backwards (`track.py:252`); `d_progress` is
signed by `dir_sign`. **Every new feature that reads a track-frame quantity must
apply `dir_sign` too** — the planned signal bus (Part K) stores the corrected
value once to remove this footgun.

---

## 3. Per-channel derivation — `ClassicFeatures` (28-dim)

Built at `features.py:128-153`; width `8 + 2·lookahead_k` (`features.py:118`,
default `lookahead_k=10` → 28). Normalization divisors come from
`physics/limits.py`.

| Ch | Feature | Formula | Root sensor | Uses yaw? |
|----|---------|---------|-------------|-----------|
| 0 | `v_forward/max_speed` | `vx·cy + vy·sy`, ÷ `max_speed` | `get_vel()` | **yes** (world→body) |
| 1 | `v_lateral` | `−vx·sy + vy·cy` | `get_vel()` | **yes** |
| 2 | `yaw_rate/norm` | `ang[:,2] / YAW_RATE_NORM` | `get_ang()` | **no** — directly measured |
| 3 | `lateral·dir_sign/half_width` | `localize(pos)`, clamp `half_width≥0.1` | `get_pos()` | **no** — geometry |
| 4 | `sin(heading_err)` | `sin(wrap(yaw − track_yaw − rev·π))` | quat + pos | **yes** |
| 5 | `cos(heading_err)` | `cos(...)` | quat + pos | **yes** |
| 6–7 | `last_action[2]` | `env.actions` | — | **no** — policy's own output |
| 8–17 | `lookahead_rel_x[10]/scale` | rotate `(la_pts − base_pos)` into body x | pos + `track` | **yes** |
| 18–27 | `lookahead_rel_y[10]/scale` | body y of the same offsets | pos + `track` | **yes** |

The look-ahead points come from `track.lookahead_points_m(...)`: `k` centerline
samples at fixed arc-length meters ahead (`lookahead_spacing_m`, default 0.45 m
→ a 4.5 m horizon on **every** track, interpolated between waypoints), rotated
into the body frame. Before ADR 0003 the walk was index-based, so the horizon
in meters varied ~48× with each track's waypoint spacing.

---

## 4. Per-channel derivation — `PerceptionFeatures` (29-dim)

Built at `features.py:213`; a **different observation**, not an extension of
Classic. Two blocks:

- **CNN targets** (`features.py:216-224`) — what a camera *could* tell you:
  `lateral`, `heading/π`, `speed/MAX_SPEED`, `yaw_rate/norm`, sideslip
  `β = atan2(v_lateral, v_forward)/BETA_NORM`, and `curvature_ahead` at each
  configured horizon (`track.curvature_ahead`, `track.py:237`). These are the
  channels a deployed CNN is supervised to regress from pixels.
- **Policy-only error/history channels** (`features.py:226-239+`) — action-
  conditioned residuals vs a fixed nominal bicycle model (`speed_err`,
  `steer_err`, `yaw_err`) plus rolled histories of previous actions/errors, using
  the buffers allocated in `__init__` (`features.py:177-180`).

`cnn_target_slice` (`features.py:209`) marks the leading `(0, targets)` range — the
"pixel-observable subset." This is exactly the overlap the signal bus (Part K)
generalizes: the subset a pixel actor can learn, vs the full privileged vector a
critic can read.

---

## 5. The `FeatureSet` contract

From the base class (`features.py:66-102`):

```python
class FeatureSet:
    def __init__(self, env, params: dict): ...   # store env; allocate history buffers
    @property
    def dim(self) -> int: ...                     # vector width for THIS instance
    def compute(self) -> torch.Tensor: ...        # (N, dim) — called every step
    def reset(self, env_ids) -> None: ...         # clear per-env history on respawn
    @classmethod
    def dim_for(cls, *, lookahead_k, params): ... # width WITHOUT building an env
    @classmethod
    def layout_for(cls, *, lookahead_k, params): ...  # human-readable channel names
    cnn_target_slice: tuple[int, int] | None      # pixel-predictable channel range
```

Two subtleties:

- **`dim_for`/`layout_for` are classmethods** because the builder, model cards,
  and rollout metas need the width *before* an env exists (`feature_dim`/
  `feature_layout`, `features.py:34-63`; rollout meta at `datasets/rollout.py:171`).
- **`reset(env_ids)`** only matters with history. `PerceptionFeatures` keeps
  buffers (`features.py:177-180`) and clears them; `ClassicFeatures` is stateless
  and inherits the no-op.

---

## 6. How to add a new feature vector

### 6.1 The recipe

1. Subclass `FeatureSet` in `features.py`.
2. Implement `dim_for` (static width), `layout_for` (channel names), the `dim`
   property (delegates to `dim_for`), and `compute()` (reads `env.<field>`,
   normalizes, `cat`/`stack`s into `(N, dim)`).
3. If you need a *rate* the sim doesn't track as state (e.g. Δ lateral, jerk),
   allocate a `self._prev_*` buffer in `__init__` and clear it in `reset`.
4. Select it via config — no registry, no edit to `base_env`.

### 6.2 Minimal worked example

A compact 6-dim kinematics-only vector:

```python
class MinimalFeatures(FeatureSet):
    """Compact kinematic-only state: no look-ahead, no history."""

    @property
    def dim(self) -> int:
        return self.dim_for(lookahead_k=self.env.lookahead_k, params=self.params)

    @classmethod
    def dim_for(cls, *, lookahead_k: int, params: dict) -> int:
        return 6

    @classmethod
    def layout_for(cls, *, lookahead_k: int, params: dict) -> str:
        return ("v_forward/max_speed, v_lateral, yaw_rate/norm, "
                "lateral/half_width, sin(heading_err), cos(heading_err)")

    def compute(self) -> torch.Tensor:
        env = self.env
        return torch.stack([
            env.v_forward / env.cfg["action"]["max_speed"],
            env.v_lateral,
            env.yaw_rate / YAW_RATE_NORM,
            env.lateral * env.dir_sign / env.half_width.clamp(min=0.1),
            torch.sin(env.heading_err),
            torch.cos(env.heading_err),
        ], dim=1)
```

### 6.3 History example (finite-differencing your own signal)

If you want the *rate of change* of lateral offset:

```python
def __init__(self, env, params):
    super().__init__(env, params)
    self._prev_lateral = torch.zeros(env.num_envs, device=env.device)

def compute(self):
    env = self.env
    d_lateral = env.lateral - self._prev_lateral
    self._prev_lateral = env.lateral.clone()
    ...  # include d_lateral as a channel

def reset(self, env_ids):
    self._prev_lateral[env_ids] = 0.0   # else a respawn reads a huge spurious rate
```

This is the same pattern `PerceptionFeatures` uses for its action-error histories.

### 6.4 Wiring it in (dependency injection)

Select through config — `obs.feature_set` takes the **class**, `obs.feature_params`
is the `params` dict handed to `__init__` (`configs/schema.py:60-68`):

```python
cfg = get_env_cfg()
cfg["obs"]["feature_set"] = MinimalFeatures     # None -> ClassicFeatures
cfg["obs"]["feature_params"] = {}               # e.g. {"horizons": (1.0, 3.0)} for Perception
```

The env instantiates whatever class you pass (`base_env.py:224`); `num_state_obs`
follows from `feature_set.dim`, so the network input width tracks automatically.

---

## 7. Gotchas

- **Normalize.** Divide by the `physics/limits.py` constants (`MAX_SPEED`,
  `YAW_RATE_NORM`, `BETA_NORM`, `CURVATURE_NORM`, ...) so channels sit ~`[-1, 1]`;
  the policy trains far worse on raw-magnitude inputs.
- **Apply `dir_sign`** to any track-frame quantity (§2.5), or reversed-direction
  episodes will read mirrored features.
- **Clear history in `reset`**, or a respawn produces a spurious spike on the
  first step of the new episode.
- **Set `cnn_target_slice`** if the vector is meant for a vision/perception setup
  and some leading channels are pixel-predictable — the rollout collector records
  it into the dataset meta (`datasets/rollout.py:171-175`).
- **`dim_for` must be pure** in `lookahead_k`/`params` (no env access) — it's
  called before any env exists.

---

See also: `rewards-actions.md` (the reward reads the same `_post_physics` fields),
`tracks.md` (how `localize`/`lookahead`/`curvature_ahead` turn a raw `(x,y)` into
`lateral`/`track_yaw`/`progress`), and `REFACTOR_PLAN.md` Part K (the planned
signal bus that unifies features, reward, and cost over one vocabulary).
