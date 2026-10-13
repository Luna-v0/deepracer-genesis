# 0005 — Control timing: a per-run decision period, a first-class frame interval, and actuation latency in milliseconds

- Status: accepted
- Date: 2026-10-12

## Context

The policy decides once per `sim.dt × decimation = 0.01 × 2 = 20 ms`, and no
spec could change that: `EnvSpec` had no timing field, so the builder always
took the config defaults. Every timing effect was quantized to whole control
steps: the legacy `delay_steps` delayed the command by `d × 20 ms` and fed the
*delayed* command to the features (which the car cannot know), `latency_steps`
aged the frame by `l × 20 ms`, and the camera stack held consecutive decisions,
so a 4-frame stack spanned only 60 ms (about 4 cm at 2 m/s).

The DeepRacer Evo decides at its camera rate (about 15 fps, 62–71 ms between
frames in the Sandha et al. measurements), with a capture-to-servo latency of
tens of milliseconds that varies decision to decision, and it occasionally
skips a decision when inference queues. A sim2real policy must train against a
decision period, a frame spacing and an actuator latency that match the car,
and the timing must be randomized so the policy tolerates the variation. The
design study (`2026-10-12-timing-dr-design.md` in the maintenance registry)
compared four designs; this ADR records option A, its phase 1.

## Decision

1. **`EnvSpec.control_dt_s`** (None = 0.02 s) is the policy decision period.
   The builder derives `decimation = control_dt_s / sim.dt` and refuses a
   non-integer ratio (`SpecError`); `sim.dt` stays 0.01 s, which is also the
   resolution of every timing knob. The episode cap stays in seconds, and the
   evaluator and telemetry already follow `env.dt`, so lap times and per-row
   `dt` are correct at any period.

2. **`EnvSpec.frame_interval_s`** (None = `control_dt_s`) is the spacing of
   the stacked camera frames, a whole multiple of the control period. The
   vision env keeps a ring of `(frame_stack − 1) · stride + 1` observed frames
   and stacks every `stride`-th one, oldest first; a respawned env primes the
   whole ring with its first frame. Latency (`latency_steps`) is applied
   before the push, so the B-001 ordering (camera → features) is intact.
   The interval is a deployment contract carried by every link of the
   pipeline: `meta.json` records `control_dt`, `frame_stride` and
   `frame_interval_s`; `RolloutDataset(stride=s)` stacks rows `s` steps apart
   and exposes `frame_interval_s`; `save_checkpoint(net, path,
   frame_interval_s=…)` stores it in the CNN checkpoint and
   `CNNPerceptionFeatures` refuses a checkpoint whose interval differs from
   the env's `dt × stride`; the ONNX model card's camera entry states
   `control_dt_s`, `frame_interval_s`, `frame_stride` and
   `stack_source = "policy_decisions"` (the stack holds the frames observed at
   decisions, every `frame_stride`-th one — not every camera frame).

3. **Actuation timing knobs** on `ActionDRSpec` / `DomainRandomizationActions`
   (catalog layer `actuation`, thesis group `dynamics`, cfg `action_dr.*`):
   `latency_ms` (per-episode base latency range in ms, placeholder 30–90),
   `latency_jitter_ms` (uniform `±` per-decision jitter, placeholder ±10) and
   `decision_drop` (per-decision drop probability, placeholder 0–0.1; the
   previous command holds one more period). `randomization/actuation_timing.py`
   implements a per-env pending-command queue that the substep loop drains:
   a command issued at substep `t` lands at `max(previous landing,
   t + ceil(L / sim.dt))`, so commands never reorder, and `car.drive` runs
   every substep with the command that is active. `env.actions` stays the
   *issued* command (the features and the reward penalties read it);
   `env.applied_actions` is what the physics follows and what telemetry's
   `steer`/`throttle` record. All draws come from a dedicated
   `torch.Generator` seeded from `spec.seed` (`cfg["sim"]["seed"]`), so
   switching the knobs on leaves spawn and image-aug draws unchanged, and
   nothing is drawn while they are off. A reset clears only the respawned
   envs' queues and redraws their base latency.

4. **`PPO.horizon_s` / `PPO.lam_horizon_s`** (None = per-step `gamma` /
   `gae_lambda`) are time horizons in seconds. The backend sets
   `gamma = exp(−T / horizon_s)` at the env's control period `T` (and the same
   for `lam`), so an HPO study searches the horizon independently of the
   decision rate. `exp(−T/H)` was chosen over `γ0^(T/T0)` because the horizon
   is then a plain number of seconds; `horizon_s = 1.99` reproduces 0.99 at
   50 Hz exactly, `2.0` gives 0.99005.

5. **`action_rate` is referenced to the 50 Hz period** when `control_dt_s`
   is set: the builder sets `reward.action_rate_dt_ref = 0.02` and the term is
   multiplied by `(0.02 / T)²`, because `‖Δa‖²` grows with `T²` for a smooth
   command. Unset, the term is unchanged.

6. **`Knob.group`** (`observation` / `dynamics` / `task`) tags every catalog
   knob with the thesis grouping, as data only (`by_group()` filters it).

All new fields default to None/0 and are **hash-exempt while unset**
(`HASH_EXEMPT_ENV_WHEN_NONE`, `HASH_EXEMPT_ACTION_DR_WHEN_UNSET`, and the
`horizon_s` keys only enter the PPO dict when set), so every existing spec id,
run dir and the knob-OFF step digests are bit-identical.

## Alternatives

- **Per-env clocks on a fine tick with sample-and-hold (option B)**: fully
  per-env periods, but it needs an SMDP-PPO subclass (per-transition `γ^τ`,
  masking), 5–7× the transitions, a render every tick, and per-row `dt` in
  eval and telemetry. Deferred until the Evo measurements show period
  variance that jitter plus drops cannot reproduce.
- **A shared period resampled per rollout (option C)**: cheap, but
  non-stationary batches and a period switch mid-episode. Phase 1 keeps the
  period per run; the hook for C is the single `control_period_s(spec.env)`
  read in `spec_to_train_cfg` plus `sim.cfg["sim"]["decimation"]` (see
  Consequences).
- **Timing values in the observation (Time-in-State)** and **a privileged
  critic with the true timing**: additive follow-ups; the camera actor's
  single-input ONNX contract keeps them off the actor for now.
- **Keeping `delay_steps` as the only latency**: it is quantized to the
  control period and feeds the features the delayed command; it stays
  bit-identical for existing runs but is refused together with the new knobs.
- **5 ms physics (`dt = 0.005`)** for finer latency resolution: twice the
  physics cost; 10 ms matches the Evo's measurement noise. Revisit with data.

## Consequences

- A car-faithful run sets `control_dt_s=0.07` (placeholder, Sandha's 62–71 ms
  window), `frame_interval_s` at the same value (one decision per frame), the
  three timing knobs, and `horizon_s` instead of `gamma`. The camera frame
  stack then covers 3× more motion per interval than at 50 Hz, so the
  `frame_stack` HPO must be re-run at the new spacing and the perception
  dataset collected (or stride-matched) at it.
- The car-side node (dr-onnx) must reproduce the model card's timing fields
  exactly: decide every `control_dt_s`, stack every `frame_stride`-th decision
  frame oldest first, prime by repetition.
- `ActuationTiming` sizes its queue from the latency range, so a latency
  above `(capacity − 2) × decimation` substeps is impossible by construction;
  the range must be set at build (the DR editor cannot poke it live).
- Telemetry's `steer`/`throttle` columns now record the applied command (the
  issued one unless the timing knobs delay it); the parquet schema is
  unchanged.
- The catalog page gains a `Group` column; `by_group("dynamics")` lists the
  knobs for a dynamics ablation.
- Phase 2 (a resampled decision period, per rollout or per env) must only
  (a) rebuild `decimation` and `env.dt` between rollouts, (b) recompute
  `gamma`/`lam` from `horizon_s` at the new period — the one place is
  `spec_to_train_cfg`'s `control_period_s` read — and (c) re-size the
  `ActuationTiming` queue for the smallest decimation (`max_pending`).
