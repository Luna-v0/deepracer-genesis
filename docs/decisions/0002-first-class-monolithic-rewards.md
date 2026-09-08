# 0002 — Rewards: a fn may return the reward itself, not only weighted terms

- Status: accepted
- Date: 2026-09-08

## Context

A reward fn was typed `env -> {term: (N,) tensor}` and `mdp.compute_reward`
reduced it to `sum(scale * terms[name])` over `reward_scales`. There was no
declared way to say "this torch expression IS the reward": the escape hatch
(`{"total": expr}` with `scales={"total": 1.0}`) worked but was unadvertised,
untested, and lossy — a single-term reward collapsed the TensorBoard
decomposition to one row, its constants had to live in closures (invisible to
the content hash, `eval_record.json`, and HPO), and the K.5 learnability check
silently skipped any fn without declared `reads`. Every reward written for the
round-3 studies bent itself into the term-vocabulary + scales shape as a
result. (PROBLEMS.md P12.)

## Decision

Widen the contract to `env -> dict[str, Tensor] | Tensor` and make the
monolithic form lossless:

1. **Verbatim rewards.** A bare `(N,)` tensor is the step reward, copied into
   `rew_buf` unmodified and logged as `Episode/rew_total`. With empty scales, a
   dict's reserved `total` term means the same thing.
2. **Diagnostic channels.** A `_`-prefixed term accumulates into
   `episode_sums` (logged as `Episode/diag_<name>`) but never into `rew_buf`,
   so a monolithic reward keeps a per-term breakdown.
3. **`RewardShaping(params=...)` → `EnvSpec.reward_params` →
   `env.reward_params`.** Closure constants become spec fields: content-hashed
   (only when non-empty — empty is pruned from `id()` so every pre-existing
   hash and run dir stays valid, pinned by golden ids in
   `tests/test_custom_rewards.py`), recorded, and searchable.
4. **No silent K.5 waiver.** `spec.validate()` warns when a custom fn declares
   no `reads` — an opaque reward is exactly the case that should be checked.

Every ambiguous combination raises at the first step: bare tensor + scales,
named terms + no scales, unscaled non-`_` terms beside `total`, a scale naming
a `_` term. The custom-fn-needs-scales check moved from env construction to
the first `compute_reward` call — the earliest point the return shape is known.

## Alternatives

- **Params as a second fn argument** (`fn(env, params)`) — breaks every
  existing reward's signature; `env.reward_params` keeps `env -> reward`.
- **Hash `reward_params` unconditionally** — invalidates all existing run-dir
  ids for a default-valued field; pruning-when-empty gets hash visibility only
  when the feature is used.
- **Make `reads` mandatory (raise)** — breaks the studies repo's existing
  undeclared fns for a check that is advisory by design; a warning is loud
  enough.

## Consequences

- "Just return the reward" is now the documented path
  (`docs/concepts/rewards-actions.md`); the weighted-terms path is
  byte-identical for all existing fns.
- Specs from studies with undeclared custom rewards now emit a `UserWarning`
  at validate (intended).
- The `-10` terminal crash penalty is still added outside `compute_reward`
  (P11, open): a verbatim reward is still not the whole effective reward.
