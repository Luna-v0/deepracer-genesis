"""The "prove" suite: assert a knob does something real, on its declared axis.

Every check runs at a shared teleported pose so any difference IS the knob
(the ``verify_nyx_env_map`` technique, generalized). Checks per knob kind:

- offline (image aug / world colour / pixel noise): has-effect at the knob's
  own stage AND at the policy ``stack`` endpoint; axis conformance (per-step
  resample / per-episode constancy / per-env difference); range sanity
  (NaN + clip fraction at the endpoints); sampler coverage.
- mount: live re-roll -> frames move, per env.
- env-map: cross-env sky difference at one pose (session must be built with
  the knob: rebuild-class).
- physics: cross-env divergence of the knob's DECLARED signals under
  identical actions from identical poses (session built with the DR on).
- action noise: post-DR actions differ across envs for one shared command.
- knobs the session's renderer/modality does NOT support: the check is that
  ``spec.validate()`` REFUSES them (declared-and-enforced beats silent).

Thresholds are defaults tuned on the reinvent scene; override per call.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import torch

from .knobs import REGISTRY, EditorKnob, get
from .pipeline import apply_world_color, replay_stages, replay_temporal
from .rng import seeded
from .session import EditorSession

THRESH = {"effect": 1e-3, "per_env": 0.01, "per_step": 1e-4, "same": 1e-3,
          "signal": 1e-3}

# knobs whose SUGGESTED range is legitimately mild get a lower has-effect bar
# (blur's sigma<=0.5 builds a 3x3 kernel that is 62-99% identity, and ~1 in 10
# draws skips it outright) — a real finding the verdict surfaces as WEAK
THRESH_OVERRIDES = {"blur": 1e-5}

_RENDERER_NAME = {"batch": "madrona", "nyx": "nyx", "rasterizer": "rasterizer"}


@dataclass
class Verdict:
    """One check's outcome.

    Attributes:
        check: Check name (``has_effect``, ``axis``, ``range_sanity``,
            ``coverage``, ``declared_refusal``, ``skipped``).
        knob: Knob name.
        passed: Whether the check passed.
        detail: One-line human explanation.
        metrics: The numbers behind the verdict.
    """

    check: str
    knob: str
    passed: bool
    detail: str
    metrics: dict = field(default_factory=dict)


def _pair_max(frames: torch.Tensor) -> float:
    """Max mean-absolute difference over all env pairs.

    Args:
        frames: ``(N, ...)`` batch.

    Returns:
        ``max_{i<j} mean|frames[i] - frames[j]|``.
    """
    n = frames.shape[0]
    return max((frames[i] - frames[j]).abs().mean().item()
               for i in range(n) for j in range(i + 1, n)) if n > 1 else 0.0


def _range_dr(knob: EditorKnob) -> dict:
    """The full-range DR dict for a knob (real sampling, not a sweep point).

    Args:
        knob: An offline knob.

    Returns:
        The dr dict activating the knob across its whole suggested space.
    """
    s = knob.space
    if knob.kind == "aug_range":
        return {"image_aug": {knob.name: (s.lo, s.hi)}}
    if knob.kind == "aug_scalar":
        hi = s.hi if hasattr(s, "hi") else s.m
        return {"image_aug": {knob.name: int(hi) if knob.name == "latency_steps"
                              else float(hi)}}
    if knob.kind == "world_color":
        return {"world_color": s.hi}
    return {"pixel_noise": s.hi}                       # pixel_noise


def session_renderer(session: EditorSession) -> str:
    """The catalog-vocabulary renderer name of a live session.

    Args:
        session: The live session.

    Returns:
        ``"madrona"``, ``"nyx"``, or ``"rasterizer"``.
    """
    return _RENDERER_NAME[session.env.cfg["vision"].get("vision_renderer", "batch")]


# ------------------------------------------------------------------ checks
def _prove_offline(session: EditorSession, knob: EditorKnob,
                   seed: int) -> list[Verdict]:
    """Offline-knob checks: has-effect, axis, range sanity, coverage."""
    out: list[Verdict] = []
    raw = session.raw()
    dr = _range_dr(knob)
    stage = {"aug_range": "image_aug", "aug_scalar": "image_aug",
             "world_color": "world_color", "pixel_noise": "pixel_noise"}[knob.kind]
    base = replay_stages(raw, {}, frame_stack=4, seed=seed)
    on = replay_stages(raw, dr, frame_stack=4, seed=seed)

    if knob.name in ("latency_steps", "frame_drop"):
        # temporal knobs act on a SEQUENCE; a static instant is identity by
        # definition — prove on a synthetic moving sequence instead
        frames = [torch.roll(raw, i * 3, dims=3) for i in range(10)]
        seen = replay_temporal(frames, dr, seed=seed)
        if knob.name == "latency_steps":
            k = dr["image_aug"]["latency_steps"]
            ok = all(torch.equal(seen[t], frames[t - k]) for t in range(k, 10))
            out.append(Verdict("has_effect", knob.name, ok,
                               f"policy frames lag the render by exactly {k} steps"
                               if ok else "latency did not delay frames",
                               {"latency_steps": k}))
            out.append(Verdict("axis", knob.name, ok,
                               "per-step history-dependent (delayed sequence)",
                               {}))
        else:
            # drops are PER ENV: count env-steps whose frame repeats
            repeats = sum(
                int(((seen[t] - seen[t - 1]).abs()
                     .flatten(1).amax(dim=1) == 0).sum())
                for t in range(1, 10))
            total = 9 * raw.shape[0]
            ok = repeats > 0
            out.append(Verdict("has_effect", knob.name, ok,
                               f"{repeats}/{total} env-steps repeated the "
                               f"previous frame at "
                               f"drop={dr['image_aug']['frame_drop']}",
                               {"repeats": repeats, "total": total}))
            out.append(Verdict("axis", knob.name, ok,
                               "per-step, per-env stochastic frame repetition",
                               {}))
        return out

    # stochastic knobs (a blur draw can skip entirely, a cutout coin can miss
    # every env): take the max effect over a few seeds so one quiet draw
    # doesn't fail a live knob — while a truly inert knob still reads 0
    d_stage = d_stack = 0.0
    for s in range(seed, seed + 5):
        on_s = replay_stages(raw, dr, frame_stack=4, seed=s)
        d_stage = max(d_stage, (on_s[stage] - base[stage]).abs().mean().item())
        d_stack = max(d_stack, (on_s["stack"] - base["stack"]).abs().mean().item())
    theta = THRESH_OVERRIDES.get(knob.name, THRESH["effect"])
    ok = d_stage > theta and d_stack > theta
    weak = ok and d_stage < THRESH["effect"]
    out.append(Verdict("has_effect", knob.name, ok,
                       f"max |Δ| over 5 seeds: {d_stage:.5f} at {stage}, "
                       f"{d_stack:.5f} at the policy stack (θ={theta})"
                       + (" — WEAK: real but marginal at the suggested range"
                          if weak else ""),
                       {"d_stage": d_stage, "d_stack": d_stack}))

    # axis: per-step resample vs per-episode constancy, plus per-env spread
    # (max over a few seeds — one quiet stochastic draw must not fail a live
    # knob; per-env spread reuses the strongest draw for the same reason)
    step_delta, env_delta = 0.0, _pair_max(on[stage])
    for s in range(seed, seed + 3):
        with seeded(s, raw.device):
            a = replay_stages(raw, dr)[stage]
            b = replay_stages(raw, dr)[stage]
        step_delta = max(step_delta, (a - b).abs().mean().item())
        env_delta = max(env_delta, _pair_max(a))
    if knob.kind == "world_color":
        with seeded(seed, raw.device):
            from ...randomization.visual import sample_world_color
            color = sample_world_color(raw.shape[0], dr["world_color"], raw.device)
        same = (apply_world_color(raw, dr["world_color"], color=color)
                - apply_world_color(raw, dr["world_color"], color=color)
                ).abs().max().item()
        ok = step_delta > THRESH["per_step"] and same < 1e-6
        detail = (f"palette constant while held (Δ={same:.2e}), changes on "
                  f"resample (Δ={step_delta:.4f}) — per-episode confirmed")
    else:
        step_theta = THRESH_OVERRIDES.get(knob.name, THRESH["per_step"])
        ok = step_delta > step_theta
        detail = (f"two consecutive draws differ (Δ={step_delta:.5f}) — "
                  f"per-step confirmed")
    env_theta = THRESH_OVERRIDES.get(knob.name, THRESH["per_env"] / 10)
    ok = ok and env_delta > env_theta
    out.append(Verdict("axis", knob.name, ok,
                       detail + f"; per-env spread {env_delta:.5f}",
                       {"step_delta": step_delta, "env_delta": env_delta}))

    # range sanity: endpoints — NaN scan + clip fraction
    s = knob.space
    lo, hi = (0.0, s.m) if hasattr(s, "m") else (s.lo, s.hi)
    from .knobs import dr_for_value
    metrics, bad = {}, False
    for tag, v in (("lo", lo), ("hi", hi)):
        img = replay_stages(raw, dr_for_value(knob, v), seed=seed)["image_aug" if
                            "aug" in knob.kind else stage]
        nan = bool(torch.isnan(img).any())
        clip = ((img <= 0) | (img >= 1)).float().mean().item()
        metrics[tag] = {"value": v, "nan": nan, "clip_frac": round(clip, 4)}
        bad = bad or nan
    out.append(Verdict("range_sanity", knob.name, not bad,
                       f"no NaN across [{lo}, {hi}]; clip fractions "
                       f"lo={metrics['lo']['clip_frac']} hi={metrics['hi']['clip_frac']}"
                       if not bad else "NaN at a range endpoint", metrics))

    # coverage: the sampler actually spans the declared space
    if hasattr(s, "sample"):
        with seeded(seed, raw.device):
            draws = s.sample(1000, raw.device).float()
        if hasattr(s, "m"):
            in_b = bool((draws.abs() <= s.m + 1e-6).all())
            span = (draws.max() - draws.min()).item() / max(2 * s.m, 1e-9)
        else:
            in_b = bool(((draws >= s.lo - 1e-6) & (draws <= s.hi + 1e-6)).all())
            span = (draws.max() - draws.min()).item() / max(s.hi - s.lo, 1e-9)
        ok = in_b and (span > 0.9 or s.hi == s.lo)
        out.append(Verdict("coverage", knob.name, ok,
                           f"1000 draws within bounds={in_b}, span {span:.0%} "
                           f"of the declared range", {"span": span}))
    return out


def _prove_mount(session: EditorSession, knob: EditorKnob,
                 seed: int) -> list[Verdict]:
    """Mount-jitter checks: live re-roll moves frames, per env."""
    base = session.raw()
    hi = knob.space.m
    pitch, pos = (hi, 0.0) if knob.name == "camera_pitch_jitter" else (0.0, hi)
    session.reroll_mount(pitch, pos, seed=seed)
    after = session.raw()
    moved = (after - base).abs().mean().item()
    spread = _pair_max(after)
    ok = moved > THRESH["effect"] and spread > THRESH["effect"]
    return [Verdict("has_effect", knob.name, ok,
                    f"re-roll moved frames |Δ|={moved:.4f}, per-env spread "
                    f"{spread:.4f} (per run: re-rolled, not per episode)",
                    {"moved": moved, "spread": spread})]


def _prove_env_map(session: EditorSession, knob: EditorKnob) -> list[Verdict]:
    """Env-map checks: per-env sky difference at one shared pose."""
    if not session.env.cfg["vision"].get("env_map"):
        return [Verdict("skipped", knob.name, False,
                        "session was built without vision.env_map — env maps "
                        "bake at scene.build (rebuild-class); build the prove "
                        "session with the knob enabled", {})]
    spread = _pair_max(session.raw())
    ok = spread > THRESH["per_env"]
    return [Verdict("has_effect", knob.name, ok,
                    f"cross-env sky spread {spread:.4f} at one pose "
                    f"(θ={THRESH['per_env']}; per-env, fixed for the run)",
                    {"spread": spread})]


def _prove_physics(session: EditorSession, knob: EditorKnob,
                   steps: int = 40) -> list[Verdict]:
    """Physics checks: declared signals diverge across envs, same actions."""
    leaf = knob.cfg_path.rsplit(".", 1)[1]
    rand = session.env.cfg.get("rand", {})
    val = rand.get(leaf)
    if isinstance(val, (tuple, list)):
        inert = tuple(val) in ((1.0, 1.0), (0.0, 0.0))
    else:
        inert = not val
    if not rand.get("randomize") or inert:
        return [Verdict("skipped", knob.name, False,
                        f"session built without rand.{leaf} active — physics "
                        "DR applies once per run at build (rebuild-class); "
                        "build the prove session with the knob enabled", {})]
    env = session.env
    act = torch.zeros(env.num_envs, 2, device=env.device)
    act[:, 1] = 0.5
    traces = {name: [] for name in knob.source.signals}
    for _ in range(steps):
        env.step(act)
        for name in traces:
            traces[name].append(env.signals[name].detach().float().clone())
    spreads = {name: torch.stack(t).std(dim=1).max().item()
               for name, t in traces.items()}
    best = max(spreads.values())
    ok = best > THRESH["signal"]
    return [Verdict("has_effect", knob.name, ok,
                    f"identical actions from one pose; max cross-env std of "
                    f"declared signals {spreads} (θ={THRESH['signal']}; "
                    "per run: drawn once at build)", {"spreads": spreads})]


def _prove_action(session: EditorSession, knob: EditorKnob,
                  seed: int) -> list[Verdict]:
    """Action-noise checks: post-DR actions differ across envs."""
    if knob.name == "delay_steps":
        return [Verdict("skipped", knob.name, False,
                        "delay ring buffer is sized at env construction — "
                        "build the session with cfg_updates={'action_dr': "
                        "{'delay_steps': k}} to prove it", {})]
    hi = knob.space.hi
    session.poke_action_dr(steer_noise=hi if knob.name == "steer_noise" else 0.0,
                           speed_noise=hi if knob.name == "speed_noise" else 0.0)
    act = torch.zeros(session.num_envs, 2, device=session.env.device)
    with seeded(seed, session.env.device):
        session.env.step(act)
    spread = _pair_max(session.env.actions)
    session.poke_action_dr()
    ok = spread > 1e-4
    return [Verdict("has_effect", knob.name, ok,
                    f"one shared command, post-DR actions spread {spread:.4f} "
                    f"across envs at noise={hi}", {"spread": spread})]


def prove_refusal(knob: EditorKnob, renderer: str) -> Verdict:
    """Check that an unsupported knob is REFUSED at build time.

    Args:
        knob: The knob the current renderer/modality does not support.
        renderer: Catalog renderer name the probe spec targets.

    Returns:
        A PASS verdict iff ``spec.validate()`` raises for the combination —
        "inert but loudly declared" is the honesty contract.
    """
    import warnings

    from ...experiment import stages as st
    from ...experiment.spec import SpecError
    from ...experiment.stages import (AsymmetricCameraPolicy,
                                      CameraEnvironment, PPO)
    stage_map = {
        "mount": lambda: st.DomainRandomizationCamera(camera_jitter=True),
        "env_map": lambda: st.DomainRandomizationTrackAppearance(
            env_map_tint=(0.35, 0.75)),
        "physics": lambda: st.DomainRandomizationPhysics(
            track_width=(0.9, 1.15)) if knob.name == "track_width_scale"
            else None,
    }
    maker = stage_map.get(knob.kind)
    stage = maker() if maker else None
    if stage is None:
        return Verdict("skipped", knob.name, False,
                       f"no refusal probe for kind {knob.kind!r}", {})
    env = CameraEnvironment(render="nyx" if renderer == "nyx" else "madrona",
                            backend="cpu" if renderer == "rasterizer" else "gpu",
                            num_envs=2)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (env >> stage >> AsymmetricCameraPolicy() >> PPO()).build()
        return Verdict("declared_refusal", knob.name, False,
                       f"knob is inert under {renderer} but validate() did "
                       "NOT refuse it — matrix drift", {})
    except SpecError as e:
        return Verdict("declared_refusal", knob.name, True,
                       f"inert under {renderer} AND refused at build: "
                       f"{str(e)[:90]}", {})


def prove_knob(session: EditorSession, name: str, *,
               seed: int = 0) -> list[Verdict]:
    """Run every applicable check for one knob against a live session.

    Args:
        session: The live session (teleported to a shared pose by the caller;
            :func:`run_suite` does this).
        name: Knob name.
        seed: Seed for all sampled checks.

    Returns:
        The verdict list (never empty).
    """
    knob = get(name)
    renderer = session_renderer(session)
    if "camera" not in knob.modalities:
        return [prove_refusal(knob, renderer)]
    if renderer not in knob.renderers:
        return [prove_refusal(knob, renderer)]
    if knob.liveness == "offline":
        return _prove_offline(session, knob, seed)
    if knob.kind == "mount":
        return _prove_mount(session, knob, seed)
    if knob.kind == "env_map":
        return _prove_env_map(session, knob)
    if knob.kind == "physics":
        return _prove_physics(session, knob)
    if knob.kind == "action":
        return _prove_action(session, knob, seed)
    return [Verdict("skipped", name, False,
                    f"static scene knob — compare via an A/B build", {})]


def run_suite(session: EditorSession, names: list[str], *, seed: int = 0,
              waypoint: int = 5, out_json: Optional[str] = None
              ) -> list[Verdict]:
    """Prove a list of knobs at one shared pose; optionally persist verdicts.

    Args:
        session: The live session.
        names: Knob names to prove.
        seed: Seed for all sampled checks.
        waypoint: Shared teleport waypoint.
        out_json: Optional JSON path for the verdict record.

    Returns:
        All verdicts, in knob order.
    """
    session.teleport(waypoint=waypoint)
    verdicts: list[Verdict] = []
    for name in names:
        verdicts.extend(prove_knob(session, name, seed=seed))
    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w") as f:
            json.dump([dataclasses.asdict(v) for v in verdicts], f, indent=2)
    return verdicts
