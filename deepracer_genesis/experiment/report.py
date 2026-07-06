"""Reporter (plan section 5.4): grouped combination table + before/after
delta tables, regenerated from stored eval_record.json files — no re-training.
"""

from __future__ import annotations

import csv
import glob
import math
import os
import statistics
from collections import defaultdict

from .evaluator import EvalRecord

# Within an ablation_group, the baseline variant is the first one matching a
# hint (plan section 5.3 pairings); lexicographic first otherwise.
BASELINE_HINTS = ("no_dr", "none", "baseline", "end2end", "reward_penalty",
                  "feature", "madrona")

TABLE_METRICS = ("completion_rate", "lap_time_s", "mean_progress_m",
                 "offtrack_rate", "mean_return", "mean_cost",
                 "cost_violation_rate")


def load_records(root: str = "runs") -> list[EvalRecord]:
    """Every eval_record.json under `root` (one per finished run)."""
    return [EvalRecord.load(p)
            for p in sorted(glob.glob(os.path.join(root, "**", "eval_record.json"),
                                      recursive=True))]


def spec_axes(spec: dict) -> dict:
    """The combination axes of plan section 5.1, derived from a spec dump."""
    env, policy = spec["env"], spec["policy"]
    obs_dr, action_dr = spec["obs_dr"], spec["action_dr"]
    parts = []
    if obs_dr["image_aug"] or obs_dr["camera_jitter"]:
        parts.append("obs")
    if obs_dr["physics"]:
        parts.append("physics")
    if any(action_dr[k] for k in ("steer_noise", "speed_noise", "delay_steps")):
        parts.append("action")
    dr = "full" if len(parts) == 3 else ("+".join(parts) or "none")
    return {
        "modality": env["modality"],
        "render": env["render"],
        "algorithm": spec["algorithm"]["kind"],
        "asymmetry": ("asymmetric" if set(policy["critic_keys"]) != set(policy["actor_keys"])
                      else "symmetric"),
        "encoder": spec["encoder"]["kind"],
        "dr_profile": dr,
    }


def _agg(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def _fmt(mean: float, std: float, digits: int = 3) -> str:
    if isinstance(mean, float) and math.isnan(mean):
        return "-"
    return f"{mean:.{digits}g} ± {std:.{digits}g}" if std else f"{mean:.{digits}g}"


def grouped_rows(records: list[EvalRecord]) -> list[dict]:
    """One row per combination cell, metrics mean ± std over seeds/records."""
    cells: dict[tuple, list[EvalRecord]] = defaultdict(list)
    for r in records:
        cells[tuple(sorted(spec_axes(r.spec).items()))].append(r)
    rows = []
    for key, recs in sorted(cells.items()):
        row = dict(key)
        row["n_runs"] = len(recs)
        row["seeds"] = sorted({r.seed for r in recs})
        for m in TABLE_METRICS:
            vals = [r.metrics[m] for r in recs
                    if m in r.metrics and not math.isnan(r.metrics[m])]
            if vals:
                row[m] = _agg(vals)
        rows.append(row)
    return rows


def delta_rows(records: list[EvalRecord]) -> dict[str, dict]:
    """Per ablation_group: baseline pick + (treatment - baseline) deltas."""
    groups: dict[str, dict[str, list[EvalRecord]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.ablation_group and r.variant:
            groups[r.ablation_group][r.variant].append(r)

    out = {}
    for group, variants in sorted(groups.items()):
        if len(variants) < 2:
            continue
        names = sorted(variants)
        baseline = next((v for h in BASELINE_HINTS for v in names if h in v), names[0])
        base_metrics = {m: _agg([r.metrics[m] for r in variants[baseline]
                                 if m in r.metrics and not math.isnan(r.metrics[m])])
                        for m in TABLE_METRICS
                        if any(m in r.metrics for r in variants[baseline])}
        deltas = {}
        for v in names:
            if v == baseline:
                continue
            d = {}
            for m, (bmean, _) in base_metrics.items():
                vals = [r.metrics[m] for r in variants[v]
                        if m in r.metrics and not math.isnan(r.metrics[m])]
                if vals:
                    vmean, vstd = _agg(vals)
                    d[m] = (vmean - bmean, vstd)
            deltas[v] = d
        out[group] = {"baseline": baseline, "base_metrics": base_metrics,
                      "deltas": deltas,
                      "n": {v: len(rs) for v, rs in variants.items()}}
    return out


def build_report(root: str = "runs", out_md: str | None = None,
                 out_csv: str | None = None) -> str:
    records = load_records(root)
    out_md = out_md or os.path.join(root, "report.md")
    out_csv = out_csv or os.path.join(root, "report.csv")

    lines = ["# Experiment report", "",
             f"{len(records)} run record(s) under `{root}/`.", "",
             "## Combination table", ""]
    rows = grouped_rows(records)
    axes = ["modality", "render", "algorithm", "asymmetry", "encoder", "dr_profile"]
    present = [m for m in TABLE_METRICS if any(m in r for r in rows)]
    lines.append("| " + " | ".join(axes + ["runs"] + present) + " |")
    lines.append("|" + "---|" * (len(axes) + 1 + len(present)))
    for row in rows:
        cells = [str(row[a]) for a in axes] + [str(row["n_runs"])]
        cells += [_fmt(*row[m]) if m in row else "-" for m in present]
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## Before/after (ablation pairs)", ""]
    for group, info in delta_rows(records).items():
        lines.append(f"### {group}  (baseline: `{info['baseline']}`)")
        lines.append("")
        metrics = sorted(info["base_metrics"])
        lines.append("| variant | " + " | ".join(f"Δ {m}" for m in metrics) + " |")
        lines.append("|" + "---|" * (len(metrics) + 1))
        for v, d in sorted(info["deltas"].items()):
            cells = [f"{d[m][0]:+.3g} ± {d[m][1]:.2g}" if m in d else "-"
                     for m in metrics]
            lines.append(f"| {v} | " + " | ".join(cells) + " |")
        lines.append("")

    md = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
    with open(out_md, "w") as f:
        f.write(md)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(axes + ["n_runs", "seeds"]
                   + [x for m in present for x in (m, m + "_std")])
        for row in rows:
            flat = [row[a] for a in axes] + [row["n_runs"], ";".join(map(str, row["seeds"]))]
            for m in present:
                mean, std = row.get(m, (float("nan"), float("nan")))
                flat += [mean, std]
            w.writerow(flat)
    return md
