"""Charts generated from an EvalRecord (Part N.4).

Isolated so the core training path carries no plotting dependency — matplotlib
is an optional extra, imported lazily here. Reads an ``EvalRecord`` (or its
dict dump) and writes PNGs:

- **Learning curves** from ``eval_history`` (metric vs frames) — the in-loop trend.
- **Per-track bars** from ``holdout`` (out-of-loop N.3 result) — one grouped bar
  chart per metric across the real-track array.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

#: metrics charted by default (present-in-data ones are used)
DEFAULT_METRICS = ("completion_rate", "lap_time_s", "offtrack_rate", "mean_speed_mps")


def _require_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")            # headless; no display needed
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:  # optional extra
        raise ImportError(
            "charts need matplotlib (an optional extra): pip install matplotlib"
        ) from e


def _as_dict(record) -> dict:
    """Accept an EvalRecord or its dict dump."""
    if hasattr(record, "eval_history"):
        return {"eval_history": record.eval_history, "holdout": record.holdout,
                "metrics": record.metrics}
    return {"eval_history": record.get("eval_history", []),
            "holdout": record.get("holdout", {}),
            "metrics": record.get("metrics", {})}


def learning_curves(record, out_dir: str,
                    metrics: Sequence[str] = DEFAULT_METRICS) -> list[str]:
    """Write metric-vs-frames learning curves from ``eval_history``.

    Args:
        record: An EvalRecord or its dict dump.
        out_dir: Directory to write PNGs into (created if missing).
        metrics: Metric names to plot (those present in the history).

    Returns:
        Paths of the PNGs written (empty if there is no periodic history).
    """
    data = _as_dict(record)
    history = data["eval_history"]
    if not history:
        return []
    plt = _require_mpl()
    os.makedirs(out_dir, exist_ok=True)
    frames = [h.get("frames", i) for i, h in enumerate(history)]
    written = []
    for m in metrics:
        ys = [h.get(m) for h in history]
        if not any(y is not None for y in ys):
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(frames, ys, marker="o")
        ax.set_xlabel("env frames")
        ax.set_ylabel(m)
        ax.set_title(f"learning curve — {m}")
        ax.grid(True, alpha=0.3)
        path = os.path.join(out_dir, f"learning_{m}.png")
        fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)
        written.append(path)
    return written


def per_track_bars(record, out_dir: str,
                   metrics: Sequence[str] = DEFAULT_METRICS) -> list[str]:
    """Write per-track bar charts from the out-of-loop ``holdout`` results.

    Args:
        record: An EvalRecord or its dict dump.
        out_dir: Directory to write PNGs into (created if missing).
        metrics: Metric names to chart (those present per track).

    Returns:
        Paths of the PNGs written (empty if there is no holdout section).
    """
    data = _as_dict(record)
    holdout = data["holdout"]
    if not holdout:
        return []
    plt = _require_mpl()
    os.makedirs(out_dir, exist_ok=True)
    tracks = list(holdout)
    written = []
    for m in metrics:
        vals = [holdout[t].get(m) for t in tracks]
        if not any(v is not None for v in vals):
            continue
        vals = [float(v) if v is not None else float("nan") for v in vals]
        fig, ax = plt.subplots(figsize=(max(6, len(tracks) * 1.2), 4))
        ax.bar(range(len(tracks)), vals, color="tab:blue")
        ax.set_xticks(range(len(tracks)))
        ax.set_xticklabels(tracks, rotation=30, ha="right")
        ax.set_ylabel(m)
        ax.set_title(f"holdout per-track — {m}")
        ax.grid(True, axis="y", alpha=0.3)
        path = os.path.join(out_dir, f"holdout_{m}.png")
        fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)
        written.append(path)
    return written


def render_charts(record, run_dir: str,
                  metrics: Sequence[str] = DEFAULT_METRICS) -> list[str]:
    """Render all eval charts under ``run_dir/charts/``.

    Args:
        record: An EvalRecord or its dict dump.
        run_dir: The run directory; charts land in ``run_dir/charts/``.
        metrics: Metric names to chart.

    Returns:
        Paths of every PNG written.
    """
    out_dir = os.path.join(run_dir, "charts")
    return (learning_curves(record, out_dir, metrics)
            + per_track_bars(record, out_dir, metrics))
