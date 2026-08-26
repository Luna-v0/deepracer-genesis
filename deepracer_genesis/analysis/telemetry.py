"""Per-step trajectory telemetry: the SIM_TRACE-equivalent log.

The community ``deepracer-utils`` analysis stack is built on one artifact —
a per-step, per-episode DataFrame of poses, actions, and rewards. This
simulator never persisted it (positions live only as GPU tensors); the
:class:`TelemetryRecorder` captures it during eval rollouts and flushes to
parquet inside the run directory. Column names follow the deepracer-utils
vocabulary where the concepts overlap, so anyone fluent in the community
notebooks is at home.

Positions are stored TRACK-LOCAL (the Part O tile offset of each env's
variant is subtracted at capture), because analysis wants the track frame —
world coordinates are meaningless across zoo tiles.
"""

from __future__ import annotations

import os

import numpy as np
import torch

#: parquet column order (schema v1)
COLUMNS = ("step", "env", "episode", "track", "x", "y", "yaw", "steer",
           "throttle", "speed", "reward", "progress_delta", "progress_m",
           "lateral", "half_width", "off_track", "done", "track_len", "dt")


def _np(t: torch.Tensor) -> np.ndarray:
    """One captured tensor's GPU→CPU hop.

    Args:
        t: Any per-env tensor read off the live env.

    Returns:
        The detached numpy copy.
    """
    return t.detach().cpu().numpy()


class TelemetryRecorder:
    """Capture per-step car telemetry from a live env during a rollout.

    Reads only tensors the env already computes each step (pose, post-DR
    actions, track-frame quantities, the ``v_forward`` signal) — one small
    GPU→CPU copy per step, no extra physics or render work.

    Attributes:
        env: The live ``DeepRacerEnv`` being recorded.
        rows: Number of steps captured so far.
    """

    def __init__(self, env) -> None:
        """Bind to a live env and cache its per-env static facts.

        Args:
            env: A built ``DeepRacerEnv`` (any modality with track-frame
                state; camera and feature envs both qualify).
        """
        self.env = env
        ev = env.track.variant_idx
        self._names = np.array([env.track.names[i] for i in ev.tolist()])
        self._offset = env.track.variant_offset[ev]            # (N, 2) gpu
        self._track_len = env.track.total_len_env.cpu().numpy()
        self._dt = float(env.dt)
        self._episode = np.zeros(env.num_envs, dtype=np.int64)
        self._frames: list[dict] = []
        self.rows = 0

    def step(self, reward: torch.Tensor, done: torch.Tensor,
             off_track: torch.Tensor, progress_delta: torch.Tensor) -> None:
        """Record one control step for every env.

        Call AFTER ``env.step(...)`` with that step's outcome tensors (the
        eval loop has them in hand); pose/action/track-frame state is read
        off the env directly.

        Args:
            reward: ``(N,)`` per-env step reward.
            done: ``(N,)`` per-env done flags (episode ids advance after).
            off_track: ``(N,)`` off-track (incl. flipped) flags this step.
            progress_delta: ``(N,)`` progress gained this step in metres
                (already lap-wrap-corrected by the env).
        """
        env = self.env
        pos = _np(env.base_pos[:, :2] - self._offset)
        done_np = _np(done).astype(bool)
        self._frames.append({
            "env": np.arange(env.num_envs),
            "episode": self._episode.copy(),
            "x": pos[:, 0],
            "y": pos[:, 1],
            "yaw": _np(env.yaw),
            "steer": _np(env.actions[:, 0]),
            "throttle": _np(env.actions[:, 1]),
            "speed": _np(env.signals["v_forward"]),
            "reward": _np(reward),
            "progress_delta": _np(progress_delta),
            "progress_m": _np(env.progress_m),
            "lateral": _np(env.lateral),
            "half_width": _np(env.half_width),
            "off_track": _np(off_track).astype(bool),
            "done": done_np,
        })
        self._episode += done_np.astype(np.int64)
        self.rows += 1

    def _static_columns(self) -> dict:
        """Columns known without capturing: step index and per-env facts.

        Returns:
            ``step``/``track``/``track_len``/``dt`` flat arrays, each of
            length ``rows * num_envs`` (env-major within each step).
        """
        n, t = self.env.num_envs, self.rows
        return {
            "step": np.repeat(np.arange(t), n),
            "track": np.tile(self._names, t),
            "track_len": np.tile(self._track_len, t).astype(np.float32),
            "dt": np.full(t * n, self._dt, dtype=np.float32),
        }

    def _captured_columns(self) -> dict:
        """Per-step captures stacked into flat columns.

        Returns:
            One flat array per captured key, float64 narrowed to float32
            (parquet size — nothing here needs double precision).
        """
        cols = {}
        for key in self._frames[0]:
            stacked = np.concatenate([f[key] for f in self._frames])
            if stacked.dtype == np.float64:
                stacked = stacked.astype(np.float32)
            cols[key] = stacked
        return cols

    def flush(self, path: str) -> str:
        """Write everything captured so far as one parquet file.

        Args:
            path: Destination ``.parquet`` path (parents are created).

        Returns:
            The written path.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        cols = {**self._static_columns(), **self._captured_columns()}
        table = pa.table({c: cols[c] for c in COLUMNS})
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        pq.write_table(table, path, compression="zstd")
        return path


def load_telemetry(path: str):
    """Load recorded telemetry as a pandas DataFrame.

    Args:
        path: One ``.parquet`` file, or a directory — every ``*.parquet``
            inside is concatenated with a ``source`` column (the file stem).

    Returns:
        The per-step DataFrame (one row per env per step).

    Raises:
        FileNotFoundError: If the path holds no telemetry.
    """
    import glob

    import pandas as pd

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"no telemetry parquet under {path}")
        parts = []
        for f in files:
            df = pd.read_parquet(f)
            df["source"] = os.path.splitext(os.path.basename(f))[0]
            parts.append(df)
        return pd.concat(parts, ignore_index=True)
    return pd.read_parquet(path)


def episode_agg(df):
    """Per-episode summary — the ``AnalysisUtils.simulation_agg`` port.

    Args:
        df: A telemetry DataFrame from :func:`load_telemetry`.

    Returns:
        One row per (track, env, episode): steps, ``time_s``, progress in
        metres and as a completion fraction of the lap, mean/max speed,
        total reward, off-track step count, and ``completed_lap``.
    """
    import pandas as pd

    def _one(g):
        progress = float(g["progress_delta"].sum())
        track_len = float(g["track_len"].iloc[0])
        return pd.Series({
            "steps": len(g),
            "time_s": len(g) * float(g["dt"].iloc[0]),
            "progress_total_m": progress,
            "completion": progress / max(track_len, 1e-9),
            "mean_speed": float(g["speed"].mean()),
            "max_speed": float(g["speed"].max()),
            "total_reward": float(g["reward"].sum()),
            "off_track_steps": int(g["off_track"].sum()),
            "completed_lap": progress >= 0.999 * track_len,
        })

    keys = ["track", "env", "episode"]
    return (df.groupby(keys, observed=True)
              .apply(_one, include_groups=False)
              .reset_index())
