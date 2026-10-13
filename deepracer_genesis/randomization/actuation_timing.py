"""Actuation timing DR: per-env command latency in ms, per-decision jitter, drops.

A command issued at a decision lands at the physics substep where
``max(previous landing, issue + latency)`` elapses, so commands never reorder.
"""

from __future__ import annotations

import math

import torch


class ActuationTiming:
    """Per-env pending-command queue, advanced once per physics substep.

    Latency is resolved to whole substeps (``ceil(L / sim_dt)``), so the knob's
    resolution is the physics step. All draws come from a dedicated generator
    seeded from ``seed``: turning the knob on leaves every global RNG stream
    (spawns, image aug) untouched. ``reset`` must run before the first
    ``issue``; the constructor draws nothing.

    Attributes:
        num_envs: Number of parallel envs.
        latency_ms: ``(lo, hi)`` per-episode base latency range in ms.
        jitter_ms: Per-decision uniform jitter magnitude in ms (``±jitter``).
        drop: Per-decision probability that the command is dropped.
        sim_dt: Physics substep in seconds (the latency resolution).
        capacity: Queue slots per env (bounded by the latency range).
        latency_s: ``(N,)`` per-episode base latency in seconds.
        applied: ``(N, 2)`` command the physics currently follows.
        substep: Global substep counter (issue and landing times are in it).
    """

    def __init__(self, num_envs: int, *, latency_ms: tuple[float, float] | float,
                 jitter_ms: float = 0.0, drop: float = 0.0, sim_dt: float,
                 decimation: int, device, seed: int = 0) -> None:
        """Size the queue for the latency range and seed the generator.

        Args:
            num_envs: Number of parallel envs.
            latency_ms: Per-episode base latency range ``(lo, hi)`` in ms (a
                scalar means a fixed latency).
            jitter_ms: Per-decision uniform jitter magnitude in ms.
            drop: Per-decision drop probability in ``[0, 1]``.
            sim_dt: Physics substep in seconds.
            decimation: Smallest number of substeps per decision the run uses;
                sizes the queue (a per-rollout control period may shrink it).
            device: Torch device the state lives on.
            seed: Seed of the dedicated generator.

        Raises:
            ValueError: On a negative or inverted latency range, a negative
                jitter, or a drop probability outside ``[0, 1]``.
        """
        lo, hi = ((float(latency_ms), float(latency_ms))
                  if isinstance(latency_ms, (int, float))
                  else (float(latency_ms[0]), float(latency_ms[1])))
        if lo < 0 or hi < lo:
            raise ValueError(f"latency_ms must be 0 <= lo <= hi; got {(lo, hi)}")
        if jitter_ms < 0:
            raise ValueError(f"latency_jitter_ms must be >= 0; got {jitter_ms}")
        if not 0.0 <= drop <= 1.0:
            raise ValueError(f"decision_drop must be in [0, 1]; got {drop}")
        self.num_envs = int(num_envs)
        self.latency_ms = (lo, hi)
        self.jitter_ms = float(jitter_ms)
        self.drop = float(drop)
        self.sim_dt = float(sim_dt)
        self._device = torch.device(device)
        self._gen = torch.Generator(device=self._device)
        self._gen.manual_seed(int(seed))
        self.capacity = max_pending(hi, self.jitter_ms, self.sim_dt, decimation)
        n, q, dev = self.num_envs, self.capacity, self._device
        self.latency_s = torch.full((n,), lo / 1000.0, device=dev)
        self.applied = torch.zeros(n, 2, device=dev)
        self.substep = 0
        self._q_time = torch.zeros(n, q, dtype=torch.long, device=dev)
        self._q_act = torch.zeros(n, q, 2, device=dev)
        self._head = torch.zeros(n, dtype=torch.long, device=dev)
        self._len = torch.zeros(n, dtype=torch.long, device=dev)
        self._last_landing = torch.full((n,), -1, dtype=torch.long, device=dev)
        self._ar_n = torch.arange(n, device=dev)
        self._ar_q = torch.arange(q, device=dev)

    # ------------------------------------------------------------- helpers
    def _offset_substeps(self, latency_ms: torch.Tensor) -> torch.Tensor:
        """Whole substeps a latency spans: ``ceil(L / dt)`` with float slack."""
        dt_ms = self.sim_dt * 1000.0
        return torch.ceil(latency_ms.clamp(min=0.0) / dt_ms - 1e-6).long()

    def pending(self) -> torch.Tensor:
        """Return the ``(N,)`` count of commands not yet landed."""
        return self._len.clone()

    # --------------------------------------------------------------- reset
    def reset(self, env_ids: torch.Tensor) -> None:
        """Clear the given envs' queues and resample their base latency.

        Args:
            env_ids: Indices of the envs that were just respawned.
        """
        n = len(env_ids)
        if n == 0:
            return
        lo, hi = self.latency_ms
        draw = torch.rand(n, generator=self._gen, device=self._device)
        latency = (lo + (hi - lo) * draw) / 1000.0
        # clone-then-write: the state may be an inference tensor (rsl-rl runs
        # collection under inference_mode) and a reset may run outside it
        self.latency_s = self.latency_s.clone()
        self.latency_s[env_ids] = latency
        self.applied = self.applied.clone()
        self.applied[env_ids] = 0.0
        self._len = self._len.clone()
        self._len[env_ids] = 0
        self._head = self._head.clone()
        self._head[env_ids] = 0
        self._last_landing = self._last_landing.clone()
        self._last_landing[env_ids] = -1

    # --------------------------------------------------------------- issue
    def issue(self, command: torch.Tensor) -> None:
        """Queue this decision's command for every env (or drop it).

        Args:
            command: ``(N, 2)`` command in ``[-1, 1]`` issued at the current
                substep.

        Raises:
            RuntimeError: If an env's queue is full (cannot happen within the
                latency bound the capacity was sized for).
        """
        n = self.num_envs
        latency_ms = self.latency_s * 1000.0
        if self.jitter_ms > 0:
            jitter = torch.rand(n, generator=self._gen, device=self._device)
            latency_ms = latency_ms + (2.0 * jitter - 1.0) * self.jitter_ms
        keep = torch.ones(n, dtype=torch.bool, device=self._device)
        if self.drop > 0:
            coin = torch.rand(n, generator=self._gen, device=self._device)
            keep = coin >= self.drop
        landing = torch.maximum(self._last_landing,
                                self.substep + self._offset_substeps(latency_ms))
        if bool((keep & (self._len >= self.capacity)).any()):
            raise RuntimeError(
                f"actuation timing queue overflow (capacity {self.capacity}); "
                "the latency range exceeds what the queue was sized for")
        tail = (self._head + self._len) % self.capacity
        rows = self._ar_n[keep]
        self._q_time = self._q_time.clone()
        self._q_time[rows, tail[keep]] = landing[keep]
        self._q_act = self._q_act.clone()
        self._q_act[rows, tail[keep]] = command[keep]
        self._len = self._len + keep.long()
        self._last_landing = torch.where(keep, landing, self._last_landing)

    # ------------------------------------------------------------- advance
    def advance(self) -> torch.Tensor:
        """Land every command due at this substep; return the applied command.

        Returns:
            The ``(N, 2)`` command the physics follows during this substep.
        """
        q = self.capacity
        pos = (self._head[:, None] + self._ar_q[None, :]) % q
        times = self._q_time.gather(1, pos)
        occupied = self._ar_q[None, :] < self._len[:, None]
        # landings are non-decreasing along the queue, so the due ones form a
        # prefix; the newest due command wins and the older ones are discarded
        due = occupied & (times <= self.substep)
        n_due = due.sum(dim=1)
        has = n_due > 0
        sel = (self._head + n_due - 1) % q
        newest = self._q_act[self._ar_n, sel]
        self.applied = torch.where(has[:, None], newest, self.applied)
        self._head = (self._head + n_due) % q
        self._len = self._len - n_due
        self.substep += 1
        return self.applied


def max_pending(latency_hi_ms: float, jitter_ms: float, sim_dt: float,
                decimation: int) -> int:
    """Return the queue capacity :class:`ActuationTiming` allocates.

    Args:
        latency_hi_ms: Upper bound of the base latency range in ms.
        jitter_ms: Jitter magnitude in ms.
        sim_dt: Physics substep in seconds.
        decimation: Substeps per decision.

    Returns:
        The number of commands that can pend at once, plus one spare slot.
    """
    # A command lands at most ceil((hi + jitter) / dt) substeps after its
    # issue (landing = max(previous landing, issue + L) keeps that bound by
    # induction), so at most that many / decimation + 1 commands pend at once.
    max_offset = math.ceil((latency_hi_ms + jitter_ms) / (sim_dt * 1000.0) - 1e-6)
    return max_offset // max(int(decimation), 1) + 2
