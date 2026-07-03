"""Track registry: waypoint geometry + mesh paths for DeepRacer tracks.

Waypoint .npy files come from the original AWS DeepRacer simapp
(`simulation/routes/<track>.npy`) with shape (W, 6):
[center_x, center_y, inner_x, inner_y, outer_x, outer_y].
"""

import os

import numpy as np
import torch

from .. import ASSETS_DIR

# name -> (mesh, route npy, optional field-overlay mesh) relative to assets/.
# The overlay replaces ground submeshes whose alpha-textured materials the
# Madrona batch renderer renders transparent (background bleed-through).
TRACKS = {
    "reinvent_base": ("tracks/reinvent/reinvent_base.dae", "routes/reinvent_base.npy",
                      "tracks/reinvent/field.dae"),
    "reInvent2019_track": ("tracks/reInvent2019_track/reInvent2019_track.dae",
                           "routes/reInvent2019_track.npy", None),
    "2022_reinvent_champ": ("tracks/2022_reinvent_champ/2022_reinvent_champ.dae",
                            "routes/2022_reinvent_champ.npy", None),
}


class Track:
    """Vectorized (GPU) track geometry queries against the centerline."""

    def __init__(self, name, device):
        mesh_rel, route_rel, field_rel = TRACKS[name]
        self.name = name
        self.mesh_path = os.path.join(ASSETS_DIR, mesh_rel)
        self.field_path = os.path.join(ASSETS_DIR, field_rel) if field_rel else None
        # OBJ conversion (same geometry/textures) for renderers that can't read DAE (Nyx)
        base = os.path.basename(mesh_rel).rsplit(".", 1)[0]
        self.obj_path = os.path.join(os.path.dirname(self.mesh_path), "obj", base + ".obj")
        self.device = device

        wps = np.load(os.path.join(ASSETS_DIR, route_rel)).astype(np.float32)
        # AWS routes commonly repeat the first waypoint at the end; drop it.
        if np.allclose(wps[0, :2], wps[-1, :2], atol=1e-6):
            wps = wps[:-1]

        center = torch.tensor(wps[:, 0:2], device=device)
        inner = torch.tensor(wps[:, 2:4], device=device)
        outer = torch.tensor(wps[:, 4:6], device=device)

        self.center = center                                   # (W, 2)
        self.half_width = 0.5 * (outer - inner).norm(dim=1)    # (W,)
        self.n_wps = center.shape[0]

        nxt = torch.roll(center, -1, dims=0)
        seg = nxt - center
        seg_len = seg.norm(dim=1).clamp(min=1e-6)
        self.tangent = seg / seg_len[:, None]                  # (W, 2), points along driving direction
        # left normal of the tangent: positive lateral offset = left of center
        self.normal = torch.stack([-self.tangent[:, 1], self.tangent[:, 0]], dim=1)
        self.track_yaw = torch.atan2(self.tangent[:, 1], self.tangent[:, 0])
        self.cum_len = torch.cat([torch.zeros(1, device=device), seg_len.cumsum(0)[:-1]])  # (W,)
        self.total_len = seg_len.sum()

    def localize(self, pos_xy):
        """For a batch of positions (N, 2), return per-env track frame quantities.

        Returns dict with: wp_idx (N,), lateral (N,) signed offset (+ = left),
        half_width (N,), progress_m (N,) arclength in [0, L), track_yaw (N,).
        """
        d = torch.cdist(pos_xy, self.center)                   # (N, W)
        wp_idx = d.argmin(dim=1)                               # (N,)
        c = self.center[wp_idx]
        t = self.tangent[wp_idx]
        n = self.normal[wp_idx]
        rel = pos_xy - c
        lateral = (rel * n).sum(dim=1)
        along = (rel * t).sum(dim=1)
        progress_m = torch.remainder(self.cum_len[wp_idx] + along, self.total_len)
        return {
            "wp_idx": wp_idx,
            "lateral": lateral,
            "half_width": self.half_width[wp_idx],
            "progress_m": progress_m,
            "track_yaw": self.track_yaw[wp_idx],
        }

    def lookahead(self, wp_idx, k, stride=2):
        """Indices of k upcoming waypoints: (N, k)."""
        offs = torch.arange(1, k + 1, device=self.device) * stride
        return torch.remainder(wp_idx[:, None] + offs[None, :], self.n_wps)

    def spawn_pose(self, wp_idx, lateral_noise=0.0, yaw_noise=0.0):
        """Spawn position/yaw at given waypoint indices (N,). Returns pos_xy (N,2), yaw (N,)."""
        n = wp_idx.shape[0]
        c = self.center[wp_idx]
        nml = self.normal[wp_idx]
        lat = (torch.rand(n, device=self.device) * 2 - 1) * lateral_noise
        pos = c + nml * lat[:, None]
        yaw = self.track_yaw[wp_idx] + (torch.rand(n, device=self.device) * 2 - 1) * yaw_noise
        return pos, yaw


def balanced_variant_mapping(n_variants, n_envs, device):
    """Same contiguous block mapping Genesis uses to assign heterogeneous
    morph variants to environments (kinematic_solver._balanced_variant_mapping)."""
    if n_envs >= n_variants:
        base, extra = divmod(n_envs, n_variants)
        sizes = [base + 1] * extra + [base] * (n_variants - extra)
        return torch.repeat_interleave(
            torch.arange(n_variants, device=device),
            torch.tensor(sizes, device=device))
    return torch.arange(n_envs, device=device)


class MultiTrack:
    """Batched geometry queries across per-env track variants.

    Pads all tracks' waypoint arrays to a common length (padding pushed to
    +inf so argmin never selects it) and gathers by each env's variant index.
    """

    def __init__(self, names, num_envs, device):
        self.tracks = [Track(n, device) for n in names]
        self.names = list(names)
        self.device = device
        self.variant_idx = balanced_variant_mapping(len(self.tracks), num_envs, device)  # (N,)

        W = max(t.n_wps for t in self.tracks)
        V = len(self.tracks)

        def pad(attr, fill):
            out = torch.full((V, W, *getattr(self.tracks[0], attr).shape[1:]), fill, device=device)
            for v, t in enumerate(self.tracks):
                out[v, : t.n_wps] = getattr(t, attr)
            return out

        self.center = pad("center", float("inf"))         # (V, W, 2)
        self.tangent = pad("tangent", 0.0)
        self.normal = pad("normal", 0.0)
        self.track_yaw = pad("track_yaw", 0.0)
        self.cum_len = pad("cum_len", 0.0)
        self.half_width = pad("half_width", 1.0)
        self.n_wps_v = torch.tensor([t.n_wps for t in self.tracks], device=device)
        self.total_len_v = torch.stack([t.total_len for t in self.tracks])
        self.mesh_paths = [t.mesh_path for t in self.tracks]
        self.obj_paths = [t.obj_path for t in self.tracks]

        # per-env views
        self._ev = self.variant_idx
        self.total_len_env = self.total_len_v[self._ev]     # (N,)
        self.n_wps_env = self.n_wps_v[self._ev]

    @property
    def total_len(self):
        # used by the env's lap-wrap logic as a (N,)-broadcastable quantity
        return self.total_len_env

    def localize(self, pos_xy, envs_idx=None):
        ev = self._ev if envs_idx is None else self._ev[envs_idx]
        C = self.center[ev]                                  # (N, W, 2)
        d = (C - pos_xy[:, None, :]).norm(dim=2)
        wp_idx = d.argmin(dim=1)
        ar = torch.arange(len(ev), device=self.device)
        c = C[ar, wp_idx]
        t = self.tangent[ev, wp_idx]
        n = self.normal[ev, wp_idx]
        rel = pos_xy - c
        lateral = (rel * n).sum(dim=1)
        along = (rel * t).sum(dim=1)
        L = self.total_len_v[ev]
        progress_m = torch.remainder(self.cum_len[ev, wp_idx] + along, L)
        return {
            "wp_idx": wp_idx,
            "lateral": lateral,
            "half_width": self.half_width[ev, wp_idx],
            "progress_m": progress_m,
            "track_yaw": self.track_yaw[ev, wp_idx],
        }

    def lookahead(self, wp_idx, k, stride=2):
        offs = torch.arange(1, k + 1, device=self.device) * stride
        return torch.remainder(wp_idx[:, None] + offs[None, :], self.n_wps_env[:, None])

    def lookahead_points(self, la_idx):
        ev = self._ev[:, None].expand_as(la_idx)
        return self.center[ev, la_idx]                       # (N, K, 2)

    def spawn_pose(self, env_ids, random_start, lateral_noise=0.0, yaw_noise=0.0):
        ev = self._ev[env_ids]
        n = len(env_ids)
        if random_start:
            wp = (torch.rand(n, device=self.device) * self.n_wps_v[ev]).long()
        else:
            wp = torch.zeros(n, dtype=torch.long, device=self.device)
        c = self.center[ev, wp]
        nml = self.normal[ev, wp]
        lat = (torch.rand(n, device=self.device) * 2 - 1) * lateral_noise
        pos = c + nml * lat[:, None]
        yaw = self.track_yaw[ev, wp] + (torch.rand(n, device=self.device) * 2 - 1) * yaw_noise
        return pos, yaw
