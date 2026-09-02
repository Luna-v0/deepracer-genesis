"""Track registry: waypoint geometry and mesh paths for DeepRacer tracks."""

import math
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

# generated tracks (official routes with procedural meshes, custom-designed
# tracks) are discovered on import — see tools/track_builder.py
_GENERATED = os.path.join(ASSETS_DIR, "tracks", "generated")
if os.path.isdir(_GENERATED):
    for _name in sorted(os.listdir(_GENERATED)):
        _d = os.path.join(_GENERATED, _name)
        if os.path.exists(os.path.join(_d, "route.npy")) and _name not in TRACKS:
            _rel = os.path.relpath(_d, ASSETS_DIR)
            TRACKS[_name] = (f"{_rel}/track.obj", f"{_rel}/route.npy", None)


class Track:
    """Per-variant track geometry container (centerline + derived quantities).

    A plain geometry holder: :class:`MultiTrack` reads these tensors to build
    its padded batched view and owns all the actual queries (localize /
    lookahead / spawn). ``Track`` carries no query methods of its own.

    Attributes:
        name: track identifier used to look up assets.
        mesh_path: filesystem path to the track mesh.
        field_path: path to the field-overlay mesh, or None.
        obj_path: OBJ mesh path for renderers that can't read DAE.
        device: torch device holding the geometry tensors.
        center: centerline waypoint positions.
        half_width: half of the track width at each waypoint.
        n_wps: number of waypoints.
        tangent: unit vector along the driving direction per waypoint.
        normal: left normal of the tangent per waypoint.
        track_yaw: heading of the centerline per waypoint.
        cum_len: cumulative arclength at each waypoint.
        total_len: total centerline length.
        curvature: signed curvature per waypoint.
    """

    def __init__(self, name, device):
        mesh_rel, route_rel, field_rel = TRACKS[name]
        self.name = name
        self.mesh_path = os.path.join(ASSETS_DIR, mesh_rel)
        self.field_path = os.path.join(ASSETS_DIR, field_rel) if field_rel else None
        # OBJ conversion (same geometry/textures) for renderers that can't read DAE (Nyx);
        # generated tracks are OBJ already
        if self.mesh_path.endswith(".obj"):
            self.obj_path = self.mesh_path
        else:
            base = os.path.basename(mesh_rel).rsplit(".", 1)[0]
            self.obj_path = os.path.join(os.path.dirname(self.mesh_path), "obj", base + ".obj")
        self.device = device

        wps = np.load(os.path.join(ASSETS_DIR, route_rel)).astype(np.float32)
        # AWS routes commonly repeat the first waypoint at the end; drop it.
        if np.allclose(wps[0, :2], wps[-1, :2], atol=1e-6):
            wps = wps[:-1]
        # 54 of the 62 routes also repeat one mid-loop, and a zero-length
        # segment has no direction: arctan2(0, 0) poisons yaw, then curvature.
        step = np.roll(wps[:, :2], -1, axis=0) - wps[:, :2]
        wps = wps[np.linalg.norm(step, axis=1) >= 1e-3]

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
        # signed curvature per waypoint: d(yaw)/ds in waypoint order
        # (+ = left turn when driving in waypoint order)
        dyaw = torch.remainder(torch.roll(self.track_yaw, -1) - self.track_yaw
                               + math.pi, 2 * math.pi) - math.pi
        self.curvature = dyaw / seg_len                        # (W,)


def grid_offsets(n_variants, spacing, device):
    """World-XY offset per variant on a near-square grid (Part O tiling).

    Zero/degenerate spacing yields all-zero offsets (no tiling).

    Args:
        n_variants: Number of track variants.
        spacing: Metres between tiles; <=0 disables tiling.
        device: Torch device for the returned tensor.

    Returns:
        A ``(n_variants, 2)`` tensor of per-variant world-XY offsets.
    """
    if spacing <= 0 or n_variants <= 1:
        return torch.zeros(n_variants, 2, device=device)
    cols = int(math.ceil(math.sqrt(n_variants)))
    idx = torch.arange(n_variants, device=device)
    col = (idx % cols).float()
    row = torch.div(idx, cols, rounding_mode="floor").float()
    return torch.stack([col * spacing, row * spacing], dim=1)


def balanced_variant_mapping(n_variants, n_envs, device):
    """Map variants to envs in contiguous blocks, matching Genesis morph assignment."""
    if n_envs >= n_variants:
        base, extra = divmod(n_envs, n_variants)
        sizes = [base + 1] * extra + [base] * (n_variants - extra)
        return torch.repeat_interleave(
            torch.arange(n_variants, device=device),
            torch.tensor(sizes, device=device))
    return torch.arange(n_envs, device=device)


class MultiTrack:
    """Batched geometry queries across per-env track variants, padded to a common length.

    Attributes:
        tracks: per-variant Track objects.
        names: track names in variant order.
        device: torch device holding the geometry tensors.
        variant_idx: per-env variant assignment.
        center: padded centerline positions per variant.
        tangent: padded driving-direction tangents per variant.
        normal: padded left normals per variant.
        track_yaw: padded centerline headings per variant.
        cum_len: padded cumulative arclengths per variant.
        curvature: padded signed curvature per variant.
        half_width: padded half track widths per variant.
        n_wps_v: waypoint count per variant.
        total_len_v: total length per variant.
        mesh_paths: mesh path per variant.
        obj_paths: OBJ mesh path per variant.
        total_len_env: total track length per env.
        n_wps_env: waypoint count per env.
    """

    def __init__(self, names, num_envs, device, grid_spacing=0.0):
        self.tracks = [Track(n, device) for n in names]
        self.names = list(names)
        self.device = device
        self.variant_idx = balanced_variant_mapping(len(self.tracks), num_envs, device)  # (N,)

        W = max(t.n_wps for t in self.tracks)
        V = len(self.tracks)

        # Part O spatial tiling: each variant is translated to its own world
        # tile so a camera env sees only its home track. Applied to ABSOLUTE
        # fields only (waypoint centers below, spawn positions in spawn_pose);
        # tangent/normal/half_width/cum_len/curvature are local/differential and
        # therefore offset-invariant. grid_spacing=0 -> all-zero -> no tiling.
        self.grid_spacing = grid_spacing
        self.variant_offset = grid_offsets(V, grid_spacing, device)   # (V, 2)

        def pad(attr, fill):
            out = torch.full((V, W, *getattr(self.tracks[0], attr).shape[1:]), fill, device=device)
            for v, t in enumerate(self.tracks):
                out[v, : t.n_wps] = getattr(t, attr)
            return out

        self.center = pad("center", float("inf"))         # (V, W, 2)
        # translate each variant's real waypoints onto its tile (padding stays
        # +inf: inf + finite = inf, and localize relies on +inf padding to never
        # win the nearest-waypoint argmin)
        for v, t in enumerate(self.tracks):
            self.center[v, : t.n_wps] += self.variant_offset[v]
        self.tangent = pad("tangent", 0.0)
        self.normal = pad("normal", 0.0)
        self.track_yaw = pad("track_yaw", 0.0)
        self.cum_len = pad("cum_len", float("inf"))   # +inf pad keeps rows sorted
        self.curvature = pad("curvature", 0.0)
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

    def lookahead(self, wp_idx, k, stride=2, dir_sign=None):
        """Indices of the k upcoming waypoints per env, (N, k); `dir_sign` (N,) of +/-1 reverses walk direction."""
        offs = torch.arange(1, k + 1, device=self.device) * stride
        offs = offs[None, :]
        if dir_sign is not None:
            offs = offs * dir_sign[:, None].long()
        return torch.remainder(wp_idx[:, None] + offs, self.n_wps_env[:, None])

    def curvature_ahead(self, progress_m, distances, dir_sign=None):
        """Signed track curvature sampled at fixed arclengths ahead.

        Args:
            progress_m: (N,) current arc position of each env.
            distances: (H,) meters ahead (in each env's own driving
                direction) at which to probe.
            dir_sign: (N,) of +/-1; reversed envs probe backwards along the
                waypoint order and see the curvature sign flipped.

        Returns:
            (N, H) curvature in 1/m, signed in the env's own driving frame
            (+ = the road bends left, from the driver's perspective).
        """
        d = torch.as_tensor(distances, device=self.device, dtype=progress_m.dtype)
        sign = torch.ones_like(progress_m) if dir_sign is None else dir_sign
        s = torch.remainder(progress_m[:, None] + sign[:, None] * d[None, :],
                            self.total_len_env[:, None])       # (N, H)
        rows = self.cum_len[self._ev]                           # (N, W), inf-padded
        idx = (torch.searchsorted(rows, s.contiguous()) - 1).clamp(min=0)
        idx = torch.minimum(idx, (self.n_wps_env[:, None] - 1).long())
        k = torch.gather(self.curvature[self._ev], 1, idx)      # (N, H)
        return k * sign[:, None]

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
