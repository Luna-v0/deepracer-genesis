"""Raw frame banks: render once on the GPU box, iterate anywhere.

A bank is a directory of raw (DR-stripped) camera frames captured at a
teleport pose grid, plus a ``meta.json``. Everything downstream of the render
is pure torch (:mod:`.pipeline`), so a bank makes the whole image tier —
sweeps, stage strips, offline prove checks — instant and Genesis-free.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Sequence

import numpy as np
import torch


class FrameBank:
    """A recorded pose-grid of raw frames.

    Attributes:
        meta: The bank's provenance (track, renderer, resolution, poses,
            num_envs, and a ``raw: true`` marker — banks are only valid when
            recorded from a DR-stripped session).
        poses: ``(P, 3)`` array of (waypoint, lateral_frac, yaw_off) rows.
    """

    def __init__(self, path: str) -> None:
        """Open a recorded bank.

        Args:
            path: The bank directory.

        Raises:
            FileNotFoundError: If the directory holds no bank.
        """
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)
        data = np.load(os.path.join(path, "frames.npz"))
        self._images = data["image"]          # (P, N, H, W, 3) uint8
        self.poses = data["pose"]             # (P, 3)
        self._path = path

    def raw(self, pose: int = 0, device="cpu") -> torch.Tensor:
        """Raw frames at one pose, in pipeline convention.

        Args:
            pose: Pose-grid index.
            device: Device for the returned tensor.

        Returns:
            ``(N, 3, H, W)`` float frames in [0, 1].
        """
        arr = torch.from_numpy(self._images[pose]).to(device)
        return arr.permute(0, 3, 1, 2).float().div_(255.0)

    def __len__(self) -> int:
        """Number of recorded poses."""
        return self._images.shape[0]

    @staticmethod
    def record(session, out_dir: str, *,
               waypoints: Sequence[int] = (0, 5, 10, 20, 40),
               lateral_fracs: Sequence[float] = (0.0,),
               yaw_offs: Sequence[float] = (0.0,)) -> "FrameBank":
        """Capture a pose grid from a live session into a bank directory.

        Args:
            session: A live :class:`~.session.EditorSession` (must be
                DR-stripped, which the session constructors guarantee).
            out_dir: Destination directory.
            waypoints: Waypoint indices of the grid.
            lateral_fracs: Lateral offsets (fraction of half-width).
            yaw_offs: Heading offsets (radians).

        Returns:
            The recorded bank, reopened from disk.
        """
        os.makedirs(out_dir, exist_ok=True)
        images, poses = [], []
        for wp in waypoints:
            for lat in lateral_fracs:
                for yaw in yaw_offs:
                    session.teleport(waypoint=int(wp), lateral_frac=float(lat),
                                     yaw_off=float(yaw))
                    raw = session.raw()                       # (N, 3, H, W)
                    images.append((raw.permute(0, 2, 3, 1) * 255)
                                  .byte().cpu().numpy())
                    poses.append((int(wp), float(lat), float(yaw)))
        np.savez_compressed(os.path.join(out_dir, "frames.npz"),
                            image=np.stack(images),
                            pose=np.asarray(poses, dtype=np.float32))
        vision = session.env.cfg["vision"]
        meta = {
            "raw": True,
            "track": session.env.cfg["sim"]["track"],
            "renderer": vision.get("vision_renderer", "batch"),
            "camera_res": list(vision["camera_res"]),
            "num_envs": session.num_envs,
            "poses": [list(p) for p in poses],
        }
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return FrameBank(out_dir)
