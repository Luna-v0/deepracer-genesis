"""Train the perception CNN on the collected rollouts.

Run with ``uv run python -m experiments.perception.train_cnn``; the best
checkpoint by validation loss lands under ``runs/cnn/``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from deepracer_genesis.perception.dataset import (HOLDOUT_TRACKS, TRAINING_TRACKS,
                                                  RolloutDataset)
from deepracer_genesis.perception.model import PerceptionCNN

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
JITTER = True       # camera jitter on the training frames; False = as rendered
# a jittered run writes its own checkpoint, so the clean one is never lost
CHECKPOINT_DIR = REPO_ROOT / "runs" / "cnn"
CHECKPOINT = CHECKPOINT_DIR / ("perception_jittered.pt" if JITTER
                               else "perception.pt")

EPOCHS = 8
BATCH = 64
LR = 1e-4
SEED = 0
VAL_MAX = 60_000    # validation sample: past this we are measuring noise
# jitter costs ~3x per stack, so the loaders bound the run rather than the GPU
TRAIN_WORKERS = 14
VAL_WORKERS = 4


def evaluate(net: nn.Module, loader: DataLoader, loss_fn: nn.Module,
             device: torch.device) -> float:
    """Return the mean loss over a loader, leaving the net in train mode.

    Args:
        net: The network under evaluation.
        loader: Batches to evaluate over.
        loss_fn: Loss to average.
        device: Device to move batches to.

    Returns:
        The sample-weighted mean loss.
    """
    net.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += loss_fn(net(x), y).item() * len(x)
            n += len(x)
    net.train()
    return total / n


def main() -> None:
    """Train the CNN and save the best checkpoint by validation loss."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = RolloutDataset(tracks=TRAINING_TRACKS, jitter=JITTER, seed=SEED)
    val_ds = RolloutDataset(tracks=HOLDOUT_TRACKS)
    if len(val_ds) > VAL_MAX:      # fixed draw, so runs stay comparable
        g = torch.Generator().manual_seed(SEED)
        val_ds = Subset(val_ds,
                        torch.randperm(len(val_ds), generator=g)[:VAL_MAX].tolist())
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=TRAIN_WORKERS, persistent_workers=True,
                          prefetch_factor=4)
    val_dl = DataLoader(val_ds, batch_size=BATCH, num_workers=VAL_WORKERS,
                        persistent_workers=True)

    net = PerceptionCNN().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    logger.info("train %d | val %d | %s | jitter %s",
                len(train_ds), len(val_ds), device, JITTER)
    logger.info("writing to %s", CHECKPOINT)
    best = float("inf")
    for epoch in range(EPOCHS):
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x)

        train_loss = total / len(train_ds)
        val_loss = evaluate(net, val_dl, loss_fn, device)
        logger.info("epoch %3d  train %.5f  val %.5f  lr %.2e",
                    epoch, train_loss, val_loss, sched.get_last_lr()[0])
        sched.step()

        if val_loss < best:
            best = val_loss
            torch.save(net.state_dict(), CHECKPOINT)
            logger.info("  -> saved")

    logger.info("best val: %.5f", best)


if __name__ == "__main__":
    main()
