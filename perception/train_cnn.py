"""Train the perception CNN on the collected rollouts.

    python -m perception.train_cnn

Saves the best checkpoint by validation loss to ``perception/perception.pt``.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from perception.dataset import (HOLDOUT_TRACKS, REPO_ROOT, TRAINING_TRACKS,
                                RolloutDataset)
from perception.model import PerceptionCNN

AUGMENT = True      # camera jitter on the training frames; False = as rendered
# an augmented run writes its own checkpoint, so the clean one is never lost
CHECKPOINT = REPO_ROOT / "perception" / (
    "perception_augmented.pt" if AUGMENT else "perception.pt")

EPOCHS = 8          # 8x more data than before, so fewer epochs are enough
BATCH = 64
LR = 1e-4
VAL_MAX = 60_000    # validation sample: past this we are measuring noise


def evaluate(net, loader, loss_fn, device):
    net.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += loss_fn(net(x), y).item() * len(x)
            n += len(x)
    net.train()
    return total / n


def main():
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")

    train_ds = RolloutDataset(tracks=TRAINING_TRACKS, augment=AUGMENT)
    val_ds = RolloutDataset(tracks=HOLDOUT_TRACKS)
    if len(val_ds) > VAL_MAX:      # fixed draw, so runs stay comparable
        g = torch.Generator().manual_seed(0)
        val_ds = Subset(val_ds, torch.randperm(len(val_ds), generator=g)[:VAL_MAX].tolist())
    # augmenting costs ~3x per stack, so the loaders bound the run, not the GPU:
    # more workers than cores still helps, they spend part of their time on the
    # cache file rather than on the CPU
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=14, persistent_workers=True,
                          prefetch_factor=4)
    val_dl = DataLoader(val_ds, batch_size=BATCH, num_workers=4, persistent_workers=True)

    net = PerceptionCNN().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    print(f"train {len(train_ds):,} | val {len(val_ds):,} | {device}"
          f" | augment {AUGMENT}")
    print(f"writing to {CHECKPOINT.name}")
    best = float("inf")
    for epoch in range(EPOCHS):
        total = 0.0
        for n, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            loss = loss_fn(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
            if n % 50 == 0:
                done = n * BATCH / len(train_ds) * 100
                print(f"\r  epoch {epoch:3}  {done:5.1f}%", end="", flush=True)

        train_loss = total / len(train_ds)
        print(f"\r  epoch {epoch:3}  validating...", end="", flush=True)
        val_loss = evaluate(net, val_dl, loss_fn, device)
        print(f"\r  epoch {epoch:3}  train {train_loss:.5f}  val {val_loss:.5f}"
              f"  lr {sched.get_last_lr()[0]:.2e}")
        sched.step()

        if val_loss < best:
            best = val_loss
            torch.save(net.state_dict(), CHECKPOINT)
            print("    -> saved")

    print(f"\nbest val: {best:.5f}")


if __name__ == "__main__":
    main()
