"""
Lesson 3 (Route B) — train a Dronet-style CNN to predict obstacle bearing
=========================================================================
This is the secondary route, and it's the bridge to real nano-drones. We train
a small convolutional network (inspired by PULP-Dronet, which runs on the
Crazyflie's AI-deck) to look at a camera image and output the obstacle's
bearing -- learning to imitate Lesson 2's hand-written detector.

Why bother, when the detector already works? Because a neural net:
  * doesn't need hand-tuned color thresholds (it learns features),
  * generalizes to things color-thresholding can't handle, and
  * is what actually fits on the GAP8 chip in Lesson 4.

Run gen_dataset.py first, then:
  python train_cnn.py --epochs 40

Saves the model to output/dronet_cnn.pth (git-ignored).
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

DATA = os.path.join(os.path.dirname(__file__), "output", "imitation_dataset.npz")
MODEL = os.path.join(os.path.dirname(__file__), "output", "dronet_cnn.pth")
BEARING_SCALE = 30.0  # labels are within ~+/-30 deg; scale to ~[-1, 1] for training


class TinyDronet(nn.Module):
    """A compact CNN: 3 conv blocks -> global pool -> 2 FC -> 1 number (bearing).

    Deliberately small so it trains in seconds and could be quantized onto a
    microcontroller later (the spirit of PULP-Dronet)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.ReLU(),  # 64 -> 32
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),  # 32 -> 16
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),  # 16 -> 8
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):  # x: (N, 3, H, W)
        return self.head(self.features(x)).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    if not os.path.exists(DATA):
        raise SystemExit(f"No dataset at {DATA}. Run: python gen_dataset.py first.")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[INFO] training on {device}")

    blob = np.load(DATA)
    # (N, H, W, 3) -> (N, 3, H, W) tensor; labels scaled to ~[-1, 1].
    X = torch.tensor(blob["X"]).permute(0, 3, 1, 2)
    y = torch.tensor(blob["y"]) / BEARING_SCALE

    n_val = max(1, int(0.2 * len(X)))
    perm = torch.randperm(len(X))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    Xtr, ytr = X[train_idx].to(device), y[train_idx].to(device)
    Xva, yva = X[val_idx].to(device), y[val_idx].to(device)

    model = TinyDronet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), args.batch):
            b = order[i : i + args.batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()
        if epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                val_mae_deg = (model(Xva) - yva).abs().mean().item() * BEARING_SCALE
            print(f"  epoch {epoch:3d}  val MAE = {val_mae_deg:.2f} deg")

    torch.save(model.state_dict(), MODEL)
    print(f"[INFO] saved model to {MODEL}")

    # Show a few predictions vs the teacher's labels.
    model.eval()
    with torch.no_grad():
        pred = model(Xva[:6]).cpu().numpy() * BEARING_SCALE
    true = (yva[:6].cpu().numpy()) * BEARING_SCALE
    print("  sample  predicted   teacher")
    for pr, tr in zip(pred, true):
        print(f"        {pr:+7.1f}   {tr:+7.1f}  deg")


if __name__ == "__main__":
    main()
