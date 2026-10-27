"""
See what the world model predicts: decode its latents back into pictures
=========================================================================
The nano world model (train_world_model.py) predicts the NEXT LATENT, never pixels. To look at
those predictions anyway, this script trains a small decoder (64-d latent -> 64x64 RGB) on a
dataset's frames with the checkpoint's encoder frozen, then decodes:

  * z_t            the current frame as the model embeds it ("what it keeps")
  * z_hat_{t+k}    the predictor's output for the command actually flown, k = 4/8/16/32 steps
                   (83-667 ms), next to the TRUE frames t+k
  * z_hat_{t+32}   for every candidate command from the same frame (forward / slow / veer_left /
                   veer_right / climb / hover), with the collision heads' P(warn within 32 steps)
                   under each — the action-conditioning the MPC planner relies on

Caveat: the predictor was trained to match the EMA target encoder, whose weights are not in the
checkpoint; the decoder is fitted on the online encoder's latents, so decoded predictions carry
that small mismatch on top of what a 64-d latent can hold (expect blurry pillars in the right
place, not photographs). Numbers printed per horizon: pixel MSE of the decoded prediction vs the
true future frame, against the "future == present" copy baseline, on held-out rollouts.

  python src/decode_latents.py --model src/output/world_model_pybullet_ref.pth --data src/output/wm_dataset.npz --tag ref_pybullet
  python src/decode_latents.py --model src/output/world_model_mixed_big.pth --data src/output/wm_dataset_mixed_big_60body.npz --tag mixed_big
Outputs src/output/report/pred_<tag>.png, actions_<tag>.png, decoder_<tag>.pth, metrics_<tag>.json.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gen_wm_dataset import ACTION_NAMES, ACTION_VECS, A_NORM, HORIZONS, window_valid  # noqa: E402
from train_world_model import _split_rollouts, load_model  # noqa: E402

OUT_DIR = os.path.join(HERE, "output", "report")


class Decoder(nn.Module):
    """64-d latent -> (3, 64, 64) in [0, 1]."""

    def __init__(self, d: int = 64):
        super().__init__()
        self.fc = nn.Linear(d, 64 * 8 * 8)
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(),  # 16
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),  # 32
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),  # 64
            nn.Conv2d(16, 3, 3, 1, 1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 64, 8, 8))


def frames_to_tensor(frames: np.ndarray, device) -> torch.Tensor:
    return torch.tensor(frames, dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0


def train_decoder(enc, frames: np.ndarray, device, epochs: int = 12, batch: int = 256, seed: int = 0) -> Decoder:
    torch.manual_seed(seed)
    dec = Decoder().to(device)
    opt = torch.optim.Adam(dec.parameters(), lr=2e-3)
    flat = frames.reshape(-1, *frames.shape[2:])
    n = len(flat)
    for ep in range(epochs):
        order = np.random.default_rng(seed + ep).permutation(n)
        total = 0.0
        for i in range(0, n, batch):
            x = frames_to_tensor(flat[order[i : i + batch]], device)
            with torch.no_grad():
                z = enc(x)
            loss = ((dec(z) - x) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(x)
        print(f"  decoder epoch {ep + 1}/{epochs}: mse {total / n:.4f}")
    dec.eval()
    return dec


def to_img(t: torch.Tensor) -> np.ndarray:
    return (t.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def pick_samples(data: dict, rolls, rng, n: int = 4) -> list:
    """(rollout, t) pairs on held windows that approach something (nearest obstacle within
    1.2 m by the end of the longest horizon), preferring in-path rollouts of both scene kinds."""
    L, K = data["frames"].shape[1], HORIZONS[-1]
    cands = []
    for r in rolls:
        for t in range(0, L - K, 4):
            if window_valid(data["seg"][r], t, K) and data["dists"][r, t + K] < 1.2 and data["dists"][r, t] > 0.6:
                cands.append((r, t))
    if not cands:
        cands = [(r, t) for r in rolls for t in range(0, L - K, 8) if window_valid(data["seg"][r], t, K)]
    rng.shuffle(cands)
    out, seen = [], set()
    scene = data.get("scene_type", np.zeros(len(data["frames"]), dtype=np.int8))
    for kind in (0, 1, 0, 1, 0, 1, 0, 1):  # alternate pillars / rooms when both exist
        for r, t in cands:
            if int(scene[r]) == kind and r not in seen:
                out.append((r, t))
                seen.add(r)
                break
        if len(out) >= n:
            break
    for r, t in cands:
        if len(out) >= n:
            break
        if r not in seen:
            out.append((r, t))
            seen.add(r)
    return out[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--decoder", default=None, help="reuse a decoder trained on another dataset (cross-domain look)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)
    blob = np.load(args.data)
    data = {k: blob[k] for k in blob.files}
    enc, pred, cheads, nhead, meta = load_model(args.model, device=device)
    rng = np.random.default_rng(0)
    tr, va = _split_rollouts(data, rng)
    print(f"[INFO] {args.model} on {args.data}: {len(tr)} train / {len(va)} held-out rollouts, device {device}")

    if args.decoder:
        dec = Decoder().to(device)
        dec.load_state_dict(torch.load(args.decoder, map_location=device, weights_only=True))
        dec.eval()
        print(f"[INFO] decoder reused from {args.decoder}")
    else:
        dec = train_decoder(enc, data["frames"][tr], device, epochs=args.epochs)
        torch.save(dec.state_dict(), os.path.join(OUT_DIR, f"decoder_{args.tag}.pth"))

    # --- held-out numbers: decoded prediction vs true future, against the copy baseline ----
    F, A = data["frames"], data["actions"]
    L, K = F.shape[1], HORIZONS[-1]
    idx = [(r, t) for r in va for t in range(0, L - K, 2) if window_valid(data["seg"][r], t, K)]
    mse_pred = np.zeros(len(HORIZONS))
    mse_copy = np.zeros(len(HORIZONS))
    mse_recon = 0.0
    with torch.no_grad():
        for i in range(0, len(idx), 256):
            chunk = idx[i : i + 256]
            x = frames_to_tensor(np.array([F[r, t] for r, t in chunk]), device)
            a = torch.tensor(np.array([A[r, t] for r, t in chunk]) / A_NORM, dtype=torch.float32, device=device)
            z = enc(x)
            zh = pred(z, a)  # (B, H, D)
            mse_recon += float(((dec(z) - x) ** 2).mean(dim=(1, 2, 3)).sum())
            for j, k in enumerate(HORIZONS):
                xk = frames_to_tensor(np.array([F[r, t + k] for r, t in chunk]), device)
                mse_pred[j] += float(((dec(zh[:, j]) - xk) ** 2).mean(dim=(1, 2, 3)).sum())
                mse_copy[j] += float(((x - xk) ** 2).mean(dim=(1, 2, 3)).sum())
    n = max(len(idx), 1)
    metrics = {
        "held_out_windows": len(idx),
        "reconstruction_mse": mse_recon / n,
        "horizons": list(map(int, HORIZONS)),
        "decoded_prediction_mse": (mse_pred / n).tolist(),
        "copy_baseline_mse": (mse_copy / n).tolist(),
    }
    with open(os.path.join(OUT_DIR, f"metrics_{args.tag}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("DECODE OK:", json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, list) else (round(v, 4) if isinstance(v, float) else v)) for k, v in metrics.items()}))

    # --- figure 1: now | decoded now | predicted +k | true +k -------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = pick_samples(data, va, rng, args.samples)
    cols = ["frame t", "decode(z_t)"] + [f"pred +{k}" for k in HORIZONS] + [f"true +{k}" for k in HORIZONS]
    fig, axes = plt.subplots(len(samples), len(cols), figsize=(1.35 * len(cols), 1.5 * len(samples) + 0.4))
    axes = np.atleast_2d(axes)
    scene_names = {0: "pillars", 1: "room"}
    with torch.no_grad():
        for i, (r, t) in enumerate(samples):
            x = frames_to_tensor(F[r, t][None], device)
            a = torch.tensor(A[r, t][None] / A_NORM, dtype=torch.float32, device=device)
            z = enc(x)
            zh = pred(z, a)[0]
            row = [F[r, t], to_img(dec(z)[0])] + [to_img(dec(zh[j][None])[0]) for j in range(len(HORIZONS))] + [F[r, t + k] for k in HORIZONS]
            kind = scene_names.get(int(data.get("scene_type", np.zeros(len(F)))[r]), "")
            for j, img in enumerate(row):
                axes[i, j].imshow(img)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                if i == 0:
                    axes[i, j].set_title(cols[j], fontsize=8)
            axes[i, 0].set_ylabel(f"r{r} t{t}\n{kind}, {ACTION_NAMES[int(data['act_id'][r, t])]}", fontsize=7)
    fig.suptitle(f"{os.path.basename(args.model)} on {os.path.basename(args.data)} — held command, predicted vs true future", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"pred_{args.tag}.png"), dpi=130)
    plt.close(fig)

    # --- figure 2: the same frames, every candidate command at +32 ---------------------------
    cands = torch.tensor(ACTION_VECS / A_NORM, dtype=torch.float32, device=device)
    cols2 = ["frame t"] + [f"{n} +{K}" for n in ACTION_NAMES]
    fig, axes = plt.subplots(len(samples), len(cols2), figsize=(1.45 * len(cols2), 1.65 * len(samples) + 0.4))
    axes = np.atleast_2d(axes)
    with torch.no_grad():
        for i, (r, t) in enumerate(samples):
            x = frames_to_tensor(F[r, t][None], device)
            z = enc(x)
            sv = float(data["speed"][r])
            zh = pred(z.expand(len(cands), -1), cands * sv)  # (n_a, H, D)
            p = torch.sigmoid(cheads(zh))[:, -1, :]  # (n_a, 2 rings) at the longest horizon
            axes[i, 0].imshow(F[r, t])
            axes[i, 0].set_ylabel(f"r{r} t{t}\nspeed {sv:.2f}x", fontsize=7)
            for j in range(len(ACTION_NAMES)):
                axes[i, j + 1].imshow(to_img(dec(zh[j, -1][None])[0]))
                axes[i, j + 1].set_xlabel(f"warn {float(p[j, 0]):.2f} crit {float(p[j, 1]):.2f}", fontsize=6)
            for j in range(len(cols2)):
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                if i == 0:
                    axes[i, j].set_title(cols2[j], fontsize=8)
    fig.suptitle(f"{os.path.basename(args.model)} — imagined +{K} steps per candidate command, with P(warn)/P(crit)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"actions_{args.tag}.png"), dpi=130)
    plt.close(fig)
    print(f"[INFO] saved {OUT_DIR}/pred_{args.tag}.png and actions_{args.tag}.png")


if __name__ == "__main__":
    main()
