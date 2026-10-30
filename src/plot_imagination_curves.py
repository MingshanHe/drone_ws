"""Training and test curves of the imagination-only Dreamer runs (train_dreamer.py --imag-only).

Row 1, training losses on training batches: reconstruction (per pixel), KL dyn, and for overshoot runs
the imagined-frame reconstruction (os_rec, per pixel). Row 2, the imagination test over training:
decoded gain @+32 frames (1 - MSE imagined / MSE copy-last-frame, after 8 observed steps) on 64 training
rollouts, on the held-out rollouts and on 64 unseen-house rollouts. Row 3: posterior reconstruction
and posterior-prior KL on the three test sets, and gain vs horizon for the kept checkpoint.
The x axis is training sequences seen (steps x batch), so runs with different batch sizes line up.

  python src/plot_imagination_curves.py   -> src/output/report/imagination_curves.png
"""

import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
EV = os.path.join(HERE, "output", "evals")
REP = os.path.join(HERE, "output", "report")
from draw_pipeline import BLUE, GREEN, INK, INK2, MUTED, ORANGE, SURFACE  # noqa: E402

PIX = 64 * 64 * 3
RUNS = [("imag_C_nano", "nano 1.26M, hybrid C, DreamerV3 loss", ORANGE),
        ("imag_C_nano_os", "nano 1.26M, hybrid C, + overshoot", GREEN),
        ("imag_C_size200m", "size200m 157M, hybrid C, DreamerV3 loss", "#8e44ad"),
        ("imag_rooms1280_nano_os", "nano 1.26M, 1280 indoor, + overshoot", BLUE)]
SPLITS = [("train", "training rollouts", ":"), ("held_out", "held-out rollouts", "-"), ("unseen", "unseen houses 16-23", "--")]


def train_log(tag):
    s = open(os.path.join(EV, f"dreamer_{tag}.log"), encoding="utf-8").read()
    rows = []
    for m in re.finditer(r"^\[\s*(\d+)\] (rec=.*?)  \(", s, re.M):
        d = {"step": int(m.group(1))}
        for k, v in re.findall(r"(\w+)=([-\d.]+)", m.group(2)):
            d[k] = float(v)
        rows.append(d)
    batch = re.search(r"--batch (\d+)", s)
    return rows


def main():
    matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
    runs = []
    for tag, label, color in RUNS:
        p = os.path.join(EV, f"dreamer_{tag}_imag.json")
        if not os.path.exists(p):
            print(f"[skip] {tag}: no {p}")
            continue
        j = json.load(open(p))
        runs.append((tag, label, color, j, train_log(tag)))
    fig, axes = plt.subplots(3, 3, figsize=(16, 13.5), facecolor=SURFACE)
    fig.suptitle("Dreamer imagination only (no collision / distance heads): training curves and test curves", fontsize=13, fontweight="bold", color=INK, y=0.985)
    ax = axes.flatten()
    for a in ax:
        a.set_facecolor(SURFACE)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.grid(color="#e8e7e3", linewidth=0.7)
        a.tick_params(labelsize=8, colors=INK2)
    for tag, label, color, j, tl in runs:
        b = j["batch"]
        xs = np.array([r["step"] for r in tl]) * b / 1000
        ax[0].plot(xs, [r["rec"] / PIX for r in tl], color=color, linewidth=1.6, label=label)
        ax[1].plot(xs, [r["dyn"] for r in tl], color=color, linewidth=1.6, label=label)
        if "os_rec" in tl[0]:
            ax[2].plot(xs, [r["os_rec"] / PIX for r in tl], color=color, linewidth=1.6, label=label + " (imagined 1-8 steps)")
            ax[1].plot(xs, [r["os_kl"] for r in tl], color=color, linewidth=1.0, linestyle="--")
        h = j["history"]
        hx = np.array([r["step"] for r in h]) * b / 1000
        for i, (sk, sl, _) in enumerate(SPLITS):
            ax[3 + i].plot(hx, [r[sk]["gain"]["32"] if "32" in r[sk]["gain"] else r[sk]["gain"][32] for r in h], color=color, linewidth=1.8, marker="o", markersize=2.5, label=label)
        for sk, sl, ls in SPLITS:
            ax[6].plot(hx, [r[sk]["recon"] for r in h], color=color, linestyle=ls, linewidth=1.4)
            ax[7].plot(hx, [r[sk]["kl"] for r in h], color=color, linestyle=ls, linewidth=1.4)
        kept = j["kept_step"]
        row = min(h, key=lambda r: abs(r["step"] - kept))
        for sk, sl, ls in SPLITS:
            ks = sorted(int(k) for k in row[sk]["gain"])
            ax[8].plot(ks, [row[sk]["gain"][str(k)] if str(k) in row[sk]["gain"] else row[sk]["gain"][k] for k in ks], color=color, linestyle=ls, linewidth=1.8, marker="o", markersize=3)
        ax[3 + 1].axvline(kept * b / 1000, color=color, linewidth=0.8, alpha=0.5)
    titles = ["(a) train loss: reconstruction MSE per pixel (log)", "(b) train loss: KL dyn, posterior vs prior (nats; dashed = overshoot KL)",
              "(c) train loss: overshoot, imagined-frame MSE per pixel (log)",
              "(d) test on TRAINING rollouts: gain @+32 frames", "(e) test on HELD-OUT rollouts: gain @+32 frames (vertical = kept checkpoint)",
              "(f) test on UNSEEN houses: gain @+32 frames",
              "(g) posterior reconstruction MSE per pixel on the test sets", "(h) KL posterior vs prior on the test sets (nats)",
              "(i) kept checkpoint: gain vs imagination horizon"]
    for a, t in zip(ax, titles):
        a.set_title(t, fontsize=9, color=INK, loc="left")
    for i in range(8):
        ax[i].set_xlabel("training sequences seen (thousands; steps x batch)", fontsize=8, color=INK2)
    ax[8].set_xlabel("imagined frames ahead (48 Hz)", fontsize=8, color=INK2)
    for i in (0, 2, 6):
        ax[i].set_yscale("log")
    for i in (3, 4, 5, 8):
        ax[i].axhline(0, color=MUTED, linewidth=0.8, linestyle=":")
        ax[i].set_ylim(-0.2, 1.0)
        ax[i].set_ylabel("gain (0 = copy last frame, 1 = perfect)", fontsize=8, color=INK2)
    ax[0].legend(fontsize=7.5, frameon=False, loc="upper right")
    ax[2].legend(fontsize=7, frameon=False, loc="upper right")
    from matplotlib.lines import Line2D
    ax[8].legend([Line2D([], [], color=INK2, linestyle=ls) for _, _, ls in SPLITS], [sl for _, sl, _ in SPLITS], fontsize=7.5, frameon=False, loc="upper right")
    ax[6].legend([Line2D([], [], color=INK2, linestyle=ls) for _, _, ls in SPLITS], [sl for _, sl, _ in SPLITS], fontsize=7.5, frameon=False, loc="upper right")
    fig.text(0.012, 0.006, "Imagination test: observe 8 steps (667 ms), then imagine 8 steps from memory and the flown commands; decoded frames vs true frames. "
             "\nHeld-out = the 20% of rollouts of the training data never trained on; unseen = houses 16-23. Every run uses the same unseen set; the held-out sets of C and of the 1280-rollout data differ.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    path = os.path.join(REP, "imagination_curves.png")
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    print("CURVES OK ->", path, [r[0] for r in runs])


if __name__ == "__main__":
    main()
