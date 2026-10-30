"""Train vs held-out vs unseen: how well does a Dreamer imagine the rollouts it was trained on?

Both checkpoints were trained on the hybrid dataset C (wm_dataset_pb_plus_rooms.npz) with the same
rollout-level split (seed 0). For each model the open-loop imagination (observe 8 steps, then imagine
from memory and the flown commands) is scored on five groups of sequences:
  train rooms / held-out rooms / train pybullet / held-out pybullet  (from C's own split)
  unseen rooms (houses 16-23, never used for training)
Training groups are subsampled to 64 rollouts so every group has a similar size.

  python src/eval_train_gap.py   -> src/output/report/train_gap.png, train_gap.json
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "output")
REP = os.path.join(OUT, "report")

from draw_pipeline import BLUE, GREEN, INK, INK2, MUTED, ORANGE, SURFACE  # noqa: E402
from dreamer.data import STRIDE  # noqa: E402
from eval_imagination import SHOW_STEPS, run_case  # noqa: E402
from gen_wm_dataset import HORIZONS  # noqa: E402
from train_world_model import _split_rollouts  # noqa: E402

PURPLE, GOLD = "#8e44ad", "#d4a017"
DATA_C = "wm_dataset_pb_plus_rooms.npz"
UNSEEN = "wm_dataset_rooms_unseen60.npz"
MODELS = [("size200m", "DreamerV3 size200m (157M)", "world_model_dreamer_pb_plus_rooms_size200m.pth", "-"),
          ("nano", "nano C 6k (1.26M)", "world_model_dreamer_pb_plus_rooms6k.pth", "--")]
GROUPS = [("train_rooms", "train rooms", PURPLE), ("val_rooms", "held-out rooms", ORANGE), ("unseen_rooms", "unseen rooms 16-23", GREEN),
          ("train_pb", "train pybullet", BLUE), ("val_pb", "held-out pybullet", GOLD)]
STRIPS = ["train_rooms", "val_rooms", "unseen_rooms"]


def groups():
    blob = np.load(os.path.join(OUT, DATA_C))
    st = blob["scene_type"]
    tr, va = _split_rollouts({k: blob[k] for k in ("frames", "in_path", "seg")}, np.random.default_rng(0))
    rng = np.random.default_rng(1)
    pick = lambda rs, n=64: sorted(rng.choice(rs, min(n, len(rs)), replace=False).tolist())
    g = {"train_rooms": (DATA_C, pick([r for r in tr if st[r] == 1])), "val_rooms": (DATA_C, [r for r in va if st[r] == 1]),
         "train_pb": (DATA_C, pick([r for r in tr if st[r] == 0])), "val_pb": (DATA_C, [r for r in va if st[r] == 0]),
         "unseen_rooms": (UNSEEN, list(range(64)))}
    return g


def summarise(r):
    mp, mc = np.array(r["mse_pred"]), np.array(r["mse_copy"])
    return dict(n_seq=r["n_seq"], mse32=float(mp[7]), copy32=float(mc[7]), gain32=float(1 - mp[7] / mc[7]), mse88=float(mp[-1]), gain88=float(1 - mp[-1] / mc[-1]),
                mae32=float(r["mae_pred"][7]), persist32=float(r["mae_persist"][7]), auc32=float(r["auc"][7]), auc8=float(r["auc"][1]))


def figure(res, strips):
    matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
    ncol = len(SHOW_STEPS) + 1
    H = 15.5
    fig = plt.figure(figsize=(16, H), facecolor=SURFACE)
    gs = fig.add_gridspec(6, ncol, hspace=0.34, wspace=0.08, left=0.04, right=0.985, top=1 - 0.95 / H, bottom=5.6 / H)
    gs2 = fig.add_gridspec(1, 3, wspace=0.28, left=0.05, right=0.985, top=4.4 / H, bottom=1.0 / H)
    fig.suptitle("DreamerV3 size200m on its own training data vs held-out vs unseen: open-loop imagination after 8 observed steps", fontsize=13, fontweight="bold", color=INK, y=1 - 0.35 / H)
    colors = dict((k, c) for k, _, c in GROUPS)
    labels = dict((k, l) for k, l, _ in GROUPS)
    for ci, key in enumerate(STRIPS):
        st, color = strips[key], colors[key]
        for row, (imgs, ds, name) in enumerate(((st["true"], st["d_true"], "true"), (st["imag"], st["d_imag"], "imagined"))):
            r = 2 * ci + row
            ax = fig.add_subplot(gs[r, 0])
            ax.imshow(st["ctx"], interpolation="nearest")
            ax.set_title(f"rollout {st['roll']}  last obs  d={st['d_ctx']:.2f}" if row == 0 else "(same frame)", fontsize=7.3, color=INK2 if row == 0 else MUTED)
            ax.set_ylabel(f"{labels[key]}\n{name}", fontsize=8.5, color=color if row == 1 else INK2, fontweight="bold" if row == 1 else "normal")
            ax.set_xticks([]), ax.set_yticks([])
            for k, (img, dv) in enumerate(zip(imgs, ds)):
                ax = fig.add_subplot(gs[r, k + 1])
                ax.imshow(img, interpolation="nearest")
                ax.set_title(f"+{SHOW_STEPS[k] * STRIDE} frames  d={dv:.2f}" if row == 0 else f"$\\hat d$={dv:.2f}  P<0.7={st['p_imag'][k]:.2f}", fontsize=7.3, color=INK2 if row == 0 else color)
                ax.set_xticks([]), ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_edgecolor(color if row == 1 else INK2), sp.set_linewidth(1.2 if row == 1 else 0.6)
    axes = [fig.add_subplot(gs2[0, i]) for i in range(3)]
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#e8e7e3", linewidth=0.7)
        ax.tick_params(labelsize=8, colors=INK2)
    for mkey, mlabel, _, ls in MODELS:
        for gkey, glabel, color in GROUPS:
            r = res[mkey][gkey]
            h = np.arange(1, len(r["mse_pred"]) + 1) * STRIDE
            gain = 1 - np.array(r["mse_pred"]) / np.array(r["mse_copy"])
            lab = f"{glabel}" if mkey == "size200m" else None
            axes[0].plot(h, gain, color=color, linestyle=ls, linewidth=2 if ls == "-" else 1.3, label=lab)
            axes[1].plot(h, r["mae_pred"], color=color, linestyle=ls, linewidth=2 if ls == "-" else 1.3, label=lab)
            axes[2].plot(np.arange(1, 9) * STRIDE, r["auc"], color=color, linestyle=ls, linewidth=2 if ls == "-" else 1.3, marker="o" if ls == "-" else None, markersize=3.5, label=lab)
    axes[0].axhline(0, color=MUTED, linewidth=0.8, linestyle=":")
    axes[0].set_title("(a) decoded gain vs horizon (1 − MSE imagined / MSE copy)", fontsize=9.5, color=INK, loc="left")
    axes[0].set_xlabel("imagined frames ahead (48 Hz)", fontsize=8.5, color=INK2)
    axes[1].set_title("(b) clearance error of the distance head (m)", fontsize=9.5, color=INK, loc="left")
    axes[1].set_xlabel("imagined frames ahead", fontsize=8.5, color=INK2)
    axes[2].set_title("(c) collision foresight AUC (< 0.7 m within k)", fontsize=9.5, color=INK, loc="left")
    axes[2].set_xlabel("horizon k (frames)", fontsize=8.5, color=INK2)
    axes[0].legend(fontsize=7.5, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.66), title="solid = size200m, dashed = nano C 6k", title_fontsize=7.5)
    fig.text(0.04, 0.35 / H, "Groups come from the hybrid dataset C's own rollout split (seed 0); training groups subsampled to 64 rollouts. Strips: in each group the sequence whose closest approach is ≈ 0.5 m.", fontsize=7.8, color=MUTED)
    path = os.path.join(REP, "train_gap.png")
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return path


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = groups()
    res, strips, table = {}, {}, {}
    for mkey, mlabel, ckpt, _ in MODELS:
        res[mkey], table[mkey] = {}, {}
        for gkey, (dataset, rolls) in g.items():
            r, st = run_case(gkey, ckpt, dataset, False, device, rolls=rolls)
            res[mkey][gkey] = r
            table[mkey][gkey] = summarise(r)
            if mkey == "size200m":
                strips[gkey] = st
            t = table[mkey][gkey]
            print(f"[{mkey:8s} {gkey:12s}] {t['n_seq']:4d} seqs  gain@32 {t['gain32']:+.2f}  gain@88 {t['gain88']:+.2f}  MSE@32 {t['mse32']:.4f} (copy {t['copy32']:.4f})  "
                  f"clearance MAE@32 {t['mae32']:.2f} m (persist {t['persist32']:.2f})  AUC@8/@32 {t['auc8']:.2f}/{t['auc32']:.2f}", flush=True)
            torch.cuda.empty_cache()
    json.dump(table, open(os.path.join(REP, "train_gap.json"), "w"), indent=1)
    print("TRAIN-GAP OK ->", figure(res, strips))


if __name__ == "__main__":
    main()
