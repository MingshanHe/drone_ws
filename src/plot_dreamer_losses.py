"""Training curves of every nano-Dreamer run from src/output/evals/dreamer_<tag>.log:
the seven loss terms per step (train) and the held-out AUC@32 / decoded gain@32 at the eval points.

  python src/plot_dreamer_losses.py [--tags pybullet rooms ...]   -> src/output/report/dreamer_losses.png
"""

import argparse
import glob
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
from draw_pipeline import BLUE, GRAY, GREEN, INK, INK2, MUTED, ORANGE, SURFACE  # noqa: E402

PALETTE = [BLUE, ORANGE, GREEN, "#8e44ad", "#c0392b", "#111111", "#7f8c8d", "#d4a017"]
TERMS = [("rec", "reconstruction MSE (sum over pixels)", True), ("dyn", "KL dyn (nats, free 1)", False), ("rep", "KL rep", False), ("dist", "clearance two-hot CE", False),
         ("col", "collision BCE", True), ("cont", "continue BCE", True), ("cf", "counterfactual BCE", True)]


def parse(tag):
    p = os.path.join(EV, f"dreamer_{tag}.log")
    if not os.path.exists(p):
        return None
    s = open(p, encoding="utf-8").read()
    steps = re.findall(r"\[\s*(\d+)\] rec=([\d.]+) dist=([\d.]+) col=([\d.]+) cont=([\d.]+) dyn=([\d.]+) rep=([\d.]+) cf=([\d.]+)", s)
    if not steps:
        return None
    arr = np.array(steps, dtype=float)
    train = dict(step=arr[:, 0], rec=arr[:, 1], dist=arr[:, 2], col=arr[:, 3], cont=arr[:, 4], dyn=arr[:, 5], rep=arr[:, 6], cf=arr[:, 7])
    ev = re.findall(r"\[EVAL\s+(\d+)\] now-AUC=([\d.]+), open-loop AUC@4/8/16/32=[\d./]+/([\d.]+), gain@8/@32=(-?[\d.]+)/(-?[\d.]+)", s)
    fin = re.search(r"DREAMER OK \[.*?\]: (\d+) steps, now-AUC=([\d.]+), open-loop AUC@4/8/16/32=[\d./]+/([\d.]+), gain@8/@32=(-?[\d.]+)/(-?[\d.]+)", s)
    pts = [(int(a), float(c), float(e)) for a, b, c, d, e in ev] + ([(int(fin.group(1)), float(fin.group(3)), float(fin.group(5)))] if fin else [])
    params = re.search(r"params ([\d.]+)M", s)
    return dict(train=train, evals=sorted(pts), params=params.group(1) if params else "?")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--out", default="dreamer_losses.png")
    args = ap.parse_args()
    tags = args.tags or sorted(os.path.basename(p)[8:-4] for p in glob.glob(os.path.join(EV, "dreamer_*.log")) if "_cluttered" not in p and "@" not in p and "offline" not in p and "video" not in p)
    runs = {t: parse(t) for t in tags}
    runs = {t: r for t, r in runs.items() if r}
    matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
    matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(3, 3, figsize=(15, 10.5), facecolor=SURFACE)
    fig.suptitle("nano-Dreamer training curves: training losses every 250 steps (first 7 panels) and held-out evaluations (last 2)", fontsize=12.5, fontweight="bold", color=INK, y=0.985)
    axes = axes.flatten()
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(color="#e8e7e3", linewidth=0.7)
        ax.tick_params(labelsize=8, colors=INK2)
    for i, (key, title, logy) in enumerate(TERMS):
        ax = axes[i]
        for j, (t, r) in enumerate(runs.items()):
            ax.plot(r["train"]["step"], r["train"][key], color=PALETTE[j % len(PALETTE)], linewidth=1.4, label=f"{t} ({r['params']}M)")
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("step", fontsize=8, color=INK2)
    for i, (key, title) in enumerate(((1, "held-out open-loop AUC@32"), (2, "held-out decoded gain @32"))):
        ax = axes[7 + i]
        for j, (t, r) in enumerate(runs.items()):
            if r["evals"]:
                ax.plot([e[0] for e in r["evals"]], [e[key] for e in r["evals"]], color=PALETTE[j % len(PALETTE)], linewidth=1.6, marker="o", markersize=4, label=t)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("step", fontsize=8, color=INK2)
        if key == 1:
            ax.set_ylim(0.6, 1.02)
        else:
            ax.axhline(0, color=MUTED, linewidth=0.8, linestyle=":")
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path = os.path.join(REP, args.out)
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    print("LOSSES OK ->", path, "runs:", list(runs))


if __name__ == "__main__":
    main()
