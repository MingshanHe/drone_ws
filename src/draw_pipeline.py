"""Paper-style architecture figure of the nano world model (one frame -> future latents -> collision risk).

Everything drawn is real: the frames come from a held-out pybullet window, the latent bars, the
predicted latents, the collision probabilities and the danger-now value are the actual forward pass
of checkpoint A (world_model_pybullet_ref.pth); the label grid is the dataset's distance labels;
the parameter counts are computed from the modules.

Layout: top row = online model (frame -> encoder -> z_t -> predictor -> z_hat -> collision heads -> P grid),
bottom row = targets of the same window (future frames -> EMA encoder -> z_bar, true labels);
every loss is a vertical double arrow between a prediction above and its target below.

  python src/draw_pipeline.py --lang both   -> src/output/report/pipeline_three[_en].png
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "output")
REP = os.path.join(OUT, "report")

from gen_wm_dataset import ACTION_NAMES, A_NORM, DANGER_R, HORIZONS, window_valid  # noqa: E402
from train_world_model import CollisionHeads, DangerNowHead, Encoder, MultiPredictor, RADII, load_model  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
BLUE, ORANGE, GREEN, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9890"
WINDOW = (46, 4)  # rollout, t of the illustrated held-out pybullet window (pillar ahead, command forward)

STR = {
    "zh": dict(font=["WenQuanYi Zen Hei", "DejaVu Sans"], fig="pipeline_three.png",
               title="世界模型：一帧画面 → 未来潜变量 → 碰撞风险（共 {N_total}k 参数 ≈ {KB} KB int8，三种训练集同一结构、同一超参）",
               frame="当前帧 $x_t$  64×64×3", act="命令 $a_t$ (vx, vy, vz, ω)", enc="在线编码器 $f_\\theta$ · {N_enc}k",
               pool="条带池化 1×4", fc="FC 256→64", z="$z_t\\in\\mathbb{R}^{64}$", pred="预测器 · {N_pred}k",
               mlp="68 → 128 → 64，每个视界一个头", zhat="预测潜变量 $\\hat z_{t+k}$", resid="残差 $+\\,z_t$",
               heads="碰撞头", heads2="64→2 ×4 · {N_col}", prob="预测 P(k 步内进入)", lab="真实标签（dists < r）", ring=("≤ 0.7 m", "≤ 0.35 m"),
               bce="BCE", lcol="$L_{col}$：对实飞标签", lcf="$L_{cf}$：对反事实标签", lcf2="（6 个候选命令）", now="danger-now 64→1 · {N_now}",
               pnow="P(现在危险) = {P_now}", lnow="$L_{now}$ (BCE, dists < 0.7 m)", future="未来帧（同一窗口，命令保持不变）", ema="EMA 目标编码器 $f_{\\bar\\theta}$（无梯度）",
               emaupd="$\\bar\\theta \\leftarrow 0.99\\,\\bar\\theta + 0.01\\,\\theta$", zbar="目标 $\\bar z_{t+k}$（stop-grad）",
               lpred="$L_{pred}$ = MSE（潜空间）", var="$L_{var}$ 防塌缩", total="总损失 = $L_{pred}+L_{var}+L_{col}+L_{cf}+0.5\\,L_{now}$   ·   Adam 1e-3，batch 64，80 epoch，按 rollout 切分 80/20",
               use="测试时：对 5 个候选命令各预测一次 → 碰撞头 → 选最安全的（12 Hz）", k=("+4", "+8", "+16", "+32"), ms="k = 83 / 167 / 333 / 667 ms"),
    "en": dict(font=["DejaVu Sans"], fig="pipeline_three_en.png",
               title="Nano world model: one frame → future latents → collision risk  ({N_total}k parameters ≈ {KB} KB int8, same architecture for all three training sets)",
               frame="current frame $x_t$  64×64×3", act="command $a_t$ (vx, vy, vz, ω)", enc="online encoder $f_\\theta$ · {N_enc}k",
               pool="strip pool 1×4", fc="FC 256→64", z="$z_t\\in\\mathbb{R}^{64}$", pred="predictor · {N_pred}k",
               mlp="68 → 128 → 64, one head per horizon", zhat="predicted latents $\\hat z_{t+k}$", resid="residual $+\\,z_t$",
               heads="collision heads", heads2="64→2 ×4 · {N_col}", prob="predicted P(within k steps)", lab="true labels (dists < r)", ring=("≤ 0.7 m", "≤ 0.35 m"),
               bce="BCE", lcol="$L_{col}$: vs flown labels", lcf="$L_{cf}$: vs counterfactual labels", lcf2="(6 candidate commands)", now="danger-now 64→1 · {N_now}",
               pnow="P(danger now) = {P_now}", lnow="$L_{now}$ (BCE, dists < 0.7 m)", future="future frames (same window, command held)", ema="EMA target encoder $f_{\\bar\\theta}$ (no gradient)",
               emaupd="$\\bar\\theta \\leftarrow 0.99\\,\\bar\\theta + 0.01\\,\\theta$", zbar="targets $\\bar z_{t+k}$ (stop-grad)",
               lpred="$L_{pred}$ = latent MSE", var="$L_{var}$ anti-collapse", total="total = $L_{pred}+L_{var}+L_{col}+L_{cf}+0.5\\,L_{now}$   ·   Adam 1e-3, batch 64, 80 epochs, rollout-level 80/20 split",
               use="at test time: predict once per candidate command (5) → collision heads → pick the safest (12 Hz)", k=("+4", "+8", "+16", "+32"), ms="k = 83 / 167 / 333 / 667 ms"),
}


def counts():
    n = lambda m: sum(p.numel() for p in m.parameters())
    e, p, c, d = Encoder(), MultiPredictor(), CollisionHeads(), DangerNowHead()
    return dict(enc=n(e), pred=n(p), col=n(c), now=n(d), total=n(e) + n(p) + n(c) + n(d))


def pick_window(d):
    """First (rollout, t) with the command 'forward' held over 32 steps on an in-path course whose
    labels switch on inside the window: warn ring at +16/+32, critical ring at +32 only."""
    fwd = ACTION_NAMES.index("forward")
    L = d["frames"].shape[1]
    fallback = None
    for r in np.nonzero(d["in_path"])[0]:
        for t in range(0, L - HORIZONS[-1]):
            if int(d["act_id"][r, t]) != fwd or not window_valid(d["seg"][r], t, HORIZONS[-1]):
                continue
            lab = np.array([[float(d["dists"][r, t : t + k + 1].min() < rad) for rad in RADII] for k in HORIZONS])
            if lab[:, 0].tolist() == [0, 0, 1, 1] and lab[:, 1].tolist() == [0, 0, 0, 1]:
                return int(r), int(t)
            if fallback is None and 1 <= lab[:, 0].sum() <= 3:
                fallback = (int(r), int(t))
    return fallback if fallback else WINDOW


def forward_pass():
    """Real tensors for the illustrated window from checkpoint A."""
    d = np.load(os.path.join(OUT, "wm_dataset.npz"))
    r, t = pick_window(d)
    enc, pred, cheads, nhead, meta = load_model(os.path.join(OUT, "world_model_pybullet_ref.pth"), device="cpu")
    x = torch.tensor(d["frames"][r, t][None]).permute(0, 3, 1, 2).float() / 255.0
    xs = [torch.tensor(d["frames"][r, t + k][None]).permute(0, 3, 1, 2).float() / 255.0 for k in HORIZONS]
    a = torch.tensor(d["actions"][r, t][None] / A_NORM, dtype=torch.float32)
    with torch.no_grad():
        z = enc(x)[0]
        zh = pred(z[None], a)[0]  # (H, D)
        p = torch.sigmoid(cheads(zh[None]))[0].numpy()  # (H, rings)
        pnow = float(torch.sigmoid(nhead(z[None]))[0])
        zbar = torch.cat([enc(xk) for xk in xs])  # the EMA weights are not saved: online encoder stands in for the picture
    labels = np.array([[float(d["dists"][r, t : t + k + 1].min() < rad) for rad in RADII] for k in HORIZONS])
    return dict(window=(r, t), frame=d["frames"][r, t], future=[d["frames"][r, t + k] for k in HORIZONS], action=d["actions"][r, t] / A_NORM,
                act_name=ACTION_NAMES[int(d["act_id"][r, t])], z=z.numpy(), zh=zh.numpy(), zbar=zbar.numpy(), p=p, pnow=pnow, labels=labels)


# ---------------------------------------------------------------- glyphs
def fmap(ax, x, y, w, h, depth, color, alpha=0.18):
    """A feature-map block drawn as a shallow 3-D slab (front w×h, depth = channels)."""
    dx, dy = depth * 0.55, depth * 0.32
    ax.add_patch(Polygon([(x, y + h), (x + dx, y + h + dy), (x + w + dx, y + h + dy), (x + w, y + h)], closed=True, facecolor=color, alpha=alpha * 0.7, edgecolor=color, linewidth=0.9))
    ax.add_patch(Polygon([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)], closed=True, facecolor=color, alpha=alpha * 0.5, edgecolor=color, linewidth=0.9))
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, alpha=alpha, edgecolor=color, linewidth=1.0))


def vbar(ax, x, y, w, h, values, cmap, vmin=None, vmax=None):
    v = np.asarray(values, dtype=float)[:, None]
    ax.imshow(v, extent=(x, x + w, y, y + h), cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax, interpolation="nearest", zorder=3)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=INK2, linewidth=0.7, zorder=4))


def grid(ax, x, y, cw, ch, values, cmap, vmin, vmax, text=True, fmt="{:.2f}", fontsize=9):
    H, W = values.shape
    for i in range(H):
        for j in range(W):
            v = values[i, j]
            ax.add_patch(Rectangle((x + j * cw, y + (H - 1 - i) * ch), cw, ch, facecolor=cmap((v - vmin) / (vmax - vmin)), edgecolor=SURFACE, linewidth=1.5, zorder=3))
            if text:
                ax.text(x + j * cw + cw / 2, y + (H - 1 - i) * ch + ch / 2, fmt.format(v), ha="center", va="center", fontsize=fontsize, zorder=4,
                        color=INK if (v - vmin) / (vmax - vmin) < 0.55 else "white")


def arr(ax, p, q, color=INK2, lw=1.2, rad=0.0, dashed=False, zorder=5):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, linewidth=lw, color=color, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=(0, (4, 2.5)) if dashed else "-", zorder=zorder))


def darr(ax, p, q, color=GREEN, lw=1.3):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>", mutation_scale=10, linewidth=lw, color=color, shrinkA=0, shrinkB=0, zorder=5))


def thumb(ax, img, x, y, w, h):
    ax.imshow(img, extent=(x, x + w, y, y + h), interpolation="nearest", zorder=3)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=INK2, linewidth=0.7, zorder=4))


def draw(lang, k, fp):
    subs = {"{N_total}": f"{k['total']/1e3:.0f}", "{KB}": f"{k['total']/1024:.0f}", "{N_enc}": f"{k['enc']/1e3:.1f}", "{N_pred}": f"{k['pred']/1e3:.1f}",
            "{N_col}": str(k["col"]), "{N_now}": str(k["now"]), "{P_now}": f"{fp['pnow']:.2f}"}

    def sub(v):
        for a, b in subs.items():
            v = v.replace(a, b)
        return v

    t = {key: (sub(v) if isinstance(v, str) else v) for key, v in STR[lang].items()}
    matplotlib.rcParams["font.family"] = t["font"]
    W, H = 139, 82
    fig, ax = plt.subplots(figsize=(W / 10, H / 10), facecolor=SURFACE)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(SURFACE)
    fig.text(0.012, 0.985, t["title"], fontsize=11.5, color=INK, va="top", fontweight="bold")
    blues, oranges, greys = plt.get_cmap("Blues"), plt.get_cmap("Oranges"), plt.get_cmap("Greys")
    zlo, zhi = float(min(fp["z"].min(), fp["zh"].min(), fp["zbar"].min())), float(max(fp["z"].max(), fp["zh"].max(), fp["zbar"].max()))
    small = dict(fontsize=9, color=INK2, ha="center")
    tiny = dict(fontsize=8.5, color=MUTED, ha="center")
    head = dict(fontsize=10, ha="center", fontweight="bold")
    conv = dict(fontsize=8, color=MUTED, ha="center")

    # ================= top row: online model
    yc = 60
    thumb(ax, fp["frame"], 1.5, yc - 7, 14, 14)
    ax.text(8.5, yc + 8.3, t["frame"], **small)
    # encoder: three slabs, strip pool, FC
    arr(ax, (15.8, yc), (19.7, yc), BLUE)
    ax.text(17.8, yc + 1.3, "5×5/2", **conv)
    fmap(ax, 20, yc - 5, 10, 10, 1.6, BLUE)
    ax.text(25.5, yc - 6.7, "16 @ 32²", **tiny)
    arr(ax, (31.9, yc), (35.2, yc), BLUE)
    ax.text(33.5, yc + 1.3, "3×3/2", **conv)
    fmap(ax, 35.5, yc - 3.5, 7, 7, 2.6, BLUE)
    ax.text(39.5, yc - 5.2, "32 @ 16²", **tiny)
    arr(ax, (45.3, yc), (48.5, yc), BLUE)
    ax.text(46.9, yc + 1.3, "3×3/2", **conv)
    fmap(ax, 48.8, yc - 2.5, 5, 5, 4.0, BLUE)
    ax.text(51.8, yc - 4.2, "64 @ 8²", **tiny)
    arr(ax, (56.2, yc), (59.3, yc), BLUE)
    for j in range(4):  # strip pool: 4 horizontal strips
        ax.add_patch(Rectangle((59.6 + j * 1.3, yc - 2.5), 1.3, 5, facecolor=BLUE, alpha=0.18 + 0.12 * j, edgecolor=BLUE, linewidth=0.7, zorder=3))
    ax.text(62.2, yc - 4.2, t["pool"], **tiny)
    arr(ax, (65.2, yc), (69.4, yc), BLUE)
    ax.text(69.5, yc + 3.2, t["fc"], fontsize=8, color=MUTED, ha="right")
    ax.text(38, yc + 9.4, t["enc"], color=BLUE, **head)
    # z_t and the command vector under it
    zx = 69.8
    vbar(ax, zx, yc - 9, 2.6, 18, fp["z"], blues, zlo, zhi)
    ax.text(zx + 1.3, yc - 10.2, t["z"], fontsize=9, color=INK2, ha="center", va="top")
    for j, v in enumerate(fp["action"]):
        ax.add_patch(Rectangle((zx, yc - 17.2 + j * 1.1), 2.6, 1.1, facecolor=blues(0.25 + 0.6 * abs(float(v))), edgecolor=SURFACE, linewidth=1, zorder=3))
    ax.add_patch(Rectangle((zx, yc - 17.2), 2.6, 4.4, fill=False, edgecolor=INK2, linewidth=0.7, zorder=4))
    ax.text(zx - 1.2, yc - 14.2, t["act"], fontsize=9, color=INK2, ha="right", va="center")
    ax.text(zx - 1.2, yc - 16.4, f"({fp['act_name']})", fontsize=8.5, color=MUTED, ha="right", va="center")
    # predictor
    px = 77.5
    arr(ax, (zx + 2.9, yc), (px - 0.3, yc), BLUE)
    arr(ax, (zx + 2.9, yc - 15), (px - 0.3, yc - 7.5), BLUE)  # a_t into the predictor
    ax.add_patch(FancyBboxPatch((px, yc - 9), 14, 18, boxstyle="round,pad=0,rounding_size=1.2", facecolor=ORANGE, alpha=0.10, edgecolor=ORANGE, linewidth=1.1, zorder=2))
    cols = [(px + 3, 5), (px + 7, 8), (px + 11, 5)]
    pts = {}
    for ci, (cx, n) in enumerate(cols):
        ys = np.linspace(yc - 6, yc + 6, n)
        pts[ci] = [(cx, yy) for yy in ys]
        for yy in ys:
            ax.add_patch(plt.Circle((cx, yy), 0.48, facecolor=ORANGE, edgecolor=SURFACE, linewidth=0.5, zorder=4))
    for ci in (0, 1):
        for p in pts[ci]:
            for q in pts[ci + 1]:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=ORANGE, alpha=0.25, linewidth=0.6, zorder=3)
    ax.text(px + 7, yc - 10.2, t["pred"], color=ORANGE, va="top", **head)
    ax.text(px - 0.3, yc - 12.6, t["mlp"], fontsize=8.5, color=MUTED, ha="left", va="top")
    # predicted latents (4 bars)
    bx = 95
    arr(ax, (px + 14.3, yc), (bx - 0.4, yc), ORANGE)
    for j in range(4):
        vbar(ax, bx + j * 3.3, yc - 9, 2.6, 18, fp["zh"][j], oranges, zlo, zhi)
        ax.text(bx + 1.3 + j * 3.3, yc + 9.6, t["k"][j], fontsize=8.5, color=INK2, ha="center", va="bottom")
    bcx = bx + 6.2  # centre of the four bars
    ax.text(bcx, yc + 12.4, t["zhat"], va="bottom", **small)
    arr(ax, (zx + 2.8, yc + 7.4), (bx - 0.4, yc + 8.8), BLUE, rad=-0.3, dashed=True)  # residual
    ax.text((zx + bx) / 2 + 1, yc + 14.3, t["resid"], fontsize=8.5, color=BLUE, ha="center", va="bottom")
    # collision heads -> probability grid
    gx, gy, cw, ch = 121.5, yc - 9, 6.5, 4.5
    arr(ax, (bx + 12.8, yc), (gx - 0.4, yc), ORANGE)
    hx = (bx + 12.8 + gx) / 2
    ax.text(hx, yc + 1.5, t["heads"], fontsize=9, color=ORANGE, ha="center", va="bottom", fontweight="bold")
    ax.text(hx, yc - 1.5, t["heads2"], va="top", **tiny)
    grid(ax, gx, gy, cw, ch, fp["p"], oranges, 0.0, 1.0)
    for j, s in enumerate(t["ring"]):
        ax.text(gx + cw * (j + 0.5), gy + 4 * ch + 0.5, s, fontsize=8.5, color=INK2, ha="center", va="bottom")
    for i, s in enumerate(t["k"]):
        ax.text(gx + 2 * cw + 0.6, gy + (3 - i) * ch + ch / 2, s, fontsize=8.5, color=INK2, ha="left", va="center")
    gcx = gx + cw
    ax.text(gcx, yc + 12.4, t["prob"], va="bottom", **small)
    # danger-now head above z_t
    arr(ax, (zx + 1.3, yc + 9.2), (zx + 1.3, yc + 11.8), BLUE)
    ax.add_patch(Rectangle((zx - 0.4, yc + 12.1), 3.4, 2.8, facecolor=blues(0.2 + 0.7 * fp["pnow"]), edgecolor=INK2, linewidth=0.7, zorder=3))
    ax.text(zx - 1.4, yc + 14.6, t["now"] + "    " + t["pnow"], fontsize=9, color=BLUE, ha="right", va="center")
    ax.text(zx - 1.4, yc + 12.2, t["lnow"], fontsize=8.5, color=GREEN, ha="right", va="center")

    # ================= bottom row: targets of the same window
    yb = 20
    for j, img in enumerate(fp["future"]):
        thumb(ax, img, 2 + j * 10.5, yb - 4.5, 9, 9)
        ax.text(6.5 + j * 10.5, yb - 5.6, f"$x_{{t{t['k'][j]}}}$", fontsize=9, color=INK2, ha="center", va="top")
    ax.text(22.5, yb + 6.0, t["future"], va="bottom", **small)
    arr(ax, (43.8, yb), (47, yb), GRAY)
    fmap(ax, 47.3, yb - 3.75, 7.5, 7.5, 1.2, GRAY)
    arr(ax, (56.1, yb), (57.5, yb), GRAY)
    fmap(ax, 57.8, yb - 2.75, 5.5, 5.5, 2.0, GRAY)
    arr(ax, (65.4, yb), (66.7, yb), GRAY)
    fmap(ax, 67, yb - 1.9, 3.8, 3.8, 3.0, GRAY)
    ax.text(60, yb - 6.6, t["ema"], color=INK2, va="top", **head)
    ax.text(60, yb - 9.4, t["emaupd"], va="top", **tiny)
    arr(ax, (33, yc - 7.6), (50.5, yb + 5.4), GRAY, dashed=True)  # weights copied by EMA
    arr(ax, (73.3, yb), (bx - 0.4, yb), GRAY)
    for j in range(4):
        vbar(ax, bx + j * 3.3, yb - 9, 2.6, 18, fp["zbar"][j], greys, zlo, zhi)
    ax.text(bcx, yb - 10.4, t["zbar"], va="top", **small)
    grid(ax, gx, yb - 9, cw, ch, fp["labels"], greys, -0.6, 1.8, fmt="{:.0f}")
    for j, s in enumerate(t["ring"]):
        ax.text(gx + cw * (j + 0.5), yb + 9.5, s, fontsize=8.5, color=INK2, ha="center", va="bottom")
    for i, s in enumerate(t["k"]):
        ax.text(gx + 2 * cw + 0.6, yb - 9 + (3 - i) * ch + ch / 2, s, fontsize=8.5, color=INK2, ha="left", va="center")
    ax.text(gcx, yb - 10.4, t["lab"], va="top", **small)

    # ================= losses: vertical double arrows between prediction (top) and target (bottom)
    darr(ax, (bcx, yc - 9.6), (bcx, yb + 9.6))
    ax.text(bcx - 1.8, 43.4, t["lpred"], fontsize=10, color=GREEN, ha="right", va="center", fontweight="bold")
    ax.text(bcx - 1.8, 40.6, t["var"], fontsize=9, color=GREEN, ha="right", va="center")
    ax.text(bcx - 1.8, 38.1, t["ms"], fontsize=8.5, color=MUTED, ha="right", va="center")
    darr(ax, (gcx, yc - 9.6), (gcx, yb + 11.6))
    ax.text(gcx - 1.8, 43.4, t["bce"], fontsize=10, color=GREEN, ha="right", va="center", fontweight="bold")
    ax.text(gcx - 1.8, 40.6, t["lcol"], fontsize=9, color=GREEN, ha="right", va="center")
    ax.text(gcx - 1.8, 38.1, t["lcf"], fontsize=9, color=GREEN, ha="right", va="center")
    ax.text(gcx - 1.8, 35.7, t["lcf2"], fontsize=8.5, color=MUTED, ha="right", va="center")

    # footer
    ax.text(1, 3, t["total"], fontsize=9, color=INK2, ha="left", va="center")
    ax.text(W - 1, 3, t["use"], fontsize=9, color=INK2, ha="right", va="center")

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path = os.path.join(REP, t["fig"])
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lang", choices=["zh", "en", "both"], default="both")
    args = ap.parse_args()
    k, fp = counts(), forward_pass()
    print(f"[INFO] window r{fp['window'][0]} t{fp['window'][1]} ({fp['act_name']}): P(warn)@k = {np.round(fp['p'][:, 0], 2).tolist()}, labels = {fp['labels'][:, 0].tolist()}, P(now) = {fp['pnow']:.2f}")
    for lang in (["zh", "en"] if args.lang == "both" else [args.lang]):
        print("PIPELINE OK ->", draw(lang, k, fp))


if __name__ == "__main__":
    main()
