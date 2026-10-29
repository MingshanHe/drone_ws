"""How good is the nano-Dreamer's imagination? One figure (PLAN_dreamerv3.md, M2 check).

Top: long open-loop rollouts on a held-out pybullet sequence (model A) and an UNSEEN ProcTHOR
room (model C 6k): 8 observed steps, then the prior imagines +4 ... +88 frames under the flown
commands without seeing another frame; true frames above, decoded imagination below.
Bottom, averaged over every held-out / unseen sequence (4 phases each):
  (a) decoded-frame MSE vs imagined horizon, against the "copy the last observed frame" baseline
  (b) clearance error of the dist head vs horizon, against "clearance stays what it was"
  (c) collision foresight AUC ("inside 0.7 m within k") vs horizon, against the one-frame JEPA models
  (d) decoded gain at +32 as a function of how many steps were observed first (the value of memory)

  python src/eval_imagination.py   -> src/output/report/imagination_eval.png, imagination_eval.json
"""

import json
import os
import re
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
EV = os.path.join(OUT, "evals")

from draw_pipeline import BLUE, GRAY, GREEN, INK, INK2, MUTED, ORANGE, SURFACE  # noqa: E402
from dreamer.data import STRIDE, SeqData  # noqa: E402
from gen_wm_dataset import DANGER_R, HORIZONS  # noqa: E402
from train_dreamer import load_dreamer  # noqa: E402
from train_world_model import _split_rollouts, roc_auc  # noqa: E402

CASES = [  # key, label, checkpoint, dataset, held-out split?, colour
    ("pybullet", "pybullet held-out (model A)", "world_model_dreamer_pybullet.pth", "wm_dataset.npz", True, BLUE),
    ("rooms_unseen", "unseen rooms 16–23 (model C 6k)", "world_model_dreamer_pb_plus_rooms6k.pth", "wm_dataset_rooms_unseen60.npz", False, ORANGE),
]
EXTRA = [  # optional variants, drawn as extra curves (no strip) when their checkpoint exists
    ("rooms_unseen_1280", "unseen rooms, 1280 rollouts + overshoot (B)", "world_model_dreamer_rooms1280_os.pth", "wm_dataset_rooms_unseen60.npz", False, GREEN),
]
CONTEXT = 8
SHOW_STEPS = [1, 2, 4, 6, 8, 12, 16, 22]  # imagined steps shown in the strips (x4 = frames)
CONTEXTS = [1, 2, 4, 8, 12]


@torch.no_grad()
def run_case(key, ckpt, dataset, heldout, device):
    model, _ = load_dreamer(os.path.join(OUT, ckpt), device)
    blob = np.load(os.path.join(OUT, dataset))
    data = {k: blob[k] for k in blob.files}
    rolls = _split_rollouts(data, np.random.default_rng(0))[1] if heldout else list(range(data["frames"].shape[0]))
    sd = SeqData(data, device)
    ep = sd.episodes(rolls)
    x, d = ep["frames"], ep["dists"]
    N, T = x.shape[:2]
    post, _ = model.rssm.observe(model.enc(x.flatten(0, 1)).view(N, T, -1), ep["a_prev"])
    res = {"n_seq": int(N)}

    def imagine_from(c):
        st = model.rssm.index(post, slice(None), c - 1)
        im = model.rssm.imagine(st, ep["actions"][:, c - 1 : T - 1])
        return st, model.rssm.feat(im)  # (N, T-c, F)

    # (a)(b)(c) from CONTEXT
    st, fim = imagine_from(CONTEXT)
    H = T - CONTEXT
    mse_p, mse_0, mae_p, mae_0 = [], [], [], []
    for h in range(H):
        xh = model.dec(fim[:, h]).float().clamp(0, 1)
        xt = x[:, CONTEXT + h]
        mse_p.append(float(((xh - xt) ** 2).mean()))
        mse_0.append(float(((x[:, CONTEXT - 1] - xt) ** 2).mean()))
        dh = model.twohot.mean(model.dist_head(fim[:, h]))
        mae_p.append(float((dh - d[:, CONTEXT + h]).abs().mean()))
        mae_0.append(float((d[:, CONTEXT - 1] - d[:, CONTEXT + h]).abs().mean()))
    res.update(mse_pred=mse_p, mse_copy=mse_0, mae_pred=mae_p, mae_persist=mae_0)
    # (c) collision foresight, pooled over several hand-off points so every horizon has positives
    scores, labels = [[] for _ in range(8)], [[] for _ in range(8)]
    for c in (8, 12, 16, 20):
        st_c, fim_c = imagine_from(c)
        p_now, p_steps = model.p_col(model.rssm.feat(st_c)), model.p_col(fim_c)
        safe = torch.cumprod(1.0 - p_steps, 1)
        t0 = ep["t_abs"][:, c - 1]
        for j in range(1, 9):
            scores[j - 1].append(1.0 - (1.0 - p_now[:, 0]) * safe[:, j - 1, 0])
            labels[j - 1].append((sd.future_min(ep["r"], t0, j * STRIDE) < DANGER_R).float())
    res["auc"] = [roc_auc(torch.cat(scores[j]).cpu().numpy(), torch.cat(labels[j]).cpu().numpy()) for j in range(8)]
    res["auc_pos_rate"] = [float(torch.cat(labels[j]).mean()) for j in range(8)]
    # (d) gain at +32 (8 imagined steps) vs context length
    gains = []
    for c in CONTEXTS:
        _, f = imagine_from(c)
        xh = model.dec(f[:, 7]).float().clamp(0, 1)
        xt = x[:, c + 7]
        gains.append(float(1.0 - ((xh - xt) ** 2).mean() / ((x[:, c - 1] - xt) ** 2).mean()))
    res["gain32_vs_context"] = gains
    # strip: the sequence whose closest approach after the context is nearest 0.5 m
    fut = d[:, CONTEXT:].min(1).values
    n = int(torch.argmin((fut - 0.5).abs()))
    _, f = imagine_from(CONTEXT)
    xh = model.dec(f[n]).float().clamp(0, 1)
    dh = model.twohot.mean(model.dist_head(f[n]))
    pc = model.p_col(f[n])[:, 0]
    strip = dict(roll=int(ep["r"][n]), ctx=x[n, CONTEXT - 1].permute(1, 2, 0).cpu().numpy(), d_ctx=float(d[n, CONTEXT - 1]),
                 true=[x[n, CONTEXT - 1 + s].permute(1, 2, 0).cpu().numpy() for s in SHOW_STEPS], d_true=[float(d[n, CONTEXT - 1 + s]) for s in SHOW_STEPS],
                 imag=[xh[s - 1].permute(1, 2, 0).cpu().numpy() for s in SHOW_STEPS], d_imag=[float(dh[s - 1]) for s in SHOW_STEPS], p_imag=[float(pc[s - 1]) for s in SHOW_STEPS])
    return res, strip


def jepa_refs():
    """AUC@4/8/16/32 of the one-frame models: A on held-out pybullet (train log), B/C on unseen rooms (offline grid)."""
    refs = {}
    s = open(os.path.join(EV, "train_three.log"), encoding="utf-8").read()
    m = re.search(r"=== TRAIN pybullet_ref.*?AUC@4/8/16/32=([\d./]+)", s, re.S)
    if m:
        refs["pybullet"] = [float(v) for v in m.group(1).split("/")]
    j = json.load(open(os.path.join(REP, "three_way.json")))
    refs["rooms_unseen"] = j["pb_plus_rooms"]["rooms_unseen"]["auc"]
    return refs


def figure(results, strips, refs):
    matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
    ncol = len(SHOW_STEPS) + 1
    fig = plt.figure(figsize=(16, 13.2), facecolor=SURFACE)
    gs = fig.add_gridspec(4, ncol, hspace=0.32, wspace=0.08, left=0.04, right=0.985, top=0.93, bottom=0.40)
    gs2 = fig.add_gridspec(1, 4, wspace=0.34, left=0.05, right=0.985, top=0.31, bottom=0.07)
    fig.suptitle("nano-Dreamer imagination quality: observe 8 steps (667 ms), then imagine from memory and commands only, no further images", fontsize=13, fontweight="bold", color=INK, y=0.975)
    for ci, (key, label, _, _, _, color) in enumerate(CASES):
        st = strips[key]
        for row, (imgs, ds, name) in enumerate(((st["true"], st["d_true"], "true"), (st["imag"], st["d_imag"], "imagined"))):
            r = 2 * ci + row
            ax = fig.add_subplot(gs[r, 0])
            ax.imshow(st["ctx"] if row == 0 else st["ctx"], interpolation="nearest")
            ax.set_title(f"last observed  d={st['d_ctx']:.2f}" if row == 0 else "(same frame)", fontsize=7.5, color=INK2 if row == 0 else MUTED)
            ax.set_ylabel(f"{label.split(' (')[0]}\n{name}", fontsize=8.5, color=color if row == 1 else INK2, fontweight="bold" if row == 1 else "normal")
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(INK2), sp.set_linewidth(0.6)
            for k, (img, dv) in enumerate(zip(imgs, ds)):
                ax = fig.add_subplot(gs[r, k + 1])
                ax.imshow(img, interpolation="nearest")
                if row == 0:
                    ax.set_title(f"+{SHOW_STEPS[k] * STRIDE} frames  d={dv:.2f}", fontsize=7.5, color=INK2)
                else:
                    ax.set_title(f"$\\hat d$={dv:.2f}  P<0.7={st['p_imag'][k]:.2f}", fontsize=7.5, color=color)
                ax.set_xticks([]), ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_edgecolor(color if row == 1 else INK2), sp.set_linewidth(1.2 if row == 1 else 0.6)
    # ---- bottom panels
    axes = [fig.add_subplot(gs2[0, i]) for i in range(4)]
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#e8e7e3", linewidth=0.7)
        ax.tick_params(labelsize=8, colors=INK2)
    for key, label, _, _, _, color in CASES + [c for c in EXTRA if c[0] in results]:
        r = results[key]
        h = np.arange(1, len(r["mse_pred"]) + 1) * STRIDE
        axes[0].plot(h, r["mse_pred"], color=color, linewidth=2, label=f"{label.split(' (')[0]}: imagined, decoded")
        axes[0].plot(h, r["mse_copy"], color=color, linewidth=1.4, linestyle="--", alpha=0.8, label="   baseline: copy last observed frame")
        axes[1].plot(h, r["mae_pred"], color=color, linewidth=2, label=f"{label.split(' (')[0]}: distance head")
        axes[1].plot(h, r["mae_persist"], color=color, linewidth=1.4, linestyle="--", alpha=0.8, label="   baseline: clearance unchanged")
        hk = np.arange(1, 9) * STRIDE
        axes[2].plot(hk, r["auc"], color=color, linewidth=2, marker="o", markersize=4, label=f"{label.split(' (')[0]}: Dreamer open-loop")
        if key in refs and key in dict((c[0], c) for c in CASES):
            axes[2].plot(list(HORIZONS), refs[key], color=color, linewidth=0, marker="s", markersize=7, markerfacecolor="none", markeredgewidth=1.6, label="   old one-frame JEPA (direct head)")
        axes[3].plot(CONTEXTS, r["gain32_vs_context"], color=color, linewidth=2, marker="o", markersize=4, label=label.split(" (")[0])
    axes[0].set_title("(a) decoded-image error (lower = better)", fontsize=9.5, color=INK, loc="left")
    axes[0].set_xlabel("imagined frames ahead (48 Hz)", fontsize=8.5, color=INK2), axes[0].set_ylabel("per-pixel MSE", fontsize=8.5, color=INK2)
    axes[1].set_title("(b) clearance error of the distance head", fontsize=9.5, color=INK, loc="left")
    axes[1].set_xlabel("imagined frames ahead", fontsize=8.5, color=INK2), axes[1].set_ylabel("MAE (m)", fontsize=8.5, color=INK2)
    axes[2].set_title("(c) collision foresight AUC (< 0.7 m within k)", fontsize=9.5, color=INK, loc="left")
    axes[2].set_xlabel("horizon k (frames)", fontsize=8.5, color=INK2), axes[2].set_ylabel("ROC-AUC", fontsize=8.5, color=INK2)
    axes[2].axhline(0.5, color=MUTED, linewidth=0.8, linestyle=":"), axes[2].set_ylim(0.45, 1.02)
    axes[3].set_title("(d) value of memory: gain @+32 frames", fontsize=9.5, color=INK, loc="left")
    axes[3].set_xlabel("steps observed before imagining (12 Hz)", fontsize=8.5, color=INK2), axes[3].set_ylabel("gain = 1 − MSE(imagined) / MSE(copy)", fontsize=8.5, color=INK2)
    axes[3].axhline(0, color=MUTED, linewidth=0.8, linestyle=":"), axes[3].set_xticks(CONTEXTS)
    for ax, loc in zip(axes, ("upper left", "upper left", "lower right", "lower right")):
        ax.legend(fontsize=6.8, frameon=False, loc=loc)
    n = ", ".join(f"{label.split(' (')[0]}: {results[key]['n_seq']} sequences" for key, label, *_ in CASES)
    if any(c[0] in results for c in EXTRA):
        n += "; green = indoor model trained for imagination: 1280 rollouts (5×) + 8-step overshooting, same 1.26M parameters"
    fig.text(0.04, 0.022, f"Curves average every held-out / unseen sequence ({n}; 4 phases per rollout).", fontsize=7.8, color=MUTED)
    fig.text(0.04, 0.006, "Strips: one sequence each whose closest approach is ≈ 0.5 m. Imagination sees no new image; the commands are the ones actually flown.", fontsize=7.8, color=MUTED)
    path = os.path.join(REP, "imagination_eval.png")
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return path


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results, strips = {}, {}
    for key, label, ckpt, dataset, heldout, _ in CASES + [c for c in EXTRA if os.path.exists(os.path.join(OUT, c[2]))]:
        results[key], strips[key] = run_case(key, ckpt, dataset, heldout, device)
        r = results[key]
        print(f"[{key}] {r['n_seq']} seqs: AUC@4..32 {np.round(r['auc'][:1] + r['auc'][1:2] + r['auc'][3:4] + r['auc'][7:8], 2).tolist()}, "
              f"MSE +32 {r['mse_pred'][7]:.4f} vs copy {r['mse_copy'][7]:.4f}, +88 {r['mse_pred'][-1]:.4f} vs {r['mse_copy'][-1]:.4f}; "
              f"dist MAE +32 {r['mae_pred'][7]:.2f} vs persist {r['mae_persist'][7]:.2f}; gain32 vs context {np.round(r['gain32_vs_context'], 2).tolist()}; strip rollout {strips[key]['roll']}")
    json.dump(results, open(os.path.join(REP, "imagination_eval.json"), "w"), indent=1)
    print("IMAGINATION OK ->", figure(results, strips, jepa_refs()))


if __name__ == "__main__":
    main()
