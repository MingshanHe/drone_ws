"""
Offline generalization grid: what does each world model predict, and where
==========================================================================
The question that matters for a real flight is not whether a checkpoint survives one
simulator's pillar course, but whether the world model it carries still *predicts* in data it
never saw: does it foresee a collision (the collision heads), does it rank the evasions
(action conditioning), and does its latent actually move the way the next frames move
(future-frame prediction). This script scores every checkpoint on every data domain,
held-out frames only, and marks the cells the checkpoint was trained on.

Per (checkpoint, domain):
  auc@k        ROC-AUC of "within 0.7 m within k steps" from the predicted latent, k = 4/8/16/32,
               on windows where the flown command was held (the executed-future labels)
  now_auc      ROC-AUC of the danger-now head (the reactive signal)
  cf_rank      on frames where the stored counterfactual oracle says one veer is dangerous and
               the other is not (both visible), how often the model ranks the safe veer safer;
               chance = 0.5 (rooms included: their oracle comes from the distance field)
  latent_gain  1 - MSE(z_hat_{t+32}, z_{t+32}) / MSE(z_t, z_{t+32}): how much of the latent's
               motion over 32 steps the predictor explains (0 = no better than "future == now")
  frame_gain   the same in pixels through a small decoder fitted on 80% of the domain's rollouts
               with the encoder frozen, scored on the other 20%: 1 - MSE(decode(z_hat_32), x_32)
               / MSE(x_t, x_32). Negative means the decoded prediction is worse than copying.

Outputs src/output/report/generalization.json / .md, results_generalization.png (heatmaps) and
pred_cross_<checkpoint>.png (predicted vs true frames in every domain).

  python src/eval_offline.py                # all checkpoints x all domains (~15 min on the GPU)
  python src/eval_offline.py --models pybullet_shipped mixed_big
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from decode_latents import Decoder, frames_to_tensor, to_img  # noqa: E402
from gen_wm_dataset import ACTION_NAMES, ACTION_VECS, A_NORM, DANGER_R, HORIZONS, counterfactual_labels, window_valid  # noqa: E402
from train_world_model import _index_samples, load_model, roc_auc  # noqa: E402

OUT_DIR = os.path.join(HERE, "output")
REP = os.path.join(OUT_DIR, "report")
MAX_ROLLOUTS = 64

DOMAINS = [  # key, label, file, scene_type filter (None = all)
    ("pybullet", "pybullet pillars", "wm_dataset.npz", None),
    ("hd_pillars", "hyperdrone pillars, hd look", "wm_dataset_mixed_big_60body.npz", 0),
    ("pb_pillars", "hyperdrone pillars, pb look", "wm_dataset_mixed_big_60body_pbstyle.npz", 0),
    ("rand_pillars", "hyperdrone pillars, random appearance (unseen)", "wm_dataset_hyperdrone_rand60.npz", None),
    ("rooms_train", "hyperdrone rooms, houses 0-15", "wm_dataset_mixed_big_60body.npz", 1),
    ("rooms_unseen", "hyperdrone rooms, houses 16-23 (unseen)", "wm_dataset_rooms_unseen60.npz", None),
]
MODELS = [  # tag, checkpoint, label, domains it was trained on
    ("pybullet_shipped", "world_model.pth", "pybullet 64 (shipped)", {"pybullet"}),
    ("pybullet_ref", "world_model_pybullet_ref.pth", "pybullet 64 (retrained)", {"pybullet"}),
    ("mixed_big", "world_model_mixed_big.pth", "hyperdrone 256, hd look", {"hd_pillars", "rooms_train"}),
    ("mixed_big_pbstyle", "world_model_mixed_big_pbstyle.pth", "hyperdrone 256, pb look", {"pb_pillars", "rooms_train"}),
    ("pb_plus_big", "world_model_pb_plus_big.pth", "pybullet + hyperdrone hd look", {"pybullet", "hd_pillars", "rooms_train"}),
    ("pb_plus_big_pbstyle", "world_model_pb_plus_big_pbstyle.pth", "pybullet + hyperdrone pb look", {"pybullet", "pb_pillars", "rooms_train"}),
    ("rooms", "world_model_rooms.pth", "hyperdrone rooms 256", {"rooms_train"}),
    ("pb_plus_rooms", "world_model_pb_plus_rooms.pth", "pybullet 64 + hyperdrone rooms 256", {"pybullet", "rooms_train"}),
]
PER_ROLLOUT = ("frames", "actions", "act_id", "seg", "dists", "pos", "pillars", "in_path", "speed", "cf_labels", "cf_visible", "scene_type")


def load_domain(file, scene_filter):
    blob = np.load(os.path.join(OUT_DIR, file))
    data = {k: blob[k] for k in blob.files}
    R = data["frames"].shape[0]
    rows = np.arange(R)
    if scene_filter is not None and "scene_type" in data:
        rows = rows[data["scene_type"] == scene_filter]
    rows = rows[:MAX_ROLLOUTS]
    sub = {k: data[k][rows] for k in PER_ROLLOUT if k in data}
    for k, v in data.items():
        if k not in sub and k not in PER_ROLLOUT:
            sub[k] = v
    if "cf_labels" not in sub:  # pybullet data: the pillar oracle
        sub["cf_labels"], sub["cf_visible"] = counterfactual_labels(sub)
    return sub


@torch.no_grad()
def encode_all(enc, frames, device, batch=512):
    out = []
    flat = frames.reshape(-1, *frames.shape[2:])
    for i in range(0, len(flat), batch):
        out.append(enc(frames_to_tensor(flat[i : i + batch], device)))
    return torch.cat(out).reshape(frames.shape[0], frames.shape[1], -1)


@torch.no_grad()
def score(enc, pred, cheads, nhead, data, device):
    F, A = data["frames"], data["actions"]
    R, L = F.shape[:2]
    Z = encode_all(enc, F, device)  # (R, L, D)
    idx, c_h = _index_samples(data)  # held windows + executed-future labels (n, H, rings)
    r, t = idx[:, 0], idx[:, 1]
    a = torch.tensor(A[r, t] / A_NORM, dtype=torch.float32, device=device)
    z = Z[r, t]
    zh = pred(z, a)  # (n, H, D)
    p = torch.sigmoid(cheads(zh))[:, :, 0].cpu().numpy()  # warn ring
    aucs = [roc_auc(p[:, j], c_h[:, j, 0]) for j in range(len(HORIZONS))]
    # latent gain at the longest horizon
    k = HORIZONS[-1]
    z_true = Z[r, t + k]
    mse_pred = float(((zh[:, -1] - z_true) ** 2).mean())
    mse_noop = float(((z - z_true) ** 2).mean())
    latent_gain = 1.0 - mse_pred / max(mse_noop, 1e-9)
    # danger-now
    now = torch.sigmoid(nhead(Z.reshape(R * L, -1))).cpu().numpy()
    now_auc = roc_auc(now, (data["dists"].reshape(-1) < DANGER_R).astype(np.float32))
    # counterfactual veer ranking from the stored oracle
    il, ir = ACTION_NAMES.index("veer_left"), ACTION_NAMES.index("veer_right")
    cf, vis = data["cf_labels"], data["cf_visible"]
    sel = (cf[:, :, il, -1, 0] != cf[:, :, ir, -1, 0]) & (vis[:, :, il] == 1) & (vis[:, :, ir] == 1)
    rr, tt = np.nonzero(sel)
    cf_rank, n_rank = float("nan"), 0
    if len(rr):
        zz = Z[rr, tt]
        sv = torch.tensor(data["speed"][rr], dtype=torch.float32, device=device)[:, None]
        al = torch.tensor(ACTION_VECS[il] / A_NORM, dtype=torch.float32, device=device) * sv
        ar = torch.tensor(ACTION_VECS[ir] / A_NORM, dtype=torch.float32, device=device) * sv
        pl = torch.sigmoid(cheads(pred(zz, al)))[:, -1, 0].cpu().numpy()
        pr = torch.sigmoid(cheads(pred(zz, ar)))[:, -1, 0].cpu().numpy()
        left_safe = cf[rr, tt, il, -1, 0] == 0
        cf_rank, n_rank = float(np.mean(np.where(left_safe, pl < pr, pr < pl))), int(len(rr))
    return {"auc": aucs, "now_auc": now_auc, "latent_gain": latent_gain, "cf_rank": cf_rank, "n_rank": n_rank, "n_windows": int(len(idx))}, Z


def fit_decoder(enc, frames, device, epochs=6, seed=0):
    torch.manual_seed(seed)
    dec = Decoder().to(device)
    opt = torch.optim.Adam(dec.parameters(), lr=2e-3)
    flat = frames.reshape(-1, *frames.shape[2:])
    for ep in range(epochs):
        order = np.random.default_rng(seed + ep).permutation(len(flat))
        for i in range(0, len(flat), 256):
            x = frames_to_tensor(flat[order[i : i + 256]], device)
            with torch.no_grad():
                z = enc(x)
            loss = ((dec(z) - x) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    dec.eval()
    return dec


@torch.no_grad()
def frame_gain(enc, pred, dec, data, rolls, device):
    F, A = data["frames"], data["actions"]
    L, k = F.shape[1], HORIZONS[-1]
    idx = [(r, t) for r in rolls for t in range(0, L - k, 2) if window_valid(data["seg"][r], t, k)]
    mp = mc = mr = 0.0
    for i in range(0, len(idx), 256):
        chunk = idx[i : i + 256]
        x = frames_to_tensor(np.array([F[r, t] for r, t in chunk]), device)
        xk = frames_to_tensor(np.array([F[r, t + k] for r, t in chunk]), device)
        a = torch.tensor(np.array([A[r, t] for r, t in chunk]) / A_NORM, dtype=torch.float32, device=device)
        z = enc(x)
        zh = pred(z, a)[:, -1]
        mp += float(((dec(zh) - xk) ** 2).mean(dim=(1, 2, 3)).sum())
        mc += float(((x - xk) ** 2).mean(dim=(1, 2, 3)).sum())
        mr += float(((dec(z) - x) ** 2).mean(dim=(1, 2, 3)).sum())
    n = max(len(idx), 1)
    return {"frame_gain": 1.0 - (mp / n) / max(mc / n, 1e-9), "pred_mse": mp / n, "copy_mse": mc / n, "recon_mse": mr / n}


def cross_figure(tag, label, enc, pred, decoders, domains, device, rng):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = (8, 32)
    cols = ["frame t"] + [f"pred +{k}" for k in ks] + [f"true +{k}" for k in ks]
    fig, axes = plt.subplots(len(domains), len(cols), figsize=(1.45 * len(cols), 1.5 * len(domains) + 0.5))
    axes = np.atleast_2d(axes)
    with torch.no_grad():
        for i, (key, dlabel, data, dec, test_rolls) in enumerate(domains):
            L = data["frames"].shape[1]
            cands = [(r, t) for r in test_rolls for t in range(0, L - 32, 4) if window_valid(data["seg"][r], t, 32) and data["dists"][r, t + 32] < 1.2]
            if not cands:
                cands = [(r, t) for r in test_rolls for t in range(0, L - 32, 8) if window_valid(data["seg"][r], t, 32)]
            r, t = cands[rng.integers(len(cands))]
            x = frames_to_tensor(data["frames"][r, t][None], device)
            a = torch.tensor(data["actions"][r, t][None] / A_NORM, dtype=torch.float32, device=device)
            zh = pred(enc(x), a)[0]
            imgs = [data["frames"][r, t]] + [to_img(dec(zh[HORIZONS.index(k)][None])[0]) for k in ks] + [data["frames"][r, t + k] for k in ks]
            for j, img in enumerate(imgs):
                axes[i, j].imshow(img)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                if i == 0:
                    axes[i, j].set_title(cols[j], fontsize=8)
            axes[i, 0].set_ylabel(dlabel.replace(", ", ",\n"), fontsize=6.5)
    fig.suptitle(f"{label}: predicted vs true future in every domain (command: the one flown)", fontsize=9)
    fig.tight_layout()
    path = os.path.join(REP, f"pred_cross_{tag}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def heatmaps(results, model_rows, domain_cols):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    ramp = LinearSegmentedColormap.from_list("blue", ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"])
    panels = (
        ("Collision foresight: AUC of 'within 0.7 m in 32 steps'", lambda m: m["auc"][-1], 0.5, 1.0, "{:.2f}"),
        ("Action conditioning: safe-veer ranking accuracy", lambda m: m["cf_rank"], 0.5, 1.0, "{:.2f}"),
        ("Future-frame prediction: decoded gain vs copy", lambda m: m["frame_gain"], -0.5, 0.5, "{:+.2f}"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), facecolor=SURFACE)
    for ax, (title, pick, lo, hi, fmt) in zip(axes, panels):
        grid = np.array([[pick(results[m][d]) for d in domain_cols] for m in model_rows], dtype=float)
        norm = np.clip((grid - lo) / (hi - lo), 0, 1)
        ax.imshow(norm, cmap=ramp, vmin=0, vmax=1, aspect="auto")
        for i, m in enumerate(model_rows):
            for j, d in enumerate(domain_cols):
                v = grid[i, j]
                seen = d in dict((x[0], x[3]) for x in MODELS)[m]
                txt = ("–" if np.isnan(v) else fmt.format(v)) + ("*" if seen else "")
                ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color="white" if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(domain_cols)))
        short = {"pybullet": "pybullet", "hd_pillars": "hd look", "pb_pillars": "pb look",
                 "rand_pillars": "random\nlook†", "rooms_train": "rooms\n0-15", "rooms_unseen": "rooms\n16-23†"}
        ax.set_xticklabels([short.get(d, d) for d in domain_cols], fontsize=7.5, color=INK2)
        ax.set_yticks(range(len(model_rows)))
        ax.set_yticklabels([dict((x[0], x[2]) for x in MODELS)[m] for m in model_rows], fontsize=8, color=INK2)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=8)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.text(0.01, 0.985, "Does the world model still predict outside its training data? rows = checkpoints, columns = data domains (held-out frames; * = domain seen in training)",
             fontsize=11, color=INK, va="top")
    fig.text(0.01, 0.94, "Darker = better. AUC and ranking: 0.5 is chance. Decoded gain: 0 = predicted frame no better than the current one. "
                         "Columns: pybullet pillars; hyperdrone pillars (hd / pb / random look); hyperdrone rooms (houses 0-15 / 16-23). † = unseen by every checkpoint.",
             fontsize=8.2, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.5)
    path = os.path.join(REP, "results_generalization.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", nargs="*", default=[m[0] for m in MODELS])
    ap.add_argument("--epochs", type=int, default=6, help="decoder epochs per (checkpoint, domain)")
    ap.add_argument("--domains", nargs="*", default=[d[0] for d in DOMAINS])
    ap.add_argument("--out", default="generalization", help="basename of the json/md/png outputs")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(REP, exist_ok=True)
    domains = [(k, lbl, load_domain(f, flt)) for k, lbl, f, flt in DOMAINS if k in args.domains and os.path.exists(os.path.join(OUT_DIR, f))]
    print(f"[INFO] {len(domains)} domains: " + ", ".join(f"{lbl} ({d['frames'].shape[0]} rollouts)" for _, lbl, d in domains))
    results = {}
    rng = np.random.default_rng(0)
    for tag, ckpt, label, seen in MODELS:
        if tag not in args.models:
            continue
        enc, pred, cheads, nhead, meta = load_model(os.path.join(OUT_DIR, ckpt), device=device)
        results[tag] = {}
        cross = []
        for key, dlabel, data in domains:
            R = data["frames"].shape[0]
            split = np.random.default_rng(1).permutation(R)
            fit_rolls, test_rolls = split[: int(0.8 * R)], split[int(0.8 * R) :]
            m, _ = score(enc, pred, cheads, nhead, data, device)
            dec = fit_decoder(enc, data["frames"][fit_rolls], device, epochs=args.epochs)
            m.update(frame_gain(enc, pred, dec, data, test_rolls, device))
            m["seen"] = key in seen
            results[tag][key] = m
            cross.append((key, dlabel, data, dec, test_rolls))
            print(f"  {tag:20s} {dlabel:45s} auc@32 {m['auc'][-1]:.2f}  now {m['now_auc']:.2f}  rank {m['cf_rank']:.2f} (n={m['n_rank']})  "
                  f"latent gain {m['latent_gain']:+.2f}  frame gain {m['frame_gain']:+.2f}{'  *seen' if m['seen'] else ''}")
        cross_figure(tag, label, enc, pred, None, cross, device, rng)
    with open(os.path.join(REP, f"{args.out}.json"), "w") as f:
        json.dump(results, f, indent=2)
    if len(results) == len(MODELS):
        heatmaps(results, [m[0] for m in MODELS], [d[0] for d in domains])
    # markdown tables
    lines = []
    dl = {d[0]: d[1] for d in DOMAINS}
    ml = {m[0]: m[2] for m in MODELS}
    for title, pick, fmt in (
        ("碰撞预判 AUC@32（0.5 = 随机）", lambda m: m["auc"][-1], "{:.2f}"),
        ("碰撞预判 AUC@8", lambda m: m["auc"][1], "{:.2f}"),
        ("danger-now AUC", lambda m: m["now_auc"], "{:.2f}"),
        ("安全绕行方向排序准确率（0.5 = 随机）", lambda m: m["cf_rank"], "{:.2f}"),
        ("潜变量预测增益 @32（0 = 不比「未来 = 现在」好）", lambda m: m["latent_gain"], "{:+.2f}"),
        ("解码未来帧增益 @32（0 = 不比复制当前帧好）", lambda m: m["frame_gain"], "{:+.2f}"),
    ):
        lines.append(f"**{title}**\n")
        lines.append("| 训练数据 \\ 数据域 | " + " | ".join(dl[k] for k, _, _ in domains) + " |")
        lines.append("|---|" + "---|" * len(domains))
        for tag in results:
            cells = []
            for k, _, _ in domains:
                m = results[tag][k]
                v = pick(m)
                cells.append(("–" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)) + ("\\*" if m["seen"] else ""))
            lines.append(f"| {ml[tag]} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("\\* = 该数据域参与了这个 checkpoint 的训练（in-sample）；其余单元格是训练时没见过的数据。")
    with open(os.path.join(REP, f"{args.out}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("GENERALIZATION OK ->", os.path.join(REP, f"{args.out}.md"))


if __name__ == "__main__":
    main()
