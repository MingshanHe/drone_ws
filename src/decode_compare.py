"""
Decoded future frames: the three models on the same held-out windows
====================================================================
The world model predicts the next LATENT, never pixels. To compare what the three checkpoints
(A pybullet only, B hyperdrone indoor only, C merged) imagine, this script fits one small decoder
per (checkpoint, domain) exactly as eval_offline.py does (64-d latent -> 64x64 RGB, encoder
frozen, 80% of the domain's rollouts), then decodes each model's predicted latent for the command
actually flown at +8 and +32 steps on the SAME windows from the remaining 20%, next to the true
future frames. It also scores every (checkpoint, domain) with the decoded-frame gain
1 - MSE(decode(z_hat_k), x_k) / MSE(x_t, x_k) at k = 8 and 32 (0 = no better than copying frame t).

Caveat (same as decode_latents.py): the predictor was trained against the EMA target encoder,
whose weights are not saved; decoders are fitted on the online encoder, so decoded predictions
carry that mismatch on top of what a 64-d latent can hold. Expect blurry shapes in the right
place, not photographs.

  python src/decode_compare.py --lang both
Outputs src/output/report/decode_three[_en].png and decode_three.json.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from decode_latents import frames_to_tensor, to_img  # noqa: E402
from eval_offline import DOMAINS, MODELS, OUT_DIR, REP, fit_decoder, load_domain  # noqa: E402
from gen_wm_dataset import ACTION_NAMES, A_NORM, HORIZONS, window_valid  # noqa: E402
from train_world_model import load_model  # noqa: E402

THREE = ["pybullet_ref", "rooms", "pb_plus_rooms"]
SHOW = ["pybullet", "rand_pillars", "rooms_unseen"]
KS = (8, 32)
PER_DOMAIN = 2
STR = {
    "zh": dict(font=["WenQuanYi Zen Hei", "DejaVu Sans"], fig="decode_three.png",
               names=["A 纯 pybullet", "B 纯室内", "C 合并"],
               dom={"pybullet": "pybullet 柱子", "rand_pillars": "hyperdrone 柱子\n随机外观（未见）", "rooms_unseen": "hyperdrone 房间\n16–23（未见）"},
               now="当前帧 t", true="真实 +{k}", pred="{m}\n预测 +{k}",
               title="三个模型在同一批 held-out 窗口上解码的未来帧（命令 = 实际飞的；每个模型 × 数据域各自拟合解码器）"),
    "en": dict(font=["DejaVu Sans"], fig="decode_three_en.png",
               names=["A pybullet-only", "B indoor-only", "C merged"],
               dom={"pybullet": "pybullet pillars", "rand_pillars": "hyperdrone pillars\nrandom appearance (unseen)", "rooms_unseen": "hyperdrone rooms\n16–23 (unseen)"},
               now="frame t", true="true +{k}", pred="{m}\npred +{k}",
               title="Decoded future frames of the three models on the same held-out windows (command = the one flown; one decoder per model × domain)"),
}


@torch.no_grad()
def gains(enc, pred, dec, data, rolls, device):
    F, A = data["frames"], data["actions"]
    L = F.shape[1]
    idx = [(r, t) for r in rolls for t in range(0, L - HORIZONS[-1], 2) if window_valid(data["seg"][r], t, HORIZONS[-1])]
    mp, mc = {k: 0.0 for k in KS}, {k: 0.0 for k in KS}
    for i in range(0, len(idx), 256):
        chunk = idx[i : i + 256]
        x = frames_to_tensor(np.array([F[r, t] for r, t in chunk]), device)
        a = torch.tensor(np.array([A[r, t] for r, t in chunk]) / A_NORM, dtype=torch.float32, device=device)
        zh = pred(enc(x), a)
        for k in KS:
            xk = frames_to_tensor(np.array([F[r, t + k] for r, t in chunk]), device)
            mp[k] += float(((dec(zh[:, HORIZONS.index(k)]) - xk) ** 2).mean(dim=(1, 2, 3)).sum())
            mc[k] += float(((x - xk) ** 2).mean(dim=(1, 2, 3)).sum())
    n = max(len(idx), 1)
    return {str(k): dict(gain=1.0 - (mp[k] / n) / max(mc[k] / n, 1e-9), pred_mse=mp[k] / n, copy_mse=mc[k] / n) for k in KS} | {"windows": len(idx)}


def pick_windows(data, rolls, rng):
    """PER_DOMAIN windows from distinct held-out rollouts, each the moment 32 steps before the drone is ~0.6 m from an obstacle."""
    L = data["frames"].shape[1]
    fwd = ACTION_NAMES.index("forward")
    in_path = data.get("in_path")
    cands = {}
    for r in rolls:
        if in_path is not None and in_path[r] == 0 and np.any(in_path[rolls] == 1):
            continue  # prefer rollouts that start with an obstacle ahead
        best = None
        for t in range(0, L - HORIZONS[-1], 2):
            if int(data["act_id"][r, t]) != fwd:
                continue  # the flown command is "forward": the obstacle grows, the clearest case to read
            if window_valid(data["seg"][r], t, HORIZONS[-1]) and data["dists"][r, t + HORIZONS[-1]] < 1.2:
                d = abs(float(data["dists"][r, t + HORIZONS[-1]]) - 0.6)
                if best is None or d < best[0]:
                    best = (d, t)
        if best is not None:
            cands[int(r)] = best[1]
    rs = sorted(cands)
    rs = [rs[i] for i in rng.choice(len(rs), min(PER_DOMAIN, len(rs)), replace=False)]
    return [(r, cands[r]) for r in sorted(rs)]


def figure(lang, rows, preds, names_by_tag):
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = STR[lang]["font"]
    import matplotlib.pyplot as plt

    t = STR[lang]
    ncol = 1 + len(KS) + len(THREE) * len(KS)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(1.5 * ncol, 1.5 * len(rows) + 1.0))
    axes = np.atleast_2d(axes)
    titles = [t["now"]] + [t["true"].format(k=k) for k in KS] + [t["pred"].format(m=names_by_tag[m][lang], k=k) for m in THREE for k in KS]
    for i, (key, r, tt, data) in enumerate(rows):
        imgs = [data["frames"][r, tt]] + [data["frames"][r, tt + k] for k in KS] + [preds[(m, key, r, tt)][k] for m in THREE for k in KS]
        for j, img in enumerate(imgs):
            ax = axes[i, j]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(titles[j], fontsize=7.5)
        axes[i, 0].set_ylabel(f"{t['dom'][key]}\nr{r} t{tt}, {ACTION_NAMES[int(data['act_id'][r, tt])]}", fontsize=6.5)
    fig.suptitle(t["title"], fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(REP, t["fig"])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lang", choices=["zh", "en", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=6)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    domains = {k: load_domain(f, flt) for k, _, f, flt in DOMAINS if k in SHOW}
    splits, rows = {}, []
    for key in SHOW:
        data = domains[key]
        R = data["frames"].shape[0]
        split = np.random.default_rng(1).permutation(R)  # the same 80/20 split as eval_offline.py
        splits[key] = (split[: int(0.8 * R)], split[int(0.8 * R) :])
        rows += [(key, r, t, data) for r, t in pick_windows(data, splits[key][1], rng)]
    print(f"[INFO] device {device}; windows: " + ", ".join(f"{k} r{r} t{t}" for k, r, t, _ in rows))
    seen = {m[0]: m[3] for m in MODELS}
    labels = {m[0]: m[2] for m in MODELS}
    names_by_tag = {m: {"zh": STR["zh"]["names"][i], "en": STR["en"]["names"][i]} for i, m in enumerate(THREE)}
    results, preds = {}, {}
    for tag in THREE:
        ckpt = next(m[1] for m in MODELS if m[0] == tag)
        enc, pred, cheads, nhead, meta = load_model(os.path.join(OUT_DIR, ckpt), device=device)
        results[tag] = {"label": labels[tag]}
        for key in SHOW:
            data = domains[key]
            fit_rolls, test_rolls = splits[key]
            dec = fit_decoder(enc, data["frames"][fit_rolls], device, epochs=args.epochs)
            g = gains(enc, pred, dec, data, test_rolls, device)
            g["seen"] = key in seen[tag]
            results[tag][key] = g
            with torch.no_grad():
                for k2, r, t, d2 in rows:
                    if k2 != key:
                        continue
                    x = frames_to_tensor(d2["frames"][r, t][None], device)
                    a = torch.tensor(d2["actions"][r, t][None] / A_NORM, dtype=torch.float32, device=device)
                    zh = pred(enc(x), a)[0]
                    preds[(tag, key, r, t)] = {k: to_img(dec(zh[HORIZONS.index(k)][None])[0]) for k in KS}
            print(f"  {tag:14s} {key:13s} decoded-frame gain @8 {g['8']['gain']:+.2f}  @32 {g['32']['gain']:+.2f}  (windows {g['windows']}{', seen' if g['seen'] else ''})")
    results["samples"] = [dict(domain=k, rollout=int(r), t=int(t), action=ACTION_NAMES[int(d["act_id"][r, t])]) for k, r, t, d in rows]
    with open(os.path.join(REP, "decode_three.json"), "w") as f:
        json.dump(results, f, indent=2)
    for lang in (["zh", "en"] if args.lang == "both" else [args.lang]):
        print("DECODE-THREE OK ->", figure(lang, rows, preds, names_by_tag))


if __name__ == "__main__":
    main()
