"""
Nano DreamerV3-style world model for the drone datasets (plan: PLAN_dreamerv3.md, M1-M2)
=======================================================================================
RSSM (GRU deter state + 16x16 categorical latent) trained on the project's rollout datasets at
the 12 Hz decision rate, with DreamerV3's losses (reconstruction, KL balancing dyn/rep with free
bits, symlog two-hot regression) plus the heads this project plans with:

  * dist head  : two-hot regression of the clearance d_t (symlog bins, 0..4 m)
  * col head   : P(d_t < 0.7 m), P(d_t < 0.35 m) at the current / imagined state
  * cont head  : P(not in collision)
  * L_cf       : counterfactual imagination consistency - from a posterior state, hold each of the
                 six commands for 1/2/4/8 model steps (= +4/8/16/32 frames) in the prior and match
                 the noisy-OR "within k" probability to the dataset's counterfactual oracle labels

Validation (held-out rollouts, rollout-level split): posterior now-AUC, OPEN-LOOP AUC@4/8/16/32
after 667 ms of observation and imagination under the flown commands, decoded-frame gain at
+8/+32 (same definition as REPORT section 4.4), imagined-distance MAE and the veer-ranking check.

  python src/train_dreamer.py --data src/output/wm_dataset.npz --out src/output/world_model_dreamer_pybullet.pth
  python src/train_dreamer.py --selftest
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dreamer.data import STRIDE, SeqData  # noqa: E402
from dreamer.nets import MLP, ConvDecoder, ConvEncoder, TwoHot, cat_kl, cat_probs  # noqa: E402
from dreamer.rssm import RSSM  # noqa: E402
from gen_wm_dataset import A_NORM, ACTION_NAMES, ACTION_VECS, COLLISION_R, DANGER_R, HORIZONS, RADII  # noqa: E402
from train_world_model import _augment, _split_rollouts, roc_auc  # noqa: E402

OUT = os.path.join(HERE, "output")
REP = os.path.join(OUT, "report")
EV_DIR = os.path.join(OUT, "evals")
CFG = dict(deter=256, stoch=16, classes=16, hidden=128, depth=16, units=128, unimix=0.01, free=1.0, bins=64, d_max=4.0)
SIZES = {  # PLAN_dreamerv3.md: nano is the report's model; wide is for the visually richer indoor data
    "nano": dict(CFG),
    "wide": dict(CFG, deter=512, stoch=32, classes=32, hidden=256, depth=32, units=256),
    # DreamerV3 configs.yaml size200m: deter 8192, hidden 1024, 32 x 64 latents, depth 64, mults [2,3,4,4], units 1024, blocks 8
    "size200m": dict(CFG, deter=8192, stoch=32, classes=64, hidden=1024, depth=64, units=1024, blocks=8, mults=[2, 3, 4, 4], bins=255),
}
SCALES = dict(rec=1.0, dist=1.0, col=1.0, cont=1.0, dyn=1.0, rep=0.1, cf=1.0)
MODEL_STEPS = [int(k) // STRIDE for k in HORIZONS]  # 1, 2, 4, 8 model steps = +4/8/16/32 frames
H_IMAG = MODEL_STEPS[-1]
N_CF = 4  # counterfactual start steps per training sequence
N_OS = 4  # overshooting start steps per training sequence (--overshoot)
CANDS = torch.tensor(ACTION_VECS / A_NORM, dtype=torch.float32)  # (6, 4) normalised commands


class Dreamer(nn.Module):
    def __init__(self, cfg: dict = CFG, action_dim: int = 4):
        super().__init__()
        self.cfg = cfg = dict(CFG, **cfg)
        self.enc = ConvEncoder(cfg["depth"], cfg.get("mults"))
        self.rssm = RSSM(self.enc.out_dim, action_dim, cfg["deter"], cfg["stoch"], cfg["classes"], cfg["hidden"], cfg["unimix"], cfg["free"], cfg.get("blocks", 1))
        fd = self.rssm.feat_dim
        self.dec = ConvDecoder(fd, cfg["depth"], cfg.get("mults"))
        self.dist_head = MLP(fd, cfg["units"], 2, cfg["bins"])
        self.twohot = TwoHot(0.0, cfg["d_max"], cfg["bins"])
        self.col_head = MLP(fd, cfg["units"], 1, len(RADII))
        self.cont_head = MLP(fd, cfg["units"], 1, 1)

    def encode(self, x: torch.Tensor, chunk: int = 512) -> torch.Tensor:
        """Encoder over a large flat batch in chunks (the size200m encoder would not fit a whole episode set)."""
        return torch.cat([self.enc(x[i : i + chunk]) for i in range(0, len(x), chunk)])

    def decode(self, feat: torch.Tensor, chunk: int = 512) -> torch.Tensor:
        return torch.cat([self.dec(feat[i : i + chunk]) for i in range(0, len(feat), chunk)])

    def p_col(self, feat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.col_head(feat).float())

    @staticmethod
    def within(p_now: torch.Tensor, p_steps: torch.Tensor) -> torch.Tensor:
        """Noisy-OR of 'inside the ring' over the current state and imagined steps 1..j, for every
        horizon in MODEL_STEPS. p_now (N,R), p_steps (N,H,R) -> (N, len(MODEL_STEPS), R)."""
        safe = torch.cumprod(1.0 - p_steps, dim=1)
        return torch.stack([1.0 - (1.0 - p_now) * safe[:, j - 1] for j in MODEL_STEPS], 1)

    def imagine_cands(self, state: dict, cands: torch.Tensor) -> torch.Tensor:
        """From posterior states (N,...) hold every candidate (A,4) for H_IMAG steps -> within (N,A,len(MODEL_STEPS),R)."""
        N, A = state["h"].shape[0], cands.shape[0]
        st = {k: v.repeat_interleave(A, 0) for k, v in state.items()}
        acts = cands.to(state["h"].device).repeat(N, 1)[:, None].expand(-1, H_IMAG, -1)
        im = self.rssm.imagine(st, acts)
        p_steps = self.p_col(self.rssm.feat(im))
        p_now = self.p_col(self.rssm.feat(state)).repeat_interleave(A, 0)
        return self.within(p_now, p_steps).view(N, A, len(MODEL_STEPS), len(RADII))


def overshoot_losses(model: Dreamer, batch: dict, post: dict, K: int, imag_only: bool = False) -> dict:
    """Imagination-focused training (PlaNet-style latent overshooting, not in DreamerV3): from posterior
    states at random steps, roll the prior K steps under the flown commands and ask the imagined states
    to (a) decode to the true future frames, (b) match the posterior reached there (KL, dyn-style),
    (c) read out the true future clearance / rings. Directly optimises what open-loop evaluation measures."""
    x, d = batch["frames"], batch["dists"]
    B, T = x.shape[:2]
    dev = x.device
    bi = torch.arange(B, device=dev).repeat_interleave(N_OS)
    ti = torch.randint(0, T - K, (B * N_OS,), device=dev)
    fut = ti[:, None] + 1 + torch.arange(K, device=dev)[None]  # (N, K) target steps
    bk = bi[:, None].expand_as(fut)
    im = model.rssm.imagine(model.rssm.index(post, bi, ti), batch["actions"][bk, fut - 1])
    fim = model.rssm.feat(im).flatten(0, 1)
    xt = x[bk, fut].flatten(0, 1)
    rec = ((model.dec(fim).float() - xt) ** 2).sum((1, 2, 3)).mean()
    post_p = cat_probs(post["logits"][bk, fut].flatten(0, 1).detach(), model.rssm.unimix)
    prior_p = cat_probs(im["logits"].flatten(0, 1), model.rssm.unimix)
    kl = cat_kl(post_p, prior_p).clamp_min(model.rssm.free).mean()
    if imag_only:
        return dict(os_rec=rec, os_kl=kl)
    dt = d[bk, fut].flatten()
    dist = model.twohot.loss(model.dist_head(fim), dt).mean()
    col = F.binary_cross_entropy_with_logits(model.col_head(fim).float(), torch.stack([(dt < r).float() for r in RADII], -1))
    return dict(os_rec=rec, os_kl=kl, os_dist=dist, os_col=col)


def losses(model: Dreamer, batch: dict, robust: bool, cf_all: bool = False, overshoot: int = 0, imag_only: bool = False) -> tuple:
    x = batch["frames"]
    B, T = x.shape[:2]
    flat_x = x.flatten(0, 1)
    embed = model.enc(_augment(flat_x) if robust else flat_x).view(B, T, -1)
    post, prior = model.rssm.observe(embed, batch["a_prev"])
    feat = model.rssm.feat(post).flatten(0, 1)
    d = batch["dists"].flatten()
    rec = ((model.dec(feat).float() - flat_x) ** 2).sum((1, 2, 3)).mean()
    if imag_only:  # world model only, as DreamerV3 without reward / continue heads
        dyn, rep = model.rssm.kl(post["logits"], prior["logits"])
        out = dict(rec=rec, dyn=dyn, rep=rep)
        if overshoot > 0:
            out.update(overshoot_losses(model, batch, post, overshoot, imag_only=True))
        return out, post
    dist = model.twohot.loss(model.dist_head(feat), d).mean()
    col_lab = torch.stack([(d < r).float() for r in RADII], -1)
    col = F.binary_cross_entropy_with_logits(model.col_head(feat).float(), col_lab)
    cont = F.binary_cross_entropy_with_logits(model.cont_head(feat).float().squeeze(-1), (d > COLLISION_R).float())
    dyn, rep = model.rssm.kl(post["logits"], prior["logits"])
    # counterfactual imagination consistency
    bi = torch.arange(B, device=x.device).repeat_interleave(N_CF)
    ti = torch.randint(0, T, (B * N_CF,), device=x.device)
    w = model.imagine_cands(model.rssm.index(post, bi, ti), CANDS)  # (N,6,4,2)
    lab, vis = batch["cf"][bi, ti], batch["vis"][bi, ti][..., None, None]
    if cf_all:  # memory-aware: a threat outside the current FOV is still answerable from the belief
        vis = torch.ones_like(vis)
    w = w.float().clamp(1e-5, 1 - 1e-5)
    nll = -(lab * torch.log(w) + (1.0 - lab) * torch.log(1.0 - w))  # BCE on probabilities (noisy-OR output, no logits)
    cf = (nll * vis).sum() / (vis.sum() * lab.shape[-1] * lab.shape[-2] + 1e-6)
    out = dict(rec=rec, dist=dist, col=col, cont=cont, dyn=dyn, rep=rep, cf=cf)
    if overshoot > 0:
        out.update(overshoot_losses(model, batch, post, overshoot))
    return out, post


IMAG_KS = (1, 2, 4, 8)  # imagined model steps scored (= +4 / +8 / +16 / +32 frames)


@torch.no_grad()
def evaluate_imagination(model: Dreamer, data: SeqData, rolls, phases=(0, 2)) -> dict:
    """Imagination-only test: posterior reconstruction error, KL between posterior and prior per step
    (how much the image surprises the model), and open-loop decoded error after 8 observed steps vs the
    copy-last-frame baseline at +4/+8/+16/+32 frames (hand-offs at model steps 8/12/16/20 pooled)."""
    was = model.training
    model.eval()
    ep = data.episodes(rolls, phases=phases)
    x = ep["frames"]
    N, T = x.shape[:2]
    post, prior = model.rssm.observe(model.encode(x.flatten(0, 1)).view(N, T, -1), ep["a_prev"])
    feat = model.rssm.feat(post).flatten(0, 1)
    recon = float(((model.decode(feat).float().clamp(0, 1) - x.flatten(0, 1)) ** 2).mean())
    kl = float(cat_kl(cat_probs(post["logits"], model.rssm.unimix), cat_probs(prior["logits"], model.rssm.unimix))[:, 1:].mean())
    mp, mc = {j: [] for j in IMAG_KS}, {j: [] for j in IMAG_KS}
    for t0 in (8, 12, 16, 20):
        if t0 + IMAG_KS[-1] > T - 1:
            continue
        st = model.rssm.index(post, slice(None), t0)
        fim = model.rssm.feat(model.rssm.imagine(st, ep["actions"][:, t0 : t0 + IMAG_KS[-1]]))
        for j in IMAG_KS:
            xt = x[:, t0 + j]
            mp[j].append(((model.decode(fim[:, j - 1]).float().clamp(0, 1) - xt) ** 2).mean((1, 2, 3)))
            mc[j].append(((x[:, t0] - xt) ** 2).mean((1, 2, 3)))
    mse = {STRIDE * j: float(torch.cat(mp[j]).mean()) for j in IMAG_KS}
    copy = {STRIDE * j: float(torch.cat(mc[j]).mean()) for j in IMAG_KS}
    model.train(was)
    return dict(n_seq=int(N), recon=recon, kl=kl, mse=mse, copy=copy, gain={k: 1.0 - mse[k] / copy[k] for k in mse})


def fmt_imag(m: dict) -> str:
    return (f"recon {m['recon']:.4f}  KL {m['kl']:.2f}  imagined MSE@4/8/16/32 " + "/".join(f"{m['mse'][k]:.4f}" for k in m["mse"])
            + "  gain@4/8/16/32 " + "/".join(f"{m['gain'][k]:+.2f}" for k in m["gain"]))


@torch.no_grad()
def evaluate(model: Dreamer, data: SeqData, rolls, fig_path: str | None = None, title: str = "") -> dict:
    model.eval()
    ep = data.episodes(rolls)
    x = ep["frames"]
    N, T = x.shape[:2]
    embed = model.encode(x.flatten(0, 1)).view(N, T, -1)
    post, _ = model.rssm.observe(embed, ep["a_prev"])
    feat = model.rssm.feat(post)
    d = ep["dists"]
    p_post = model.p_col(feat.flatten(0, 1))
    now_auc = roc_auc(p_post[:, 0].cpu().numpy(), (d.flatten() < DANGER_R).float().cpu().numpy())
    dist_mae = float((model.twohot.mean(model.dist_head(feat.flatten(0, 1))) - d.flatten()).abs().mean())
    scores, labels = {k: [] for k in HORIZONS}, {k: [] for k in HORIZONS}
    mse_p, mse_0, mae_im = {2: [], 8: []}, {2: [], 8: []}, []
    hand_offs = [t0 for t0 in (8, 12, 16, 20) if t0 + H_IMAG <= T - 1]
    for t0 in hand_offs:
        st = model.rssm.index(post, slice(None), t0)
        im = model.rssm.imagine(st, ep["actions"][:, t0 : t0 + H_IMAG])  # the flown commands, open loop
        fim = model.rssm.feat(im)
        w = model.within(model.p_col(model.rssm.feat(st)), model.p_col(fim))  # (N,4,2)
        for j, k in enumerate(HORIZONS):
            scores[k].append(w[:, j, 0])
            labels[k].append((data.future_min(ep["r"], ep["t_abs"][:, t0], int(k)) < DANGER_R).float())
        for j in mse_p:
            xt = x[:, t0 + j]
            mse_p[j].append(((model.dec(fim[:, j - 1]).float() - xt) ** 2).mean((1, 2, 3)))
            mse_0[j].append(((x[:, t0] - xt) ** 2).mean((1, 2, 3)))
        mae_im.append((model.twohot.mean(model.dist_head(fim[:, -1])) - d[:, t0 + H_IMAG]).abs())
    auc = {int(k): roc_auc(torch.cat(scores[k]).cpu().numpy(), torch.cat(labels[k]).cpu().numpy()) for k in HORIZONS}
    gain = {4 * j: float(1.0 - torch.cat(mse_p[j]).mean() / torch.cat(mse_0[j]).mean()) for j in mse_p}
    mae_im = float(torch.cat(mae_im).mean())
    # veer ranking: where the oracle says one veer is safer (warn ring, +32) and both are answerable
    vl, vr = ACTION_NAMES.index("veer_left"), ACTION_NAMES.index("veer_right")
    cf, vis = ep["cf"], ep["vis"]
    mask = (vis[..., vl] > 0) & (vis[..., vr] > 0) & (cf[..., vl, -1, 0] != cf[..., vr, -1, 0])
    mask[:, :4] = False  # give the state a few frames of context
    n_idx, t_idx = mask.nonzero(as_tuple=True)
    if len(n_idx):
        w = model.imagine_cands(model.rssm.index(post, n_idx, t_idx), CANDS[[vl, vr]])[:, :, -1, 0]  # (M,2) within 32, warn
        lab_left = cf[n_idx, t_idx, vl, -1, 0] < cf[n_idx, t_idx, vr, -1, 0]
        veer, n_veer = float(((w[:, 0] < w[:, 1]) == lab_left).float().mean()), int(len(n_idx))
    else:
        veer, n_veer = float("nan"), 0
    if fig_path:
        openloop_figure(model, ep, post, data, fig_path, title)
    model.train()
    return dict(now_auc=float(now_auc), dist_mae=dist_mae, auc=auc, gain=gain, mae_im=mae_im, veer=veer, n_veer=n_veer)


@torch.no_grad()
def openloop_figure(model: Dreamer, ep: dict, post: dict, data: SeqData, path: str, title: str, t0: int = 8, n_show: int = 3) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, d = ep["frames"], ep["dists"]
    N = x.shape[0]
    fut = d[:, t0 : t0 + H_IMAG + 1].min(1).values
    picks, seen = [], set()
    for n in torch.argsort(fut).tolist():  # the sequences that actually approach something first, distinct rollouts
        r = int(ep["r"][n])
        if r not in seen:
            picks.append(n), seen.add(r)
        if len(picks) == n_show:
            break
    cols = H_IMAG + 1
    fig, axes = plt.subplots(2 * len(picks), cols, figsize=(1.45 * cols, 3.1 * len(picks)), facecolor="#fcfcfb")
    axes = np.atleast_2d(axes)
    for row, n in enumerate(picks):
        st = model.rssm.index(post, slice(n, n + 1), t0)
        im = model.rssm.imagine(st, ep["actions"][n : n + 1, t0 : t0 + H_IMAG])
        fim = model.rssm.feat(im)[0]
        xh = torch.cat([model.dec(model.rssm.feat(st)), model.dec(fim)]).float().clamp(0, 1)  # (1+H,3,64,64)
        dh = model.twohot.mean(model.dist_head(torch.cat([model.rssm.feat(st), fim])))
        p = torch.cat([model.p_col(model.rssm.feat(st)), model.p_col(fim)])[:, 0]
        for j in range(cols):
            a_true, a_pred = axes[2 * row, j], axes[2 * row + 1, j]
            a_true.imshow(x[n, t0 + j].permute(1, 2, 0).cpu().numpy(), interpolation="nearest")
            a_pred.imshow(xh[j].permute(1, 2, 0).cpu().numpy(), interpolation="nearest")
            a_true.set_title(("t" if j == 0 else f"+{STRIDE * j}") + f"  d={float(d[n, t0 + j]):.2f}", fontsize=7.5, color="#52514e")
            a_pred.set_title(f"d̂={float(dh[j]):.2f}  P<0.7={float(p[j]):.2f}", fontsize=7, color="#2a78d6")
            for a in (a_true, a_pred):
                a.set_xticks([]), a.set_yticks([])
        axes[2 * row, 0].set_ylabel("true", fontsize=8)
        axes[2 * row + 1, 0].set_ylabel("posterior | imagined", fontsize=8)
    fig.suptitle(title + f"  -  open loop: {t0 * STRIDE} frames observed, then {H_IMAG} imagined steps under the flown commands", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)


def fmt(m: dict) -> str:
    auc = "/".join(f"{m['auc'][int(k)]:.2f}" for k in HORIZONS)
    return (f"now-AUC={m['now_auc']:.2f}, open-loop AUC@4/8/16/32={auc}, gain@8/@32={m['gain'][8]:.2f}/{m['gain'][32]:.2f}, "
            f"dist-MAE post/imag@32={m['dist_mae']:.2f}/{m['mae_im']:.2f} m, veer-ranking={m['veer']:.2f} (n={m['n_veer']})")


def train(data_dict: dict, args, device: str) -> tuple:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tr_rolls, va_rolls = _split_rollouts(data_dict, rng)
    data = SeqData(data_dict, device)
    cfg = SIZES[getattr(args, "size", "nano")]
    model = Dreamer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] {data.R} rollouts x {data.T} model steps ({STRIDE} frames/step); train/val rollouts {len(tr_rolls)}/{len(va_rolls)}; "
          f"params {n_params/1e6:.2f}M ({n_params/1024:.0f} KB int8); device {device}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup)))
    use_amp = device == "cuda"
    tr_rolls_np = np.array(tr_rolls)
    hist, t0, agg = [], time.perf_counter(), {}
    best = [-1e9, 0, None]
    imag_only = getattr(args, "imag_only", False)
    # imagination test sets: 64 fixed training rollouts, the held-out rollouts, optionally unseen houses
    tr_probe = sorted(np.random.default_rng(1).choice(tr_rolls, min(64, len(tr_rolls)), replace=False).tolist())
    unseen = None
    if getattr(args, "unseen", None):
        ub = np.load(args.unseen)
        unseen = SeqData({k: ub[k] for k in ub.files}, device)
        unseen_rolls = list(range(min(64, unseen.R)))
    imag_hist = []

    def imag_test(step):
        row = {"step": step, "train": evaluate_imagination(model, data, tr_probe), "held_out": evaluate_imagination(model, data, va_rolls)}
        if unseen is not None:
            row["unseen"] = evaluate_imagination(model, unseen, unseen_rolls)
        imag_hist.append(row)
        for k in ("train", "held_out", "unseen"):
            if k in row:
                print(f"[IMAG {step:6d}] {k:8s} " + fmt_imag(row[k]), flush=True)
        return row
    model.train()
    for step in range(1, args.steps + 1):
        batch = data.sample(tr_rolls_np, args.batch, args.length, rng)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            parts, _ = losses(model, batch, args.robust, getattr(args, "cf_all", False), getattr(args, "overshoot", 0), getattr(args, "imag_only", False))
        os_scale = getattr(args, "overshoot_scale", 1.0)
        loss = sum((SCALES[k] if k in SCALES else os_scale) * v for k, v in parts.items())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        sched.step()
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + float(v.detach())
        agg["gn"] = agg.get("gn", 0.0) + float(gn)
        if step % args.log_every == 0 or step == args.steps:
            n = args.log_every if step % args.log_every == 0 else step % args.log_every
            print(f"[{step:6d}] " + " ".join(f"{k}={v / n:.3f}" for k, v in agg.items()) + f"  ({(time.perf_counter() - t0) / step * 1e3:.0f} ms/step)", flush=True)
            agg = {}
        if getattr(args, "imag_eval_every", 0) and (step % args.imag_eval_every == 0 or step == 1):
            row = imag_test(step)
            if imag_only and getattr(args, "keep_best", False) and row["held_out"]["gain"][32] > best[0]:
                best[:] = [row["held_out"]["gain"][32], step, {k: v.detach().clone() for k, v in model.state_dict().items()}]
        if not imag_only and step % args.eval_every == 0 and step < args.steps:
            m = evaluate(model, data, va_rolls)
            hist.append((step, m))
            print(f"[EVAL {step:6d}] " + fmt(m), flush=True)
            sc = m["auc"][32] + m["gain"][32]
            if getattr(args, "keep_best", False) and sc > best[0]:  # held-out early stopping
                best[:] = [sc, step, {k: v.detach().clone() for k, v in model.state_dict().items()}]
                print(f"[BEST {step:6d}] AUC@32 + gain@32 = {sc:.3f}", flush=True)
            if getattr(args, "out", None) and not getattr(args, "keep_best", False):  # keep a usable checkpoint while a long run is still going
                torch.save({"model": model.state_dict(), "meta": {"version": 3, "family": "dreamer", "cfg": dict(cfg), "partial_step": step, "stride": STRIDE,
                                                                  "horizons": [int(k) for k in HORIZONS], "model_steps": MODEL_STEPS, "radii": [float(r) for r in RADII],
                                                                  "danger_r": float(DANGER_R), "a_norm": [float(v) for v in A_NORM], "action_names": list(ACTION_NAMES),
                                                                  "action_vecs": [[float(v) for v in row] for row in ACTION_VECS], "fov_deg": float(data_dict.get("fov_deg", 60.0)),
                                                                  "source": str(data_dict.get("source", "pybullet")), "n_params": int(n_params)}}, args.out + ".partial")
    fig = os.path.join(REP, f"dreamer_openloop_{args.tag}.png") if args.tag else None
    if imag_only:  # no heads trained: report imagination only, keep the best held-out gain@32
        last = imag_hist[-1] if imag_hist and imag_hist[-1]["step"] == args.steps else imag_test(args.steps)
        kept = args.steps
        if getattr(args, "keep_best", False) and best[2] is not None and best[0] > last["held_out"]["gain"][32]:
            model.load_state_dict(best[2])
            kept = best[1]
            print(f"[INFO] keep-best: restoring step {kept} (held-out gain@32 {best[0]:+.3f})", flush=True)
        json.dump({"tag": args.tag, "size": getattr(args, "size", "nano"), "batch": args.batch, "overshoot": int(getattr(args, "overshoot", 0)),
                   "kept_step": kept, "history": imag_hist}, open(os.path.join(EV_DIR, f"dreamer_{args.tag}_imag.json"), "w"), indent=1)
        m = evaluate_imagination(model, data, va_rolls)
        ckpt = {"model": model.state_dict(), "meta": {
            "version": 3, "family": "dreamer", "imag_only": True, "cfg": dict(cfg), "size": getattr(args, "size", "nano"), "stride": STRIDE,
            "horizons": [int(k) for k in HORIZONS], "model_steps": MODEL_STEPS, "radii": [float(r) for r in RADII], "danger_r": float(DANGER_R),
            "a_norm": [float(v) for v in A_NORM], "action_names": list(ACTION_NAMES), "action_vecs": [[float(v) for v in row] for row in ACTION_VECS],
            "fov_deg": float(data_dict.get("fov_deg", 60.0)), "source": str(data_dict.get("source", "pybullet")), "n_params": int(n_params),
            "steps": int(args.steps), "kept_step": int(kept), "overshoot": int(getattr(args, "overshoot", 0))}}
        return ckpt, m, None
    m = evaluate(model, data, va_rolls)
    hist.append((args.steps, m))
    kept = args.steps
    if getattr(args, "keep_best", False) and best[2] is not None and best[0] > m["auc"][32] + m["gain"][32]:
        model.load_state_dict(best[2])
        kept = best[1]
        print(f"[INFO] keep-best: restoring step {kept} (AUC@32 + gain@32 {best[0]:.3f})", flush=True)
    m = evaluate(model, data, va_rolls, fig_path=fig, title=f"Dreamer {getattr(args, 'size', 'nano')} [{args.tag}]")
    ckpt = {
        "model": model.state_dict(),
        "meta": {
            "version": 3, "family": "dreamer", "cfg": dict(cfg), "size": getattr(args, "size", "nano"), "scales": dict(SCALES), "stride": STRIDE,
            "horizons": [int(k) for k in HORIZONS], "model_steps": MODEL_STEPS, "radii": [float(r) for r in RADII],
            "danger_r": float(DANGER_R), "a_norm": [float(v) for v in A_NORM], "action_names": list(ACTION_NAMES),
            "action_vecs": [[float(v) for v in row] for row in ACTION_VECS],
            "fov_deg": float(data_dict.get("fov_deg", 60.0)), "source": str(data_dict.get("source", "pybullet")),
            "n_params": int(n_params), "steps": int(args.steps), "kept_step": int(kept), "robust": bool(args.robust), "cf_all": bool(getattr(args, "cf_all", False)),
            "overshoot": int(getattr(args, "overshoot", 0)), "overshoot_scale": float(getattr(args, "overshoot_scale", 1.0)), "history": hist,
        },
    }
    return ckpt, m, fig


def load_dreamer(path: str, device: str = "cpu") -> tuple:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    if meta.get("family") != "dreamer":
        raise SystemExit(f"{path} is not a dreamer checkpoint")
    model = Dreamer(meta["cfg"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, meta


def _selftest(device: str) -> None:
    rng = np.random.default_rng(0)
    R, L = 6, 120
    seg = np.repeat(np.arange(4), L // 4)[None].repeat(R, 0).astype(np.int16)
    seg[::2] = 0  # passive rollouts for the stratified split
    act_id = rng.integers(0, 6, (R, 4))[:, seg[0]]
    data = dict(frames=rng.integers(0, 255, (R, L, 64, 64, 3), dtype=np.uint8), actions=ACTION_VECS[act_id].astype(np.float32),
                act_id=act_id.astype(np.int16), seg=seg, dists=rng.uniform(0.1, 3.0, (R, L)).astype(np.float32),
                pos=np.zeros((R, L, 3), np.float32), pillars=np.full((R, 3, 2), np.nan, np.float32),
                in_path=np.array([True, False] * 3), speed=np.ones(R, np.float32))

    class A:
        steps, batch, length, lr, warmup, clip, log_every, eval_every, seed, robust, tag = 6, 4, 12, 3e-4, 2, 100.0, 3, 100, 0, True, ""
        overshoot, overshoot_scale = 4, 1.0

    ckpt, m, _ = train(data, A, device)
    assert all(math.isfinite(v) for v in (m["now_auc"], m["dist_mae"], m["mae_im"])), m
    path = os.path.join(OUT, "dreamer_selftest.pth")
    torch.save(ckpt, path)
    model, meta = load_dreamer(path, device)
    x = torch.rand(2, 3, 64, 64, device=device)
    st, _ = model.rssm.obs_step(model.rssm.initial(2), torch.zeros(2, 4, device=device), model.enc(x))
    w = model.imagine_cands(st, CANDS)
    assert w.shape == (2, 6, 4, 2) and bool(((w >= 0) & (w <= 1)).all()), w.shape
    os.remove(path)
    print("SELFTEST OK:", fmt(m))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", default=os.path.join(OUT, "wm_dataset.npz"))
    ap.add_argument("--out", default=os.path.join(OUT, "world_model_dreamer_pybullet.pth"))
    ap.add_argument("--tag", default="pybullet", help="name for the open-loop figure")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--length", type=int, default=24, help="model steps per training sequence (<= 30)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--clip", type=float, default=100.0)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--robust", action="store_true", help="Lesson-20 appearance augmentation on the encoder input")
    ap.add_argument("--cf-all", action="store_true", help="counterfactual loss on every (frame, command), also when the threat is outside the current FOV (memory-aware)")
    ap.add_argument("--size", choices=list(SIZES), default="nano")
    ap.add_argument("--imag-only", action="store_true", help="world model only: reconstruction + KL (+ overshoot), no collision / distance / continue / counterfactual heads")
    ap.add_argument("--imag-eval-every", type=int, default=0, help="imagination test (train / held-out / unseen) every N steps")
    ap.add_argument("--unseen", default=None, help="dataset of unseen houses for the imagination test")
    ap.add_argument("--keep-best", action="store_true", help="save the weights of the best held-out eval (AUC@32 + gain@32), not the last")
    ap.add_argument("--overshoot", type=int, default=0, help="imagination-focused training: K-step latent/pixel overshooting from random posterior states (0 = off)")
    ap.add_argument("--overshoot-scale", type=float, default=1.0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--eval-only", action="store_true", help="re-evaluate the checkpoint at --out on the held-out rollouts and redraw its figure")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.selftest:
        return _selftest(device)
    blob = np.load(args.data)
    data = {k: blob[k] for k in blob.files}
    print(f"[INFO] dataset {args.data}: source={data.get('source', 'pybullet')}, camera {float(data.get('fov_deg', 60.0)):.0f} deg")
    if args.eval_only:
        model, meta = load_dreamer(args.out, device)
        _, va_rolls = _split_rollouts(data, np.random.default_rng(args.seed))
        fig = os.path.join(REP, f"dreamer_openloop_{args.tag}.png")
        m = evaluate(model, SeqData(data, device), va_rolls, fig_path=fig, title=f"nano-Dreamer [{args.tag}]")
        print(f"DREAMER EVAL [{args.tag}]: " + fmt(m) + f", figure {fig}")
        return
    ckpt, m, fig = train(data, args, device)
    torch.save(ckpt, args.out)
    if ckpt["meta"].get("imag_only"):
        print(f"DREAMER-IMAG OK [{args.tag}]: {args.steps} steps (kept {ckpt['meta']['kept_step']}), held-out " + fmt_imag(m) + f", params={ckpt['meta']['n_params']/1e6:.2f}M, saved {args.out}")
        return
    print(f"DREAMER OK [{args.tag}]: {args.steps} steps, " + fmt(m) + f", params={ckpt['meta']['n_params']/1e6:.2f}M, saved {args.out}" + (f", figure {fig}" if fig else ""))


if __name__ == "__main__":
    main()
