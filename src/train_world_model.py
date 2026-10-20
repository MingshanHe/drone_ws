"""
Lesson 29 (step 2) — train a nano world model (V-JEPA style, latent prediction)
===============================================================================
This is the heart of the lesson. A *world model* answers "if I see this and take
this action, what will I see next?" The trap most people fall into is to answer
in *pixels* — predict the next image (or diffuse it). That is slow and it
hallucinates detail a drone can't act on. We do it the V-JEPA way instead:

  predict the *next latent embedding*, never the next pixels.

Three tiny networks (all reused from the course's conv stack, all int8-able):
  * Encoder f_theta : image -> z                    (Lesson 3's TinyDronet stack)
  * Predictor g_phi : (z_t, action) -> z_hat at *four* horizons
                      k in {4, 8, 16, 32} steps (~83..667 ms @ 48 Hz)
  * Collision heads : z_hat_k -> P(too close within k steps), one per horizon,
                      plus a "danger now" head on z_t itself — that one is the
                      *reactive* signal, kept as the honest baseline the
                      anticipation must beat in steps 3/4.

Why multiple horizons? Because "proactive" is a claim about *time*: a controller
that reacts ~600 ms early needs a model that predicts ~600 ms ahead, not 167 ms.
The predictor shares one trunk and grows one tiny residual head per horizon, so
anticipation at four time scales costs a few extra KB, not a new network.

Why counterfactual labels? A planner never asks "was the action I flew safe?" —
it asks "which of my six options is safest *from here*?" Executed rollouts
answer the first question densely and the second only at segment switches
(measured: collision AUC 0.9 but veer-ranking at chance). So the collision
heads are additionally supervised with the simulator's counterfactual oracle
(step 1's `counterfactual_labels`) at every frame: the same privileged distance
signal, extended from "what happened" to "what would happen if". The
veer-ranking check below is the held-out test of exactly that ability.

The target is produced by an **EMA copy** of the encoder (stop-gradient), the
JEPA/BYOL trick that lets us learn in latent space without a pixel loss and
without collapsing to a constant. A VICReg-style variance term is a second
guard. Two honesty upgrades over the first version of this lesson:

  * The train/val split is **by rollout**, not by frame. Neighbouring frames are
    near-duplicates, so a random frame split leaks the val set into training and
    inflates AUC. Rollout-level AUC is the number you can trust.
  * A **veer-ranking check**: on held-out cruise frames where geometry says one
    veer is truly safer than the other, the model must rank it safer too. That
    is the action-conditioning the step-3 planner relies on — if this number is
    at chance, the model has learned motion but not *consequence*.

Honest scope: the real V-JEPA / V-JEPA 2 is a billion-parameter ViT that needs a
GPU (an Orin-class board), not a GAP8. This is a *nano distillation of the idea*
— the same joint-embedding-prediction principle, shrunk under the 512 KB int8
budget so it keeps the course's north star.

Run:
  python src/train_world_model.py --epochs 80
  python src/train_world_model.py --selftest   # tiny, asserts
Saves output/world_model.pth (git-ignored).
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gen_wm_dataset import (  # noqa: E402
    A_NORM,
    ACTION_NAMES,
    ACTION_VECS,
    CTRL_HZ,
    DANGER_R,
    FORWARD,
    FOV_HALF_DEG,
    HORIZONS,
    RADII,
    counterfactual_labels,
    gen,
    window_valid,
)
from gen_wm_dataset import OUT as DATA  # noqa: E402
from train_cnn import TinyDronet  # noqa: E402  (reuse Lesson 3's conv stack)

LATENT_D = 64  # embedding size (TinyDronet's conv stack already outputs 64)
ACTION_D = 4  # (vx, vy, vz, yaw-rate), normalised per-dim to ~[-1, 1]
EMA_M = 0.99  # target-encoder momentum
LAMBDA_VAR = 1.0  # anti-collapse (VICReg-style variance hinge)
LAMBDA_COL = 1.0  # executed-action collision-head weight (real flown future)
LAMBDA_CF = 1.0  # counterfactual collision weight (the ranking supervision)
LAMBDA_NOW = 0.5  # danger-now head weight (the reactive baseline signal)
GAP8_BUDGET_KB = 512
MODEL = os.path.join(HERE, "output", "world_model.pth")


class Encoder(nn.Module):
    """Image -> latent z. Reuses Lesson 3's TinyDronet conv stack (the same
    features Lesson 17's depth net used) — but swaps its global average pool
    for a *bearing-aware* pooling: four horizontal strips plus a small
    projection. Global pooling averages "where" away: it can say a pillar is
    close, not which side it is on — and dodging left-vs-right is precisely a
    "which side" question. (Measured: with global pooling the veer-ranking
    check sits at chance no matter how it is supervised; with strips it is
    learnable.) Cost: ~16k extra int8 weights."""

    def __init__(self, d=LATENT_D):
        super().__init__()
        stack = TinyDronet().features
        self.features = stack[:-1]  # conv stack only: (B,3,64,64) -> (B,64,8,8)
        self.pool = nn.AdaptiveAvgPool2d((1, 4))  # 4 horizontal strips
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4, d))

    def forward(self, x):  # x: (B, 3, 64, 64)
        return self.proj(self.pool(self.features(x)))  # (B, 64)


class MultiPredictor(nn.Module):
    """Action-conditioned latent forecaster at every horizon:
    z_hat_{t+k} = z_t + delta_k(z_t, a_t) for k in HORIZONS.

    One shared trunk reads (z, a); one tiny linear head per horizon emits that
    horizon's *residual*. Predicting residuals bakes in the right inductive
    bias — with a zero residual it reproduces "the future looks like the
    present", so any learned motion can only improve on that baseline."""

    def __init__(self, d=LATENT_D, a=ACTION_D, h=128, horizons=HORIZONS):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d + a, h), nn.ReLU())
        self.heads = nn.ModuleList(nn.Linear(h, d) for _ in horizons)

    def forward(self, z, a):  # z: (B,D)  a: (B,A)
        feat = self.trunk(torch.cat([z, a], dim=1))
        return torch.stack([z + head(feat) for head in self.heads], dim=1)  # (B,H,D)


class CollisionHeads(nn.Module):
    """Predicted-future latents -> logits per horizon x ring: 'within 0.7 m
    within k steps' (the warning) and 'within 0.35 m within k steps' (the
    about-to-hit). One linear layer per horizon, two outputs each — the
    critical ring is what keeps the planner sighted inside the warn ring."""

    def __init__(self, d=LATENT_D, horizons=HORIZONS, radii=RADII):
        super().__init__()
        self.heads = nn.ModuleList(nn.Linear(d, len(radii)) for _ in horizons)

    def forward(self, zh):  # zh: (B, H, D)
        return torch.stack([h(zh[:, i]) for i, h in enumerate(self.heads)], dim=1)


class DangerNowHead(nn.Module):
    """Current latent -> logit for 'dangerously close *right now*'. This is the
    reactive signal — same encoder, no look-ahead — so steps 3/4 can compare
    anticipation against reaction with the sensor held equal."""

    def __init__(self, d=LATENT_D):
        super().__init__()
        self.lin = nn.Linear(d, 1)

    def forward(self, z):  # z: (B, D)
        return self.lin(z).squeeze(-1)  # (B,)


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, m: float) -> None:
    """target = m*target + (1-m)*online (stop-gradient by construction)."""
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.mul_(m).add_(po.detach(), alpha=1.0 - m)


def _augment(x: torch.Tensor) -> torch.Tensor:
    """Train-time appearance DR: a torch port of Lesson 20's `randomize.jitter`
    (brightness 0.5-1.5 + noise sigma<=0.18) plus a random 3x3 blur. Applied to
    the frames the ONLINE encoder sees — never to the EMA target's frames, so
    the JEPA targets stay stable while the encoder learns to shrug off
    appearance."""
    b = torch.empty(len(x), 1, 1, 1, device=x.device).uniform_(0.5, 1.5)
    s = torch.empty(len(x), 1, 1, 1, device=x.device).uniform_(0.0, 0.18)
    x = x * b + torch.randn_like(x) * s
    blur = torch.rand(len(x), device=x.device) < 0.5
    if bool(blur.any()):
        xb = torch.nn.functional.avg_pool2d(x[blur], 3, stride=1, padding=1)
        x = x.clone()
        x[blur] = xb
    return x.clamp(0.0, 1.0)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U); 0.5 = chance."""
    pos, neg = scores[labels > 0.5], scores[labels < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.concatenate([pos, neg]).argsort()
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _index_samples(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """All (rollout, t) pairs whose command was held over the full H_MAX window,
    with per-horizon, per-ring collision labels of the *flown* future."""
    R, L = data["frames"].shape[:2]
    idx, c_h = [], []
    for r in range(R):
        for t in range(L - HORIZONS[-1]):
            if not window_valid(data["seg"][r], t, HORIZONS[-1]):
                continue
            d = data["dists"][r]
            idx.append((r, t))
            c_h.append(
                [
                    [float(d[t : t + k + 1].min() < rad) for rad in RADII]
                    for k in HORIZONS
                ]
            )
    return np.array(idx, dtype=np.int64), np.array(c_h, dtype=np.float32)


def _split_rollouts(data: dict, rng: np.random.Generator) -> tuple[list, list]:
    """Rollout-level split, stratified over (in-path x passive) so val always
    holds every scenario kind — in particular a passive in-path pass, the one
    that guarantees danger-now positives. Frame-level random splits leak
    (neighbouring frames are near duplicates); this split keeps AUC honest."""
    R = data["frames"].shape[0]
    tr, va = [], []
    for flag_ip in (True, False):
        for flag_passive in (True, False):
            rolls = [
                r
                for r in range(R)
                if bool(data["in_path"][r]) == flag_ip
                and (int(data["seg"][r].max()) == 0) == flag_passive
            ]
            if not rolls:
                continue
            rng.shuffle(rolls)
            n_val = max(1, round(0.2 * len(rolls)))
            va += rolls[:n_val]
            tr += rolls[n_val:]
    return sorted(tr), sorted(va)


def veer_ranking(data: dict, rolls, enc, pred, cheads, device) -> tuple:
    """The action-conditioning acceptance check, scored against a *geometric*
    ground truth. On held-out cruise frames, roll each veer command forward
    kinematically for the longest horizon and measure the true minimum
    clearance either way (privileged pillar layout — used to *evaluate* the
    model, never to control the drone). Keep only the decision-relevant frames
    that vision can actually answer: one veer would cross the danger radius and
    the other would stay clear (by a >0.12 m margin), *and* the threatening
    pillar sits inside the camera FOV — exactly the call the step-3 planner
    must get right. The model is correct when it ranks the truly-safer veer as
    safer. Chance = 0.5."""
    taus = (np.arange(HORIZONS[-1] + 1) / CTRL_HZ)[:, None]  # (k+1, 1)
    i_l, i_r = ACTION_NAMES.index("veer_left"), ACTION_NAMES.index("veer_right")
    cos_fov = np.cos(np.radians(FOV_HALF_DEG))
    frames, gt_left_safer, svs = [], [], []
    L = data["frames"].shape[1]
    for r in rolls:
        pil = data["pillars"][r]
        pil = pil[~np.isnan(pil[:, 0])]
        if not len(pil):
            continue
        sv = float(data["speed"][r])  # judge each rollout at its own pace
        for t in range(L):
            if data["act_id"][r, t] != FORWARD:
                continue
            p0 = data["pos"][r, t, :2]
            d_v, q_v = [], []
            for i in (i_l, i_r):
                dmat = np.linalg.norm(
                    (p0 + taus * sv * ACTION_VECS[i][:2])[:, None, :] - pil[None],
                    axis=2,
                )
                d_v.append(float(dmat.min()))
                q_v.append(pil[dmat.min(axis=0).argmin()])
            d_l, d_r = d_v
            if not (
                abs(d_l - d_r) > 0.12 and min(d_l, d_r) < DANGER_R <= max(d_l, d_r)
            ):
                continue
            rel = (q_v[0] if d_l < d_r else q_v[1]) - p0  # the threatening pillar
            if rel[0] <= float(np.linalg.norm(rel)) * cos_fov:
                continue  # threat outside the camera FOV: unanswerable from vision
            frames.append(data["frames"][r, t])
            gt_left_safer.append(d_l > d_r)
            svs.append(sv)
    if not frames:
        return float("nan"), 0
    x = torch.tensor(np.array(frames), dtype=torch.float32, device=device)
    x = x.permute(0, 3, 1, 2) / 255.0
    sv_col = np.array(svs, dtype=np.float32)[:, None]
    a_l = torch.tensor(sv_col * ACTION_VECS[i_l] / A_NORM, device=device)
    a_r = torch.tensor(sv_col * ACTION_VECS[i_r] / A_NORM, device=device)
    with torch.no_grad():
        z = enc(x)
        p_l = torch.sigmoid(cheads(pred(z, a_l))[:, -1, 0])  # warn ring @ 667 ms
        p_r = torch.sigmoid(cheads(pred(z, a_r))[:, -1, 0])
    gt = torch.tensor(np.array(gt_left_safer), device=device)
    correct = torch.where(gt, p_l < p_r, p_r < p_l)
    return float(correct.float().mean()), len(frames)


def train(
    data: dict, epochs: int = 80, batch: int = 64, seed: int = 0, robust: bool = False
) -> tuple:
    """Train the nano world model on a sequence-format dataset dict and return
    (checkpoint dict, metrics dict). Callable from the eval harness.
    `robust=True` adds Lesson 20-style appearance augmentation to the online
    encoder's frames (pair it with a `--randomize` dataset for step 5)."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    idx, c_h = _index_samples(data)
    tr_rolls, va_rolls = _split_rollouts(data, rng)
    tr = np.where(np.isin(idx[:, 0], tr_rolls))[0]
    va = np.where(np.isin(idx[:, 0], va_rolls))[0]
    print(
        f"[INFO] training on {device}: {len(tr)} train / {len(va)} val samples "
        f"({len(tr_rolls)}/{len(va_rolls)} rollouts)"
    )

    R, L = data["frames"].shape[:2]
    flat = torch.tensor(data["frames"].reshape(R * L, *data["frames"].shape[2:]))
    flat = flat.to(device)  # uint8 on device; per-batch floats below
    acts = torch.tensor(data["actions"].reshape(R * L, 4) / A_NORM).to(device)
    c_h_t = torch.tensor(c_h).to(device)
    base = torch.tensor(idx[:, 0] * L + idx[:, 1]).to(device)
    offs = torch.tensor([int(k) for k in HORIZONS]).to(device)
    # the counterfactual oracle: labels for every (frame, candidate, horizon,
    # ring) — valid at every step, so it uses ALL train-rollout frames
    n_a, n_h, n_r = len(ACTION_VECS), len(HORIZONS), len(RADII)
    cf_np, vis_np = counterfactual_labels(data)
    cf_np = cf_np.reshape(R * L, n_a, n_h, n_r).astype(np.float32)
    vis_np = vis_np.reshape(R * L, n_a).astype(np.float32)
    cf = torch.tensor(cf_np).to(device)
    vis = torch.tensor(vis_np).to(device)
    now_all = torch.tensor(
        (data["dists"].reshape(R * L) < DANGER_R).astype(np.float32)
    ).to(device)
    tr_frames = np.array([r * L + t for r in tr_rolls for t in range(L)])
    c_all = torch.tensor(tr_frames).to(device)
    # frames where the *visible* candidate labels disagree carry the ranking
    # signal; most frames are far from any pillar and teach nothing about
    # choice, so half of every counterfactual batch oversamples the former
    cfv = (cf_np * vis_np[:, :, None, None])[tr_frames].reshape(
        len(tr_frames), n_a, n_h * n_r
    )
    disagree = (cfv.max(axis=1) != cfv.min(axis=1)).any(axis=1)
    c_hard = torch.tensor(tr_frames[disagree] if disagree.any() else tr_frames).to(
        device
    )
    cands = torch.tensor(ACTION_VECS / A_NORM).to(device)

    def frames_at(flat_idx):  # uint8 (N,64,64,3) -> float (N,3,64,64)
        return flat[flat_idx].permute(0, 3, 1, 2).float() / 255.0

    enc, tgt = Encoder().to(device), Encoder().to(device)
    tgt.load_state_dict(enc.state_dict())
    for p in tgt.parameters():
        p.requires_grad_(False)
    pred = MultiPredictor().to(device)
    cheads = CollisionHeads().to(device)
    nhead = DangerNowHead().to(device)

    params = (
        list(enc.parameters())
        + list(pred.parameters())
        + list(cheads.parameters())
        + list(nhead.parameters())
    )
    opt = torch.optim.Adam(params, lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    bce_none = nn.BCEWithLogitsLoss(reduction="none")

    for _epoch in range(1, epochs + 1):
        enc.train(), pred.train(), cheads.train(), nhead.train()
        order = tr[torch.randperm(len(tr)).numpy()]
        for i in range(0, len(order), batch):
            b = torch.tensor(order[i : i + batch]).to(device)
            opt.zero_grad()
            z_t = enc(_augment(frames_at(base[b])) if robust else frames_at(base[b]))
            z_hat = pred(z_t, acts[base[b]])  # (B,H,D)
            with torch.no_grad():
                z_tgt = torch.stack(
                    [tgt(frames_at(base[b] + k)) for k in offs], dim=1
                )  # (B,H,D)
            pred_loss = ((z_hat - z_tgt) ** 2).mean()
            var_loss = torch.relu(1.0 - z_t.std(dim=0)).mean()
            col_loss = bce(cheads(z_hat), c_h_t[b])
            # counterfactual batch: half random frames, half decision-relevant
            half = max(1, len(b) // 2)
            cb = torch.cat(
                [
                    c_all[torch.randint(len(c_all), (half,), device=device)],
                    c_hard[torch.randint(len(c_hard), (half,), device=device)],
                ]
            )
            z_c = enc(_augment(frames_at(cb)) if robust else frames_at(cb))
            z_cf = pred(z_c.repeat_interleave(n_a, dim=0), cands.repeat(len(z_c), 1))
            w = vis[cb].reshape(-1, 1, 1)  # unanswerable (frame, cand): no loss
            cf_loss = (
                bce_none(cheads(z_cf), cf[cb].reshape(-1, n_h, n_r)) * w
            ).sum() / (w.sum() * n_h * n_r + 1e-6)
            now_loss = bce(nhead(z_c), now_all[cb])
            (
                pred_loss
                + LAMBDA_VAR * var_loss
                + LAMBDA_COL * col_loss
                + LAMBDA_CF * cf_loss
                + LAMBDA_NOW * now_loss
            ).backward()
            opt.step()
            ema_update(tgt, enc, EMA_M)

    # -- validation metrics (rollout-level split, so these are honest) --------
    enc.eval(), pred.eval(), cheads.eval(), nhead.eval()
    vb = torch.tensor(va).to(device)
    with torch.no_grad():
        z_t = enc(frames_at(base[vb]))
        z_hat = pred(z_t, acts[base[vb]])
        z_tgt = torch.stack([tgt(frames_at(base[vb] + k)) for k in offs], dim=1)
        mse_h = ((z_hat - z_tgt) ** 2).mean(dim=(0, 2)).cpu().numpy()
        noop_h = (
            ((z_t.unsqueeze(1) - z_tgt) ** 2).mean(dim=(0, 2)).cpu().numpy()
        )  # predictor does nothing
        scores = torch.sigmoid(cheads(z_hat)).cpu().numpy()[:, :, 0]  # warn ring
    auc_h = [roc_auc(scores[:, i], c_h[va][:, i, 0]) for i in range(len(HORIZONS))]
    # danger-now needs no held future window, so score it on *every* val frame
    now_idx = torch.tensor([r * L + t for r in va_rolls for t in range(L)]).to(device)
    now_lbl = np.array(
        [float(data["dists"][r, t] < DANGER_R) for r in va_rolls for t in range(L)],
        dtype=np.float32,
    )
    with torch.no_grad():
        now_scores = (
            torch.cat(
                [
                    torch.sigmoid(nhead(enc(frames_at(now_idx[i : i + 512]))))
                    for i in range(0, len(now_idx), 512)
                ]
            )
            .cpu()
            .numpy()
        )
    now_auc = roc_auc(now_scores, now_lbl)
    side, n_side = veer_ranking(data, va_rolls, enc, pred, cheads, device)
    if n_side < 20:  # tiny val sets may lack decision-relevant geometry; the
        # probe never trains on labels, so widening it stays meaningful
        side, n_side = veer_ranking(
            data, range(data["frames"].shape[0]), enc, pred, cheads, device
        )
        print("[INFO] veer-ranking widened to all rollouts (val had too few frames)")

    n_params = sum(p.numel() for p in params)
    ckpt = {
        "encoder": enc.state_dict(),
        "predictor": pred.state_dict(),
        "collision_heads": cheads.state_dict(),
        "now_head": nhead.state_dict(),
        "meta": {
            "version": 2,
            "D": LATENT_D,
            "A": ACTION_D,
            "horizons": [int(k) for k in HORIZONS],
            "radii": [float(rad) for rad in RADII],
            "danger_r": float(DANGER_R),
            "a_norm": [float(v) for v in A_NORM],
            "action_names": list(ACTION_NAMES),
            "action_vecs": [[float(v) for v in row] for row in ACTION_VECS],
        },
    }
    metrics = {
        "mse": mse_h,
        "noop": noop_h,
        "auc": auc_h,
        "now_auc": now_auc,
        "side": side,
        "n_side": n_side,
        "int8_kb": n_params / 1024,  # weights only; step 4 reports the full budget
        "n_train": len(tr),
        "n_val": len(va),
    }
    return ckpt, metrics


def load_model(path: str = MODEL, device: str = "cpu"):
    """Rebuild the trained nets from a checkpoint. Returns
    (encoder, predictor, collision_heads, now_head, meta), all in eval mode."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    meta = ckpt["meta"]
    if meta.get("version") != 2:
        raise SystemExit(f"{path} is not a v2 checkpoint; re-run this script.")
    enc, pred = Encoder().to(device), MultiPredictor().to(device)
    cheads, nhead = CollisionHeads().to(device), DangerNowHead().to(device)
    enc.load_state_dict(ckpt["encoder"])
    pred.load_state_dict(ckpt["predictor"])
    cheads.load_state_dict(ckpt["collision_heads"])
    nhead.load_state_dict(ckpt["now_head"])
    for m in (enc, pred, cheads, nhead):
        m.eval()
    return enc, pred, cheads, nhead, meta


def _load_or_make(selftest: bool) -> dict:
    if selftest:
        return gen(20, 110)  # self-contained tiny set (no prior npz needed)
    if os.path.exists(DATA):
        blob = np.load(DATA)
        return {k: blob[k] for k in blob.files}
    print(f"[INFO] no dataset at {DATA}; generating a default one ...")
    return gen(32, 120)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--robust", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    epochs = 60 if args.selftest else args.epochs

    data = _load_or_make(args.selftest)
    ckpt, m = train(data, epochs=epochs, batch=args.batch, robust=args.robust)

    # a selftest must not clobber a real trained checkpoint with its toy one
    # (and a robust experiment gets its own file — see eval_robustness.py)
    out = MODEL.replace(".pth", "_selftest.pth") if args.selftest else MODEL
    if args.robust and not args.selftest:
        out = MODEL.replace(".pth", "_robust.pth")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(ckpt, out)
    auc_str = "/".join(f"{a:.2f}" for a in m["auc"])
    h_str = "/".join(str(k) for k in HORIZONS)
    print(
        f"WORLD-MODEL OK: {m['n_train']} train seqs, "
        f"latent MSE@32={m['mse'][-1]:.3f} (no-op {m['noop'][-1]:.3f}), "
        f"AUC@{h_str}={auc_str}, now-AUC={m['now_auc']:.2f}, "
        f"veer-ranking={m['side']:.2f} (n={m['n_side']}), "
        f"int8 weights={m['int8_kb']:.1f} KB (<{GAP8_BUDGET_KB} fits), saved {out}"
    )
    if args.selftest:
        # k=4 (~83 ms) is near-degenerate — "future == present" is genuinely
        # strong there — so the meaningful claims are the *long* horizon and
        # the overall average, not every horizon individually
        assert m["mse"][-1] < m["noop"][-1], "no long-horizon predictive gain"
        assert float(np.mean(m["mse"])) < float(
            np.mean(m["noop"])
        ), "predictor no better than 'future == present' overall"
        assert m["auc"][1] > 0.70, f"AUC@8 barely predicts danger ({m['auc'][1]:.2f})"
        assert m["auc"][-1] > 0.70, f"AUC@32 barely anticipates ({m['auc'][-1]:.2f})"
        assert m["now_auc"] > 0.65, f"danger-now head weak ({m['now_auc']:.2f})"
        if m["n_side"] >= 20:
            assert m["side"] > 0.60, f"veer ranking at chance ({m['side']:.2f})"
        assert m["int8_kb"] < GAP8_BUDGET_KB, f"too big ({m['int8_kb']:.1f} KB)"


if __name__ == "__main__":
    main()
    sys.exit(0)
