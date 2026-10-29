"""
Lesson 29 (step 5) — domain randomization: does the model survive a sim it
===========================================================================
never saw?

Everything so far was measured in the *same* clean simulator the model trained
in — the classic sim-to-real trap. This step *prices* that trap before anyone
pays it on hardware: take the shipped (clean-trained) model and measure it on
conditions it never saw — randomized pillar shapes/colours, 0-2 control steps
of command latency, ±8 % actuation noise, and Lesson 20's fixed unseen
appearance shift on every frame the policies see.

Three numbers, per the sim-to-real checklist:

  * clean-sim AUC        — does it work where it trained?
  * randomized-sim AUC   — does it work where it did *not* train?
  * closed-loop under randomization — AUC is a detector score; the drone flies
                           the *planner*. Crash rate and clearance, reactive vs
                           wm, everything randomized at once.

The gap this reports is the *budget* for Track B's fix, which ships alongside:
regenerate with `gen_wm_dataset.py --randomize` and retrain with
`train_world_model.py --robust` (Lesson 20's jitter on the online encoder) to
train across the variation instead of beside it. Honest scope either way:
randomization hardens against *modelled* gaps; the unmodelled ones (real
optics, real airflow) are exactly why Track E measures the sim-to-real gap on
hardware before trusting any of it.

Run:
  python src/eval_robustness.py --seeds 30
  python src/eval_robustness.py --selftest   # small, asserts
Needs output/world_model.pth (auto-trains a tiny one if missing).
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gen_wm_dataset import A_NORM, gen, make_env  # noqa: E402
from train_world_model import (  # noqa: E402
    MODEL,
    _index_samples,
    load_model,
    roc_auc,
    veer_ranking,
)
from wm_closed_loop import (  # noqa: E402
    ReactivePolicy,
    WMPolicy,
    reactive_policy,
    wm_policy,
    load_or_train,
    run_episode,
)

ROBUST_MODEL = MODEL.replace(".pth", "_robust.pth")

from randomize import shift_appearance  # noqa: E402  (Lesson 20's unseen camera)


def dataset_auc(data, enc, pred, cheads, device="cpu", shifted=False) -> float:
    """Warn-ring AUC at the longest horizon over a dataset's executed-action
    samples — optionally seen through Lesson 20's fixed appearance shift."""
    idx, c_h = _index_samples(data)
    L = data["frames"].shape[1]
    frames = data["frames"][idx[:, 0], idx[:, 1]].astype(np.float32) / 255.0
    if shifted:
        frames = shift_appearance(frames)
    acts = data["actions"][idx[:, 0], idx[:, 1]] / A_NORM
    del L
    with torch.no_grad():
        x = torch.tensor(frames, device=device).permute(0, 3, 1, 2)
        a = torch.tensor(acts, device=device)
        scores = []
        for i in range(0, len(x), 512):
            zh = pred(enc(x[i : i + 512]), a[i : i + 512])
            scores.append(torch.sigmoid(cheads(zh))[:, -1, 0])
        s = torch.cat(scores).cpu().numpy()
    return roc_auc(s, c_h[:, -1, 0])


def closed_loop(n_seeds, seed0, enc, pred, cheads, nhead, meta, randomize) -> dict:
    from learn_policy import LearnedPolicy, _load_policy, zip_path

    mk = {
        "reactive": lambda: reactive_policy(enc, nhead, meta),
        "wm": lambda: wm_policy(enc, pred, cheads, meta),
    }
    # step-6 policies join when trained: the clean-trained one shows what the
    # storm costs it; the storm-trained (_rand) ones show what training inside
    # the storm buys back
    for name, rec, rnd in (
        ("learned", False, False),
        ("learned-rand", False, True),
        ("learned-rnn-rand", True, True),
    ):
        path = zip_path(rec, rnd)
        if os.path.exists(path):
            model = _load_policy(path)
            mk[name] = lambda m=model: LearnedPolicy(m, enc, pred, cheads, meta)

    env = make_env()
    out = {name: {"crash": 0, "clear": []} for name in mk}
    for i in range(n_seeds):
        for name, factory in mk.items():
            run = run_episode(
                env, factory(), seed0 + i, in_path=True, randomize=randomize
            )
            out[name]["crash"] += int(run["crashed"])
            out[name]["clear"].append(run["min_clear"])
    env.close()
    for v in out.values():
        v["crash_rate"] = v["crash"] / n_seeds
        v["mean_clear"] = float(np.mean(v["clear"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    n_val, n_seeds = (8, 6) if args.selftest else (12, args.seeds)

    enc, pred, cheads, nhead, meta = load_or_train(device="cpu")

    print(f"[INFO] generating held-out clean + randomized val sets ({n_val} each)")
    clean = gen(n_val, 110, seed=909, randomize=False)
    rand = gen(n_val, 110, seed=909, randomize=True)
    auc_clean = dataset_auc(clean, enc, pred, cheads)
    auc_rand = dataset_auc(rand, enc, pred, cheads, shifted=True)
    veer_rand, n_veer = veer_ranking(rand, range(n_val), enc, pred, cheads, "cpu")

    print(f"[INFO] closed loop under randomization ({n_seeds} threatened courses)")
    loop = closed_loop(n_seeds, args.seed0, enc, pred, cheads, nhead, meta, True)

    print(
        f"ROBUST-WM OK: clean AUC@32={auc_clean:.2f} | randomized(+unseen-shift) "
        f"AUC@32={auc_rand:.2f} (the gap to buy back), "
        f"veer-ranking={veer_rand:.2f} (n={n_veer})\n"
        f"  closed loop under randomization — crash "
        + " / ".join(f"{k} {v['crash_rate']:.0%}" for k, v in loop.items())
        + " (clearance "
        + " / ".join(f"{v['mean_clear']:.2f}" for v in loop.values())
        + " m) — latency + actuation noise + unseen appearance, danger signal "
        "still camera-only"
    )
    if os.path.exists(ROBUST_MODEL):
        enc2, pred2, cheads2, nhead2, meta2 = load_model(ROBUST_MODEL)
        auc_rand2 = dataset_auc(rand, enc2, pred2, cheads2, shifted=True)
        loop2 = closed_loop(
            n_seeds, args.seed0, enc2, pred2, cheads2, nhead2, meta2, True
        )
        re2, wm2 = loop2["reactive"], loop2["wm"]
        print(
            f"  --randomize + --robust retrain: randomized AUC@32={auc_rand2:.2f}, "
            f"closed loop crash reactive {re2['crash_rate']:.0%} -> wm "
            f"{wm2['crash_rate']:.0%}, clearance {re2['mean_clear']:.2f} -> "
            f"{wm2['mean_clear']:.2f} m — train across the variation, not beside it"
        )
    if args.selftest:
        assert auc_clean > 0.90, f"shipped model fails even at home ({auc_clean:.2f})"
        assert auc_rand > 0.60, f"randomized AUC collapsed entirely ({auc_rand:.2f})"
        # the printed clean-vs-randomized gap IS the deliverable: the measured
        # budget that --randomize + --robust retraining must buy back


if __name__ == "__main__":
    main()
    sys.exit(0)
