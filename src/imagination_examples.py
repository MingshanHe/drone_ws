"""Many open-loop imagination examples on one page: for each of --n sequences, true frames (top)
and the decoded imagination (bottom) after --context observed steps, with d / d_hat / P(<0.7).

  python src/imagination_examples.py --model src/output/world_model_dreamer_pb_plus_rooms6k.pth --data src/output/wm_dataset_rooms_unseen60.npz --all-rollouts --tag rooms_unseen_C6k --n 8
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REP = os.path.join(HERE, "output", "report")
from draw_pipeline import INK, INK2, MUTED, ORANGE, SURFACE  # noqa: E402
from dreamer.data import STRIDE, SeqData  # noqa: E402
from gen_wm_dataset import ACTION_NAMES  # noqa: E402
from train_dreamer import load_dreamer  # noqa: E402
from train_world_model import _split_rollouts  # noqa: E402

STEPS = [1, 2, 3, 4, 6, 8, 12, 16, 22]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--context", type=int, default=8)
    ap.add_argument("--all-rollouts", action="store_true", help="same as --split all (an unseen dataset)")
    ap.add_argument("--split", choices=("held-out", "train", "all"), default="held-out", help="which rollouts of --data: the model's held-out 20%%, its training 80%%, or all")
    ap.add_argument("--scene", choices=("any", "rooms", "pybullet"), default="any", help="keep only room (scene_type 1) or pybullet (0) rollouts")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
    matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_dreamer(args.model, device)
    blob = np.load(args.data)
    data = {k: blob[k] for k in blob.files}
    split = "all" if args.all_rollouts else args.split
    if split == "all":
        rolls = list(range(data["frames"].shape[0]))
    else:  # the same rollout-level split (seed 0) that train_dreamer.py used for this dataset
        tr, va = _split_rollouts(data, np.random.default_rng(0))
        rolls = tr if split == "train" else va
    if args.scene != "any" and "scene_type" in data:
        rolls = [r for r in rolls if int(data["scene_type"][r]) == (1 if args.scene == "rooms" else 0)]
    if len(rolls) > 128:  # keep GPU memory bounded for the big model; spread over the set
        rolls = sorted(np.random.default_rng(1).choice(rolls, 128, replace=False).tolist())
    sd = SeqData(data, device)
    ep = sd.episodes(rolls, phases=[0])
    x, d = ep["frames"], ep["dists"]
    N, T = x.shape[:2]
    post, _ = model.rssm.observe(model.encode(x.flatten(0, 1)).view(N, T, -1), ep["a_prev"])
    c = args.context
    # spread the examples over the range of "how close it gets": evenly spaced quantiles of the min future clearance
    fut = d[:, c:].min(1).values
    order = torch.argsort(fut).tolist()
    picks = [order[int(q)] for q in np.linspace(0, len(order) - 1, args.n)]
    ncol = len(STEPS) + 1
    fig, axes = plt.subplots(2 * args.n, ncol, figsize=(1.45 * ncol, 2.95 * args.n), facecolor=SURFACE)
    for row, n in enumerate(picks):
        st = model.rssm.index(post, slice(n, n + 1), c - 1)
        fim = model.rssm.feat(model.rssm.imagine(st, ep["actions"][n : n + 1, c - 1 : T - 1]))[0]
        xh = model.decode(fim).float().clamp(0, 1)
        dh = model.twohot.mean(model.dist_head(fim))
        pc = model.p_col(fim)[:, 0]
        a_id = data["act_id"][rolls[n]][::STRIDE]
        for col in range(ncol):
            at, ai = axes[2 * row, col], axes[2 * row + 1, col]
            if col == 0:
                at.imshow(x[n, c - 1].permute(1, 2, 0).cpu().numpy(), interpolation="nearest")
                ai.imshow(model.decode(model.rssm.feat(st)).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy(), interpolation="nearest")
                at.set_title(f"rollout {rolls[n]}  last observed  d={float(d[n, c - 1]):.2f}", fontsize=7, color=INK2)
                ai.set_title("posterior reconstruction", fontsize=7, color=MUTED)
                at.set_ylabel("true", fontsize=8, color=INK2)
                ai.set_ylabel("imagined", fontsize=8, color=ORANGE, fontweight="bold")
            else:
                s = STEPS[col - 1]
                at.imshow(x[n, c - 1 + s].permute(1, 2, 0).cpu().numpy(), interpolation="nearest")
                ai.imshow(xh[s - 1].permute(1, 2, 0).cpu().numpy(), interpolation="nearest")
                at.set_title(f"+{s * STRIDE}  d={float(d[n, c - 1 + s]):.2f}  {ACTION_NAMES[int(a_id[c - 1 + s])][:5]}", fontsize=6.6, color=INK2)
                ai.set_title(f"$\\hat d$={float(dh[s - 1]):.2f}  P<0.7={float(pc[s - 1]):.2f}", fontsize=6.6, color=ORANGE)
            for a in (at, ai):
                a.set_xticks([]), a.set_yticks([])
                for sp in a.spines.values():
                    sp.set_edgecolor(ORANGE if a is ai and col > 0 else INK2), sp.set_linewidth(1.0 if a is ai and col > 0 else 0.5)
    fig.suptitle(args.title or f"{os.path.basename(args.model)} on {os.path.basename(args.data)}: open-loop imagination after {c} observed steps ({args.n} sequences spread over closest approach)", fontsize=10, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path = os.path.join(REP, f"imagination_examples_{args.tag}.png")
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    print(f"EXAMPLES OK [{args.tag}]: rollouts {[rolls[n] for n in picks]} -> {path}")


if __name__ == "__main__":
    main()
