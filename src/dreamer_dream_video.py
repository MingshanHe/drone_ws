"""'Dream' videos of the nano-Dreamer world model: what the camera saw vs what the model imagined.

For a few held-out rollouts, the left panel plays the real 64x64 camera frames (12 Hz model steps);
the right panel shows the model's posterior reconstruction during the first --context steps and
then its OPEN-LOOP imagination: from the belief at the end of the context, the prior is rolled
forward under the commands that were actually flown, no further frames are observed, and every
imagined state is decoded to an image and read out by the distance / collision heads.

  python src/dreamer_dream_video.py --model src/output/world_model_dreamer_pybullet.pth --data src/output/wm_dataset.npz --tag pybullet
Outputs src/output/report/dream_<tag>.mp4 and .gif.
"""

import argparse
import os
import sys

import cv2
import imageio
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dreamer.data import STRIDE, SeqData  # noqa: E402
from gen_wm_dataset import ACTION_NAMES  # noqa: E402
from train_dreamer import load_dreamer  # noqa: E402
from train_world_model import _split_rollouts  # noqa: E402

REP = os.path.join(HERE, "output", "report")
S = 256  # panel size (px)


def put(img, text, xy, scale=0.5, color=(255, 255, 255)):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def up(x):  # (3,64,64) float [0,1] -> BGR uint8 (S,S)
    img = (x.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
    return cv2.resize(np.ascontiguousarray(img), (S, S), interpolation=cv2.INTER_NEAREST)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--context", type=int, default=8, help="observed model steps before imagination starts")
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--all-rollouts", action="store_true", help="pick from every rollout (an unseen dataset) instead of the held-out split")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_dreamer(args.model, device)
    blob = np.load(args.data)
    data = {k: blob[k] for k in blob.files}
    rolls = list(range(data["frames"].shape[0])) if args.all_rollouts else _split_rollouts(data, np.random.default_rng(0))[1]
    sd = SeqData(data, device)
    ep = sd.episodes(rolls, phases=[0])
    x, d, a_id = ep["frames"], ep["dists"], data["act_id"][np.asarray(rolls)][:, ::STRIDE]
    N, T = x.shape[:2]
    embed = model.enc(x.flatten(0, 1)).view(N, T, -1)
    post, _ = model.rssm.observe(embed, ep["a_prev"])
    c = args.context
    fut = d[:, c:].min(1).values
    picks = torch.argsort((fut - 0.5).abs())[: args.rollouts].tolist()  # rollouts that come close (~0.5 m) without burying the camera in a wall
    frames_out = []
    for n in picks:
        st = model.rssm.index(post, slice(n, n + 1), c - 1)
        im = model.rssm.imagine(st, ep["actions"][n : n + 1, c - 1 : T - 1])  # steps c..T-1 under the flown commands
        f_post = model.rssm.feat(model.rssm.index(post, slice(n, n + 1), slice(0, c)))[0]  # (c, F)
        f_im = model.rssm.feat(im)[0]  # (T-c, F)
        feats = torch.cat([f_post, f_im])
        xh = model.dec(feats).float()
        dh = model.twohot.mean(model.dist_head(feats))
        p = model.p_col(feats)
        for t in range(T):
            left, right = up(x[n, t]), up(xh[t])
            phase = "observed: reconstruction" if t < c else f"IMAGINED (open loop)  +{(t - c + 1) * STRIDE} frames"
            canvas = np.full((S + 70, 2 * S + 12, 3), 252, np.uint8)
            canvas[58 : 58 + S, 0:S] = left
            canvas[58 : 58 + S, S + 12 :] = right
            put(canvas, f"camera  rollout {rolls[n]}  step {t}", (6, 16), 0.42, (30, 30, 30))
            put(canvas, f"command {ACTION_NAMES[int(a_id[n, t])]}", (6, 33), 0.42, (30, 30, 30))
            put(canvas, f"true clearance d = {float(d[n, t]):.2f} m", (6, 50), 0.42, (30, 30, 30))
            put(canvas, phase, (S + 18, 16), 0.42, (214, 120, 42) if t >= c else (30, 30, 30))
            put(canvas, f"d_hat = {float(dh[t]):.2f} m", (S + 18, 33), 0.42, (30, 30, 30))
            put(canvas, f"P<0.7m {float(p[t, 0]):.2f}   P<0.35m {float(p[t, 1]):.2f}", (S + 18, 50), 0.42, (30, 30, 30))
            bar_w = int(S * min(float(p[t, 0]), 1.0))
            cv2.rectangle(canvas, (S + 12, S + 62), (S + 12 + bar_w, S + 68), (42, 120, 214), -1)
            cv2.rectangle(canvas, (S + 12, S + 62), (2 * S + 12, S + 68), (120, 120, 120), 1)
            if t >= c:
                cv2.rectangle(canvas, (S + 12, 58), (2 * S + 11, 58 + S - 1), (42, 120, 214), 2)
            frames_out.append(canvas)
        frames_out += [frames_out[-1]] * int(args.fps)  # one-second pause between rollouts
    os.makedirs(REP, exist_ok=True)
    mp4 = os.path.join(REP, f"dream_{args.tag}.mp4")
    writer = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (frames_out[0].shape[1], frames_out[0].shape[0]))
    for f in frames_out:
        writer.write(f)
    writer.release()
    gif = os.path.join(REP, f"dream_{args.tag}.gif")
    small = [cv2.resize(f[:, :, ::-1], (520, int(f.shape[0] * 520 / f.shape[1])), interpolation=cv2.INTER_AREA) for f in frames_out]
    imageio.mimsave(gif, small, duration=1.0 / args.fps, loop=0)
    print(f"DREAM OK [{args.tag}]: rollouts {[rolls[n] for n in picks]}, {len(frames_out)} frames -> {mp4}, {gif} ({os.path.getsize(gif) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
