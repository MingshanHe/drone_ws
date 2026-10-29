"""Dreamer-style overview figure of the nano-Dreamer pipeline (PLAN_dreamerv3.md), three panels like
the DreamerV3 paper: (1) learning the world model from the dataset, unrolled over three 12 Hz steps,
(2) planning in imagination at test time, (3) the closed loop. Every image, latent, probability and
the decision shown are real: one dataset window and checkpoint A's actual forward pass / policy call.

  python src/draw_dreamer_pipeline.py   -> src/output/report/pipeline_dreamer.png
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "output")
REP = os.path.join(OUT, "report")

from draw_pipeline import BLUE, GRAY, GREEN, INK, INK2, MUTED, ORANGE, SURFACE, arr, thumb  # noqa: E402
from dreamer.data import STRIDE, SeqData  # noqa: E402
from dreamer_policy import DreamerMPCPolicy, N_SAMPLES  # noqa: E402
from gen_wm_dataset import ACTION_NAMES  # noqa: E402
from train_dreamer import CANDS, H_IMAG, load_dreamer  # noqa: E402
from wm_closed_loop import EVADE_HOLD, MARGIN_WM  # noqa: E402

ROLL, FRAME = 2, 18  # the dataset window also used by draw_pipeline.py: pillar ahead, command forward
CKPT = os.path.join(OUT, "world_model_dreamer_pybullet.pth")


@torch.no_grad()
def gather(device="cpu"):
    model, meta = load_dreamer(CKPT, device)
    blob = np.load(os.path.join(OUT, "wm_dataset.npz"))
    data = {k: blob[k] for k in blob.files}
    sd = SeqData(data, device)
    phase = FRAME % STRIDE
    # show the first decision step at which the real policy triggers on this rollout (the interesting moment)
    speed = float(data["speed"][ROLL])  # the policy's candidates at the speed this rollout was flown
    pol = DreamerMPCPolicy(model, meta, speed=speed)
    pol.begin([])
    frame_sel = None
    for t in range(phase, data["frames"].shape[1] - STRIDE * H_IMAG, STRIDE):
        if pol.decide(data["frames"][ROLL, t], np.zeros(20)) != ACTION_NAMES.index("forward") and frame_sel is None:
            frame_sel = t
    frame = frame_sel if frame_sel is not None else FRAME
    step = frame // STRIDE
    ep = sd.episodes([ROLL], phases=[phase])
    x = ep["frames"][0]
    T = x.shape[0]
    post, prior = model.rssm.observe(model.enc(x).view(1, T, -1), ep["a_prev"])
    steps = [step - 1, step, step + 1]
    feats = model.rssm.feat(post)[0]
    g = dict(frames=[x[s] for s in steps], recon=model.dec(feats[steps]).clamp(0, 1), dh=model.twohot.mean(model.dist_head(feats[steps])),
             p=model.p_col(feats[steps]), z=post["z"][0, steps].argmax(-1), zp=prior["z"][0, steps].argmax(-1),
             d=ep["dists"][0, steps], acts=[ACTION_NAMES[int(data["act_id"][ROLL, phase + STRIDE * s])] for s in steps],
             t_abs=[phase + STRIDE * s for s in steps])
    g["frame_id"] = frame
    g["cf_lab"] = sd.cf[ROLL, frame, :, :, 0].cpu().numpy()
    g["cf_pred"] = model.imagine_cands(model.rssm.index(post, slice(0, 1), step), CANDS * speed)[0, :, :, 0].cpu().numpy()
    g["speed"] = speed
    # the real policy call at FRAME: feed the frames it would have seen at 12 Hz
    pol = DreamerMPCPolicy(model, meta, speed=speed)
    pol.begin([])
    for t in range(phase, frame + 1, STRIDE):
        a = pol.decide(data["frames"][ROLL, t], np.zeros(20))
    probs = pol.probs(data["frames"][ROLL, frame])
    warn, crit = probs[:, :, 0] @ pol.h_w, probs[:, :, 1] @ pol.h_w
    g["menu"] = [ACTION_NAMES[i] for i in pol.ids]
    g["probs"], g["decision"] = probs, ACTION_NAMES[a]
    g["edge"] = float(warn[pol.i_fwd]) - min(float(warn[i]) for i in pol.i_veers)
    g["imminent"] = float(probs[pol.i_fwd, :2, 1].max())
    st = {k: v.repeat_interleave(len(pol.cands), 0) for k, v in pol.state.items()}
    fim = model.rssm.feat(model.rssm.imagine(st, pol.cands[:, None].expand(-1, H_IMAG, -1)))  # (5, 8, F)
    pick = [1, 3, 7]
    g["dreams"] = model.dec(fim[:, pick].flatten(0, 1)).clamp(0, 1).view(len(pol.cands), 3, 3, 64, 64)
    g["dream_d"] = model.twohot.mean(model.dist_head(fim[:, pick].flatten(0, 1))).view(len(pol.cands), 3)
    g["frame_now"] = data["frames"][ROLL, frame]
    rooms = np.load(os.path.join(OUT, "wm_dataset_rooms_unseen60.npz"))
    g["room_frame"] = rooms["frames"][0, 20]
    g["n_params"] = sum(p.numel() for p in model.parameters())
    return g


def box(ax, x, y, w, h, color, alpha=0.08, lw=1.0, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", facecolor=color, alpha=alpha, edgecolor=color, linewidth=lw, linestyle=ls, zorder=2))


def zgrid(ax, x, y, cls, size=5.2, dashed=False):
    """16 categorical groups as a 4x4 grid coloured by the chosen class."""
    cmap = plt.get_cmap("viridis")
    c = size / 4
    for i in range(16):
        ax.add_patch(Rectangle((x + (i % 4) * c, y + (3 - i // 4) * c), c, c, facecolor=cmap(int(cls[i]) / 15), edgecolor=SURFACE, linewidth=0.6, zorder=3))
    ax.add_patch(Rectangle((x, y), size, size, fill=False, edgecolor=INK2, linewidth=0.8, linestyle=(0, (2, 1.5)) if dashed else "-", zorder=4))


def img(ax, a, x, y, w, h):
    if torch.is_tensor(a):
        a = a.permute(1, 2, 0).cpu().numpy()
    thumb(ax, a, x, y, w, h)


def darr(ax, p, q, color=GREEN, lw=1.2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>", mutation_scale=9, linewidth=lw, color=color, shrinkA=0, shrinkB=0, zorder=5))


def draw(g):
    matplotlib.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    W, H = 150, 126
    fig, ax = plt.subplots(figsize=(W / 10, H / 10), facecolor=SURFACE)
    ax.set_xlim(0, W), ax.set_ylim(0, H), ax.set_aspect("equal"), ax.axis("off")
    ax.set_facecolor(SURFACE)
    fig.text(0.012, 0.988, f"nano-Dreamer：从数据集学一个带记忆的世界模型，再在想象里规划（{g['n_params']/1e6:.2f}M 参数，RSSM 12 Hz）", fontsize=12, fontweight="bold", color=INK, va="top")
    small = dict(fontsize=8, color=INK2, ha="center")
    tiny = dict(fontsize=7.5, color=MUTED, ha="center")
    head = dict(fontsize=11, color=INK, fontweight="bold", ha="left", va="center")

    # ================= ① learning the world model (unrolled over three 12 Hz steps)
    ax.text(2, 121, "① 从数据集学世界模型（训练，GPU）", **head)
    box(ax, 1, 78, 148, 41.5, GRAY, alpha=0.04, lw=0.6)
    for lbl, y in (("画面", 87.5), ("潜变量", 98.6), ("状态", 106.2)):
        ax.text(3.5, y, lbl, fontsize=8, color=MUTED, ha="center", va="center", rotation=90)
    cols = [16, 41, 66]
    blues = plt.get_cmap("Blues")
    for j, cx in enumerate(cols):
        img(ax, g["frames"][j], cx - 12, 82, 11, 11)
        img(ax, g["recon"][j], cx + 1, 82, 11, 11)
        ax.text(cx - 6.5, 81.2, f"$x_t$  帧 {g['t_abs'][j]}  d={float(g['d'][j]):.2f} m", fontsize=7, color=INK2, ha="center", va="top")
        ax.text(cx + 6.5, 81.2, "$\\hat x_t$ 解码重建", fontsize=7, color=INK2, ha="center", va="top")
        zgrid(ax, cx - 2.6, 96, g["z"][j])
        zgrid(ax, cx + 5.2, 96, g["zp"][j], dashed=True)
        ax.text(cx, 101.6, "$z_t$ posterior", fontsize=6.8, color=INK2, ha="center", va="bottom")
        ax.text(cx + 7.8, 101.6, "$\\hat z_t$ prior", fontsize=6.8, color=MUTED, ha="center", va="bottom")
        darr(ax, (cx + 2.8, 98.6), (cx + 5.0, 98.6), GREEN, 1.0)
        ax.text(cx + 3.9, 95.1, "KL", fontsize=6.5, color=GREEN, ha="center", va="top", fontweight="bold")
        box(ax, cx - 5, 104, 10, 4.5, BLUE, alpha=0.12)
        ax.text(cx, 106.25, "$h_t$  GRU 256", fontsize=7.5, color=BLUE, ha="center", va="center", fontweight="bold")
        arr(ax, (cx - 6.5, 93.3), (cx - 3.2, 95.7), BLUE, lw=1.0)  # encoder
        arr(ax, (cx + 1.2, 95.7), (cx + 5.5, 93.3), ORANGE, lw=1.0)  # decoder
        arr(ax, (cx - 1.2, 103.8), (cx - 1.2, 101.5), BLUE, lw=0.9)  # h -> posterior
        arr(ax, (cx + 1.8, 103.8), (cx + 6.5, 101.5), MUTED, lw=0.9, dashed=True)  # h -> prior
        ax.text(cx + 5.6, 108.1, f"$\\hat d$={float(g['dh'][j]):.2f} m", fontsize=7, color=ORANGE, ha="left", va="top")
        ax.text(cx + 5.6, 106.0, f"P<0.7={float(g['p'][j, 0]):.2f}", fontsize=7, color=ORANGE, ha="left", va="top")
        if j < 2:
            nx = cols[j + 1]
            arr(ax, (cx + 5.2, 104.6), (nx - 5.2, 104.6), BLUE, lw=1.0)  # h_t -> h_{t+1}
            arr(ax, (cx + 2.6, 101.3), (nx - 5.3, 104.1), BLUE, lw=0.8)  # z_t -> h_{t+1}
            for q in range(4):
                ax.add_patch(Rectangle((cx + 13.5 + q * 1.1, 102.2), 1.1, 1.1, facecolor=blues(0.3 + 0.5 * (q == 0)), edgecolor=SURFACE, linewidth=0.6, zorder=3))
            ax.text(cx + 18.5, 102.75, f"$a_t$ {g['acts'][j]}", fontsize=6.8, color=INK2, ha="left", va="center")
    ax.text(cols[-1] + 8, 104.6, "…", fontsize=12, color=MUTED, ha="left", va="center")
    ax.text(12, 116.4, "编码器 Conv 16/32/64/64 → posterior $q(z_t|h_t,x_t)$；GRU 转移 $h_{t+1}=f(h_t,z_t,a_t)$；prior $p(\\hat z_{t+1}|h_{t+1})$ 不看画面；解码器和距离 / 碰撞 / continue 头都读 $(h_t,z_t)$",
            fontsize=7.3, color=INK2, ha="left", va="center")
    ax.text(12, 113.6, "$z_t$ = 16 组 × 16 类的离散潜变量（格子颜色 = 选中的类），$h_t$ = 256 维 GRU 记忆；posterior 与 prior 格子不同的组 = 没看画面时猜错的组", fontsize=7.3, color=MUTED, ha="left", va="center")
    # counterfactual branch
    gx1, gx2, gy, cw, ch = 104, 121, 82, 3.0, 2.4
    ax.text(gx1 + 2 * cw + 8.5, 110.2 - 0.2, "", fontsize=1)
    ax.text(116, 108.0, "反事实想象一致性 $L_{cf}$：从信念 $s_t$ 出发，6 个命令各想象 8 步，读 P(k 步内 < 0.7 m)", fontsize=7.2, color=GREEN, ha="center", va="center", fontweight="bold")
    ax.text(gx1 + 2 * cw, 100.8, "oracle 标签", fontsize=7.5, color=INK2, ha="center", va="center")
    ax.text(gx2 + 2 * cw, 100.8, "模型想象", fontsize=7.5, color=INK2, ha="center", va="center")
    greys, oranges = plt.get_cmap("Greys"), plt.get_cmap("Oranges")
    for i, name in enumerate(ACTION_NAMES):
        yy = gy + (5 - i) * ch
        ax.text(gx1 - 0.6, yy + ch / 2, name, fontsize=6.6, color=INK2, ha="right", va="center")
        for k in range(4):
            v, pv = g["cf_lab"][i, k], g["cf_pred"][i, k]
            ax.add_patch(Rectangle((gx1 + k * cw, yy), cw, ch, facecolor=greys(0.25 + 0.5 * v), edgecolor=SURFACE, linewidth=1, zorder=3))
            ax.text(gx1 + k * cw + cw / 2, yy + ch / 2, f"{int(v)}", fontsize=6.3, color="white" if v else INK, ha="center", va="center", zorder=4)
            ax.add_patch(Rectangle((gx2 + k * cw, yy), cw, ch, facecolor=oranges(pv), edgecolor=SURFACE, linewidth=1, zorder=3))
            ax.text(gx2 + k * cw + cw / 2, yy + ch / 2, f"{pv:.2f}", fontsize=5.6, color="white" if pv > 0.55 else INK, ha="center", va="center", zorder=4)
    for k, s in enumerate(("+4", "+8", "+16", "+32")):
        ax.text(gx1 + k * cw + cw / 2, gy + 6 * ch + 0.4, s, fontsize=6.3, color=INK2, ha="center", va="bottom")
        ax.text(gx2 + k * cw + cw / 2, gy + 6 * ch + 0.4, s, fontsize=6.3, color=INK2, ha="center", va="bottom")
    darr(ax, (gx1 + 4 * cw + 0.5, gy + 3 * ch), (gx2 - 0.5, gy + 3 * ch), GREEN, 1.0)
    ax.text((gx1 + 4 * cw + gx2) / 2, gy + 3 * ch + 1.0, "BCE", fontsize=6.5, color=GREEN, ha="center", va="bottom", fontweight="bold")
    ax.text(136.5, 100.8, "损失", fontsize=8, color=GREEN, ha="left", va="center", fontweight="bold")
    for i, s in enumerate(("重建 MSE（逐像素求和）", "KL：dyn 1.0 / rep 0.1，free 1 nat", "净空 $d_t$：symlog two-hot 64 bins", "碰撞 <0.7 / <0.35 m、continue：BCE", "反事实：BCE（左图）")):
        ax.text(136.5, 97.8 - i * 3.0, s, fontsize=6.8, color=INK2, ha="left", va="center")
    ax.text(136.5, 82.5, "Adam 3e-4，batch 16 × 24 步\n20k 步 ≈ 30 min（RTX 5060 Ti）", fontsize=6.5, color=MUTED, ha="left", va="center", linespacing=1.4)

    # ================= ② planning in imagination
    ax.text(2, 75.2, f"② 测试时：在想象中规划（12 Hz，CPU 每次决策 3.5 ms；本例巡航 {g['speed']:.1f}× 基准速度）", **head)
    box(ax, 1, 36, 148, 36.5, GRAY, alpha=0.04, lw=0.6)
    img(ax, g["frame_now"], 2.5, 49, 11, 11)
    ax.text(8, 61.3, f"当前画面 $x_t$（帧 {g['frame_id']}）", fontsize=7.5, color=INK2, ha="center", va="bottom")
    arr(ax, (13.8, 54.5), (17.2, 54.5), BLUE, lw=1.0)
    box(ax, 17.5, 48.5, 13, 12, BLUE, alpha=0.12)
    ax.text(24, 56.3, "信念 $(h_t, z_t)$", fontsize=7.5, color=BLUE, ha="center", va="center", fontweight="bold")
    ax.text(24, 52.8, "编码 + posterior 更新\n（跨决策保留 = 记忆）", fontsize=6.3, color=INK2, ha="center", va="center", linespacing=1.3)
    rows_y = [70.2, 64, 57.8, 51.6, 45.4]
    ax.text(45, 72.6, "每个候选命令想象 8 步（prior）", fontsize=7, color=MUTED, ha="center", va="bottom")
    ax.text(62.5, 72.6, "解码 +8 / +16 / +32 帧", fontsize=7, color=MUTED, ha="center", va="bottom")
    ax.text(81.5, 72.6, "8 步内（8 条想象平均）", fontsize=7, color=MUTED, ha="center", va="bottom")
    ax.text(77, 70.3, "P<0.7", fontsize=6.8, color=INK2, ha="center", va="center")
    ax.text(85.5, 70.3, "P<0.35", fontsize=6.8, color=INK2, ha="center", va="center")
    ax.text(93.5, 70.3, "$\\hat d_{+32}$", fontsize=6.8, color=INK2, ha="center", va="center")
    rows_y = [67.5, 61.6, 55.7, 49.8, 43.9]
    for i, (name, yy) in enumerate(zip(g["menu"], rows_y)):
        chosen = name == g["decision"]
        if chosen:
            box(ax, 32.5, yy - 3.1, 65, 6.2, ORANGE, alpha=0.10, lw=1.2)
        arr(ax, (30.6, 54.5), (34.3, yy), BLUE, lw=0.7)
        ax.text(39.5, yy, name, fontsize=7.2, color=ORANGE if chosen else INK2, ha="right", va="center", fontweight="bold" if chosen else "normal")
        for s in range(8):
            ax.add_patch(Circle((41 + s * 1.9, yy), 0.45, facecolor=ORANGE, edgecolor=SURFACE, linewidth=0.4, zorder=4))
        ax.plot([41, 41 + 7 * 1.9], [yy, yy], color=ORANGE, linewidth=0.8, alpha=0.6, zorder=3)
        for k in range(3):
            img(ax, g["dreams"][i, k], 56 + k * 6.3, yy - 2.75, 5.5, 5.5)
        pw, pc = float(g["probs"][i, -1, 0]), float(g["probs"][i, -1, 1])
        ax.text(77, yy, f"{pw:.2f}", fontsize=7.5, color=INK, ha="center", va="center")
        ax.text(85.5, yy, f"{pc:.2f}", fontsize=7.5, color=INK, ha="center", va="center")
        ax.text(93.5, yy, f"{float(g['dream_d'][i, 2]):.2f} m", fontsize=7.5, color=INK, ha="center", va="center")
    arr(ax, (55.3, 54.5), (55.9, 54.5), ORANGE, lw=0.5)
    # decision box
    box(ax, 101, 38.5, 47.5, 32, ORANGE, alpha=0.07)
    ax.text(124.75, 68.2, "触发与代价：与单帧模型的 MPC 完全相同", fontsize=8, color=ORANGE, ha="center", va="center", fontweight="bold")
    lines = [f"触发：warn(前进) − min warn(绕行) ≥ {MARGIN_WM}，",
             "　　　或 170 ms 内 P(<0.35 m) > 0.5（兜底）",
             "代价 = 6·(0.25·warn + 0.75·crit) + 航向 + 1.5·|y| − 1.2·进度",
             "　　　（warn / crit 为 4 个视界的紧迫度加权和）",
             f"本例：差值 = {g['edge']:.2f}，兜底 = {g['imminent']:.2f} → 触发",
             f"→ 选代价最低的绕行：{g['decision']}，保持 0.5 s（{EVADE_HOLD} 步）"]
    for i, s in enumerate(lines):
        ax.text(103, 64.3 - i * 4.0, s, fontsize=7.2, color=ORANGE if i >= 4 else INK2, ha="left", va="center", fontweight="bold" if i == 5 else "normal")
    ax.plot([124.75, 124.75, 24], [38.4, 37.4, 37.4], color=BLUE, linewidth=0.9, linestyle=(0, (4, 2.5)), zorder=5)
    arr(ax, (24, 37.4), (24, 48.3), BLUE, lw=0.9, dashed=True)
    ax.text(72, 38.6, "执行的命令 $a_t$ → 48 Hz 速度环，并作为下一次信念更新的输入", fontsize=7, color=BLUE, ha="center", va="bottom")

    # ================= ③ closed loop
    ax.text(2, 33.2, "③ 数据与闭环", **head)
    bw, bh, by = 26, 20, 7
    xs = [4, 34, 64, 94, 124]
    titles = ["数据集", "训练", "checkpoint", "策略", "模拟器"]
    bodies = ["rollout 序列 @ 48 Hz：\n64×64 画面、保持的命令、\n净空距离、反事实 oracle 标签\nA 64 / B 256 / C 320 rollouts",
              "12 Hz 子采样（4 个相位）\n序列 24 步，batch 16\n20k 步，bf16，≈ 30 min\n三个训练集同一超参",
              f"{g['n_params']/1e6:.2f}M 参数 ≈ 1.2 MB int8\n编码器 + RSSM + 解码器\n+ 距离 / 碰撞 / continue 头\nmeta.family = \"dreamer\"",
              "DreamerMPCPolicy\n信念更新 + 5 候选 × 8 步 × 8 条\n触发 / 代价同 WMPolicy\n12 Hz，CPU 3.5 ms / 决策",
              "pybullet 柱子 300 条 course\nProcTHOR 房 16–23 300 起点\n48 Hz 速度控制，0.22 m 记撞"]
    for x, t, b in zip(xs, titles, bodies):
        box(ax, x, by, bw, bh, BLUE if t != "模拟器" else GRAY, alpha=0.08)
        ax.text(x + bw / 2, by + bh - 2.2, t, fontsize=8.5, color=INK, ha="center", va="center", fontweight="bold")
        ax.text(x + 1.2, by + bh - 5.0, b, fontsize=6.6, color=INK2, ha="left", va="top", linespacing=1.35)
    for x in xs[:-1]:
        arr(ax, (x + bw + 0.3, by + bh / 2 + 1.5), (x + 30 - 0.3, by + bh / 2 + 1.5), INK2, lw=1.0)
    arr(ax, (xs[4] - 0.3, by + 3.5), (xs[3] + bw + 0.3, by + 3.5), INK2, lw=1.0)
    ax.text(xs[3] + bw + 2, by + 1.8, "64×64 画面", fontsize=6.3, color=MUTED, ha="center", va="center")
    ax.text(xs[3] + bw + 2, by + bh / 2 + 3.2, "速度命令", fontsize=6.3, color=MUTED, ha="center", va="center")
    img(ax, g["frame_now"], xs[4] + 12.5, by + 1.2, 6, 6)
    img(ax, g["room_frame"], xs[4] + 19.2, by + 1.2, 6, 6)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(REP, "pipeline_dreamer.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


if __name__ == "__main__":
    g = gather()
    print(f"[INFO] window r{ROLL} frame {g['frame_id']}: decision {g['decision']}, edge {g['edge']:.2f}, imminent {g['imminent']:.2f}, P(<0.7 within 32) per cand {np.round(g['probs'][:, -1, 0].numpy(), 2).tolist()}")
    print("PIPELINE OK ->", draw(g))
