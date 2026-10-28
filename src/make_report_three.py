"""Build drone_ws/REPORT.md (zh) and REPORT_en.md (en): three training sets x two avoidance tasks.

  python src/make_report_three.py            # both languages
  python src/make_report_three.py --lang en  # English only (REPORT_en.md + results_three_en.png)
"""

import argparse
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
REP = os.path.join(OUT, "report")
EV = os.path.join(OUT, "evals")
ROOT = os.path.dirname(HERE)

MODELS = [  # tag, dataset file
    ("pybullet_ref", "wm_dataset.npz"),
    ("rooms", "wm_dataset_rooms_train60.npz"),
    ("pb_plus_rooms", "wm_dataset_pb_plus_rooms.npz"),
]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
GRAY = "#b5b4ad"

STR = {
    "zh": dict(
        labels=["A. 纯 pybullet", "B. 纯 hyperdrone 室内", "C. pybullet + hyperdrone 室内"],
        merged="C. 合并",
        font=["WenQuanYi Zen Hei", "DejaVu Sans"],
        fig="results_three.png",
        report="REPORT.md",
        t_pb="pybullet 柱子避障：成功率（in-path course 无穿越）",
        t_room="hyperdrone 室内避障：存活率（7.5 s 内无接触）",
        t_sweep="pybullet 单柱子速度扫描：wm 撞击率",
        leg_wm="wm 策略（潜变量 MPC）",
        leg_re="reactive 基线（danger-now 头）",
        suptitle="三种训练集训出的世界模型，在两个避障任务里的表现（左、中越高越好，右越低越好；误差线为 95% 置信区间）",
        conclusions="three_conclusions.md",
        offline_map={},
    ),
    "en": dict(
        labels=["A. pybullet only", "B. hyperdrone indoor only", "C. pybullet + hyperdrone indoor"],
        merged="C. merged",
        font=["DejaVu Sans"],
        fig="results_three_en.png",
        report="REPORT_en.md",
        t_pb="pybullet pillar avoidance\nsuccess rate (in-path courses, no hit)",
        t_room="hyperdrone indoor avoidance\nsurvival rate (no contact within 7.5 s)",
        t_sweep="pybullet single-pillar speed sweep\nwm crash rate",
        leg_wm="wm policy (latent MPC)",
        leg_re="reactive baseline (danger-now head)",
        suptitle="World models trained on three datasets, evaluated on two avoidance tasks (left and middle: higher is better; right: lower is better; whiskers = 95% CI)",
        conclusions="three_conclusions_en.md",
        offline_map={
            "**碰撞预判 AUC@32（0.5 = 随机）**": "**Collision-foresight AUC@32 (0.5 = chance)**",
            "**碰撞预判 AUC@8**": "**Collision-foresight AUC@8**",
            "**安全绕行方向排序准确率（0.5 = 随机）**": "**Safe-veer ranking accuracy (0.5 = chance)**",
            "**潜变量预测增益 @32（0 = 不比「未来 = 现在」好）**": "**Latent prediction gain @32 (0 = no better than \"future = now\")**",
            "**解码未来帧增益 @32（0 = 不比复制当前帧好）**": "**Decoded future-frame gain @32 (0 = no better than copying the current frame)**",
            "| 训练数据 \\ 数据域 |": "| training data \\ domain |",
            "\\* = 该数据域参与了这个 checkpoint 的训练（in-sample）；其余单元格是训练时没见过的数据。":
                "\\* = this domain was part of the checkpoint's training data (in-sample); every other cell is data unseen during training.",
        },
    ),
}


def log(name, kind):
    p = os.path.join(EV, f"{name}_{kind}.log")
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""


def rel(p):
    return os.path.relpath(p, ROOT).replace(os.sep, "/")


def wilson_half(p, n, z=1.96):
    if not n:
        return 0.0
    return 100 * z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)


def cluttered(name):
    s = log(name, "cluttered")
    g = lambda rx: re.search(rx, s)
    n = g(r"seeds=(\d+) \((\d+) in-path / (\d+) clear\)")
    crash = g(r"crash_rate:\s+reactive (\d+)% -> wm (\d+)%")
    clear = g(r"mean_min_clearance: ([\d.]+) m -> ([\d.]+) m")
    lead = g(r"mean_trigger_lead: \+?(-?\d+) ms \(n=(\d+)")
    fp = g(r"false_positive:\s+reactive (\d+)% -> wm (\d+)%")
    if not (n and crash):
        return None
    return dict(n=int(n.group(1)), n_ip=int(n.group(2)), n_cl=int(n.group(3)), crash_r=int(crash.group(1)), crash_w=int(crash.group(2)),
                clear_r=float(clear.group(1)), clear_w=float(clear.group(2)), lead=int(lead.group(1)) if lead else None, lead_n=int(lead.group(2)) if lead else 0,
                fp_r=int(fp.group(1)), fp_w=int(fp.group(2)))


def sweep(name):
    rows = re.findall(r"v=([\d.]+) m/s: crash reactive (\d+)% / wm (\d+)%", log(name, "sweep"))
    n = re.search(r"SPEED-SWEEP OK: (\d+)", log(name, "sweep"))
    return [(float(v), int(r), int(w)) for v, r, w in rows], (int(n.group(1)) if n else 0)


def rooms(name):
    s = log(name + "@hdroom", "rooms")
    g = lambda rx: re.search(rx, s)
    n = g(r"starts=(\d+) \[hyperdrone/rooms (\d+)-(\d+)")
    surv = g(r"survival \(no contact in 7.5 s\): reactive (\d+)% -> wm (\d+)%")
    ttc = g(r"time to first contact \(contacted runs\): reactive ([\d.]+) s -> wm ([\d.]+) s")
    ft = g(r"mean flight time:\s+reactive ([\d.]+) s -> wm ([\d.]+) s")
    dist = g(r"mean distance flown: reactive ([\d.]+) m -> wm ([\d.]+) m")
    clr = g(r"mean clearance:\s+reactive ([\d.]+) m -> wm ([\d.]+) m")
    ev = g(r"evaded at least once: reactive (\d+)% -> wm (\d+)%")
    if not (n and surv):
        return None
    return dict(n=int(n.group(1)), h0=int(n.group(2)), h1=int(n.group(3)), surv_r=int(surv.group(1)), surv_w=int(surv.group(2)),
                ttc_r=float(ttc.group(1)), ttc_w=float(ttc.group(2)), ft_r=float(ft.group(1)), ft_w=float(ft.group(2)),
                dist_r=float(dist.group(1)), dist_w=float(dist.group(2)), clr_r=float(clr.group(1)), clr_w=float(clr.group(2)),
                ev_r=int(ev.group(1)), ev_w=int(ev.group(2)))


def oracle():
    """(label, starts, survival %, mean flight s, mean distance m, evaded %) per oracle variant, from hd_oracle.py's log."""
    s = log("oracle", "rooms")
    return [(lab, int(n), int(sv), float(ft), float(d), int(ev)) for lab, n, sv, ft, d, ev in
            re.findall(r"ORACLE OK \[(.*?)\]: starts=(\d+) survival (\d+)%, mean flight time ([\d.]+) s, mean distance ([\d.]+) m, evaded at least once (\d+)%", s)]


def training(name):
    s = open(os.path.join(EV, "train_three.log"), encoding="utf-8").read() if os.path.exists(os.path.join(EV, "train_three.log")) else ""
    m = re.search(r"=== TRAIN " + re.escape(name) + r" .*?\n.*?training on \w+: (\d+) train / (\d+) val samples \((\d+)/(\d+) rollouts\).*?WORLD-MODEL OK: .*?latent MSE@32=([\d.]+) \(no-op ([\d.]+)\), AUC@4/8/16/32=([\d./]+), now-AUC=([\d.]+), veer-ranking=([\d.]+|nan) \(n=(\d+)\)", s, re.S)
    return m.groups() if m else None


def dataset_row(file):
    p = os.path.join(OUT, file)
    if not os.path.exists(p):
        return None
    d = np.load(p)
    R, L = d["frames"].shape[:2]
    st = d["scene_type"] if "scene_type" in d.files else np.zeros(R, dtype=np.int8)
    houses = sorted(set(d["scene_id"].tolist()) - {"pillars", "wm_dataset-pillars"}) if "scene_id" in d.files else []
    return R, L, int((st == 0).sum()), int((st == 1).sum()), len(houses), os.path.getsize(p) / 1e6


def rounded_bar(ax, y, width, height, color, xmax):
    from matplotlib.patches import FancyBboxPatch, Rectangle

    if width <= 0:
        return
    r = min(xmax * 0.011, width / 2)
    ax.add_patch(FancyBboxPatch((0, y - height / 2), width, height, boxstyle=f"round,pad=0,rounding_size={r}", linewidth=0, facecolor=color, mutation_aspect=0.1))
    ax.add_patch(Rectangle((0, y - height / 2), min(r, width), height, linewidth=0, facecolor=color))


def figure(t):
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = t["font"]
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    two_line = "\n" in t["t_pb"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3 if two_line else 4.0), facecolor=SURFACE)
    ys = list(range(len(MODELS)))[::-1]

    def style(ax):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.set_axisbelow(True)

    # A: pybullet success, B: indoor survival (wm blue, reactive gray, CI whiskers on wm)
    for ax, title, getter in (
        (axes[0], t["t_pb"], lambda g: (lambda c: (100 - c["crash_w"], 100 - c["crash_r"], c["n_ip"]) if c else None)(cluttered(g))),
        (axes[1], t["t_room"], lambda g: (lambda r: (r["surv_w"], r["surv_r"], r["n"]) if r else None)(rooms(g))),
    ):
        style(ax)
        ax.xaxis.grid(True, color=GRID, linewidth=1)
        for y, (tag, _) in zip(ys, MODELS):
            v = getter(tag)
            if v is None:
                ax.text(1, y, "–", va="center", fontsize=8, color=MUTED)
                continue
            w, r, n = v
            rounded_bar(ax, y + 0.17, r, 0.3, GRAY, 112)
            rounded_bar(ax, y - 0.17, w, 0.3, SERIES[0], 112)
            half = wilson_half(w / 100, n)
            ax.plot([max(w - half, 0), min(w + half, 100)], [y - 0.17, y - 0.17], color=INK2, linewidth=1, zorder=4)
            for xx in (max(w - half, 0), min(w + half, 100)):
                ax.plot([xx, xx], [y - 0.26, y - 0.08], color=INK2, linewidth=1, zorder=4)
            ax.text(min(w + half, 100) + 1.5, y - 0.17, f"{w:.0f}%", va="center", fontsize=8.5, color=INK)
            ax.text(r + 1.5, y + 0.17, f"{r:.0f}%", va="center", fontsize=7.5, color=MUTED)
        ax.set_xlim(0, 112)
        ax.set_ylim(-0.7, len(MODELS) - 0.3)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0", "25", "50", "75", "100%"])
        ax.set_yticks(ys)
        ax.set_yticklabels(t["labels"], fontsize=8.5, color=INK2)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=8)
    # C: crash vs speed, one line per model
    ax = axes[2]
    style(ax)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ends = []
    for j, (tag, _) in enumerate(MODELS):
        rows, n = sweep(tag)
        if not rows:
            continue
        xs, ws = [r[0] for r in rows], [r[2] for r in rows]
        ax.plot(xs, ws, color=SERIES[j], linewidth=2, solid_capstyle="round", solid_joinstyle="round", label=t["labels"][j], zorder=3)
        ax.scatter(xs, ws, s=36, color=SERIES[j], edgecolor=SURFACE, linewidth=2, zorder=4)
        ends.append([xs[-1], ws[-1], ws[-1]])
    ends.sort(key=lambda e: e[1])
    for a, b in zip(ends, ends[1:]):
        if b[2] - a[2] < 8:
            b[2] = a[2] + 8
    for x, w, y in ends:
        ax.text(x + 0.03, min(y, 100), f"{w:.0f}%", fontsize=7.5, color=INK2, va="center")
    ax.set_xlim(0.72, 1.78)
    ax.set_ylim(-4, 134)
    ax.set_xticks([0.8, 1.0, 1.2, 1.4, 1.6])
    ax.set_xticklabels(["0.8", "1.0", "1.2", "1.4", "1.6 m/s"])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    ax.set_title(t["t_sweep"], fontsize=9.5, color=INK, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper left", ncol=1, handlelength=1.5)
    fig.legend([Patch(color=SERIES[0]), Patch(color=GRAY)], [t["leg_wm"], t["leg_re"]], loc="lower center",
               bbox_to_anchor=(0.36, 0.0), ncol=2, frameon=False, fontsize=8, labelcolor=INK2)
    fig.text(0.01, 0.98, t["suptitle"], fontsize=11, color=INK, va="top")
    fig.tight_layout(rect=(0, 0.07, 1, 0.90 if two_line else 0.92))
    path = os.path.join(REP, t["fig"])
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def decode_table(lang):
    """Section 4.4 table from decode_compare.py: decoded-frame gain @8 / @32 per model x domain."""
    import json

    p = os.path.join(REP, "decode_three.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    keys = ["pybullet", "rand_pillars", "rooms_unseen"]
    head = {"zh": ["pybullet 柱子", "hyperdrone 柱子，随机外观（未见）", "hyperdrone 房间 16–23（未见）"],
            "en": ["pybullet pillars", "hyperdrone pillars, random appearance (unseen)", "hyperdrone rooms 16–23 (unseen)"]}[lang]
    first = {"zh": "训练集 \\ 数据域（增益 @8 / @32）", "en": "training set \\ domain (gain @8 / @32)"}[lang]
    lines = [f"| {first} | " + " | ".join(head) + " |", "|---|" + "---|" * len(keys)]
    for tag, label in zip([m[0] for m in MODELS], STR[lang]["labels"]):
        if tag not in d:
            continue
        cells = [f"{d[tag][k]['8']['gain']:+.2f} / {d[tag][k]['32']['gain']:+.2f}" + ("\\*" if d[tag][k]["seen"] else "") for k in keys]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build(lang):
    t = STR[lang]
    labels = t["labels"]
    fig_path = figure(t)
    C = {g: cluttered(g) for g, _ in MODELS}
    S = {g: sweep(g) for g, _ in MODELS}
    R = {g: rooms(g) for g, _ in MODELS}
    T = {g: training(g) for g, _ in MODELS}
    D = {g: dataset_row(f) for g, f in MODELS}
    O = oracle()
    dec_tab = decode_table(lang)
    pipe_fig = rel(os.path.join(REP, "pipeline_three_en.png" if lang == "en" else "pipeline_three.png"))
    dec_fig = rel(os.path.join(REP, "decode_three_en.png" if lang == "en" else "decode_three.png"))
    off = os.path.join(REP, "three_way.md")
    offline = open(off, encoding="utf-8").read().strip() if os.path.exists(off) else "_(offline grid missing)_"
    for src, dst in t["offline_map"].items():
        offline = offline.replace(src, dst)
    any_c = next((c for c in C.values() if c), None)
    any_r = next((r for r in R.values() if r), None)
    any_s = next((s for s in S.values() if s[1]), ([], 0))
    en = lang == "en"

    ds_rows = []
    for (g, f), label in zip(MODELS, labels):
        d = D[g]
        if d:
            size = f"{d[0]} rollouts × {d[1]} steps" if en else f"{d[0]} 条 × {d[1]} 步"
            scene = f"pillars {d[2]} / rooms {d[3]} ({d[4]} houses)" if en else f"柱子 {d[2]} / 房间 {d[3]}（{d[4]} 栋）"
            ds_rows.append(f"| {label} | `{f}` | {size} | {scene} | {d[5]:.0f} MB |")
    tr_rows = []
    for (g, _), label in zip(MODELS, labels):
        r = T[g]
        tr_rows.append(f"| {label} | `world_model_{g}.pth` | " + (f"{r[0]} / {r[1]} | {r[4]} ({r[5]}) | {r[6]} | {r[7]} | {r[8]} (n={r[9]}) |" if r else "– | – | – | – | – |"))
    pb_rows, rm_rows = [], []
    for (g, _), label in zip(MODELS, labels):
        c, r = C[g], R[g]
        if c:
            ci = wilson_half(c["crash_w"] / 100, c["n_ip"])
            pb_rows.append(f"| {label} | {c['crash_r']}% → **{c['crash_w']}%** (±{ci:.0f}) | {c['clear_r']:.2f} → {c['clear_w']:.2f} m | "
                           f"{(str(c['lead']) + ' ms (n=' + str(c['lead_n']) + ')') if c['lead'] is not None else '–'} | {c['fp_r']}% → {c['fp_w']}% | "
                           + " / ".join(f"{w}%" for _, _, w in S[g][0]) + " |")
        if r:
            ci = wilson_half(r["surv_w"] / 100, r["n"])
            rm_rows.append(f"| {label} | {r['surv_r']}% → **{r['surv_w']}%** (±{ci:.0f}) | {r['ttc_r']:.1f} → {r['ttc_w']:.1f} s | {r['ft_r']:.1f} → {r['ft_w']:.1f} s | "
                           f"{r['dist_r']:.2f} → {r['dist_w']:.2f} m | {r['clr_r']:.2f} → {r['clr_w']:.2f} m | {r['ev_r']}% → {r['ev_w']}% |")
    or_rows = []
    for i, (lab, n, sv, ft, d, ev) in enumerate(O):
        name = (("oracle, MPC action menu (forward / 0.5 s lateral veers, fixed yaw)", "oracle + hover when boxed in")[i] if en
                else ("oracle，MPC 动作菜单（前进 / 0.5 s 左右平移、偏航固定）", "oracle + 被堵住就悬停")[i]) if i < 2 else lab
        or_rows.append(f"| {name} | **{sv}%** (±{wilson_half(sv / 100, n):.0f}) | {ft:.1f} s | {d:.2f} m | {ev}% |")

    def gif(name, caption):
        g, m = os.path.join(REP, f"{name}.gif"), os.path.join(REP, f"{name}.mp4")
        if not os.path.exists(g):
            return ""
        return f"**{caption}**" + (f" — [mp4]({rel(m)})" if os.path.exists(m) else "") + f"\n\n![{caption}]({rel(g)})\n"

    cpath = os.path.join(REP, t["conclusions"])
    conclusions = open(cpath, encoding="utf-8").read().strip() if os.path.exists(cpath) else "<!-- CONCLUSIONS -->"
    fig_rel = rel(fig_path)
    A, B, Cm = labels[0], labels[1], t["merged"]
    n_c, ip_c, cl_c = (any_c["n"], any_c["n_ip"], any_c["n_cl"]) if any_c else ("?", "?", "?")
    n_r, h0, h1, k_h = (any_r["n"], any_r["h0"], any_r["h1"], any_r["h1"] - any_r["h0"] + 1) if any_r else ("?", "?", "?", "?")
    n_or = O[0][1] if O else "?"

    repro = f"""```bash
# {'data (conda env dreamer)' if en else '数据（conda env dreamer）'}
python src/gen_wm_dataset_hyperdrone.py --scene rooms --rollouts 256 --rooms 256 --houses 16 --house-first 0 --len 120 --fov-deg 60 --camera body --out src/output/wm_dataset_rooms_train60.npz
python src/merge_datasets.py src/output/wm_dataset.npz src/output/wm_dataset_rooms_train60.npz --out src/output/wm_dataset_pb_plus_rooms.npz
# {'training (GPU)' if en else '训练（GPU）'}
python src/train_world_model.py --epochs 80 --data src/output/wm_dataset_rooms_train60.npz --out src/output/world_model_rooms.pth
# {'pybullet task (conda env drone_python)' if en else 'pybullet 任务（conda env drone_python）'}
python src/eval_world_model_policy.py --seeds 300 --model src/output/world_model_rooms.pth
python src/speed_sweep.py --seeds 60 --no-learned --model src/output/world_model_rooms.pth --plot src/output/speed_sweep_rooms.png
# {'hyperdrone indoor task and its oracle ceiling (conda env dreamer)' if en else 'hyperdrone 室内任务及其 oracle 上限（conda env dreamer）'}
python src/hd_closed_loop.py --scene rooms --houses 8 --house-first 16 --seeds 300 --sweep-seeds 0 --model src/output/world_model_rooms.pth
python src/hd_oracle.py --houses 8 --house-first 16 --seeds 300 | tee src/output/evals/oracle_rooms.log
# {'offline prediction metrics, videos, this report' if en else '离线预测指标、视频、本报告'}
python src/eval_offline.py --models pybullet_ref rooms pb_plus_rooms --domains pybullet rand_pillars rooms_train rooms_unseen --out three_way
python src/hd_video.py --scene rooms --house 17 --seed 1017 --model src/output/world_model_rooms.pth --tag room17_rooms
python src/decode_compare.py --lang both
python src/draw_pipeline.py --lang both
python src/make_report_three.py --lang {'en' if en else 'zh'}
```"""

    if en:
        report = f"""# Three training sets × two avoidance tasks: how the world model performs

[中文版 / Chinese version](REPORT.md)

The same world model (the V-JEPA-style latent model + collision heads + MPC avoidance policy of nanodrone-ai lesson 29, `train_world_model.py --epochs 80`, identical training script and hyper-parameters for all three) is trained on three datasets and then tested closed-loop on two avoidance tasks. Every script and parameter is documented in [src/README_hyperdrone_dataset.md](src/README_hyperdrone_dataset.md); the earlier exploratory experiments (each simulator's pillar look, random appearance, the large merged datasets) are archived in [src/output/report/REPORT_archive_2026-10-27.md](src/output/report/REPORT_archive_2026-10-27.md) (Chinese).

## 1. The three training sets

| training set | file | size | scenes | file size |
|---|---|---|---|---|
{chr(10).join(ds_rows)}

- **{A}**: pillar courses from the original lesson's `gen_wm_dataset.py` (gym-pybullet-drones, CF2X + PID, 64×64, 60° body-fixed camera).
- **{B}**: `gen_wm_dataset_hyperdrone.py --scene rooms`, 16 rollouts in each of 16 furnished ProcTHOR houses (catalog entries 0–15; hyperlux ray tracing, crazyflie + RAPTOR, the same 64×64, 60° body-fixed camera). The "danger" labels come from each house's occupancy grid / distance field at flight altitude (`wm_rooms.py`).
- **{Cm}**: A and B concatenated rollout-wise (`merge_datasets.py`).

All three share the same action vocabulary (forward / slow / veer_left / veer_right / climb / hover, held segments of about 1 s), the same 4/8/16/32-step (83–667 ms) collision labels and the same counterfactual-label format.

## 2. Training (offline metrics on each dataset's held-out rollouts)

| training set | checkpoint | train / val windows | latent MSE@32 (no-op) | AUC@4/8/16/32 | now-AUC | veer ranking |
|---|---|---|---|---|---|---|
{chr(10).join(tr_rows)}

How the model is trained, from dataset to checkpoint (`train_world_model.py`): a 64×64 frame goes through a three-layer conv encoder with bearing-aware strip pooling into a 64-d latent; an action-conditioned predictor forecasts the latent 4/8/16/32 steps ahead as residuals; collision heads read "within 0.7 m / 0.35 m within k steps" off the predicted latents, and a danger-now head reads the current latent. The prediction target is an EMA copy of the encoder (JEPA-style, no pixel loss), and the collision heads are also supervised with the dataset's counterfactual oracle for all six commands. The whole model is 83k parameters (≈ 81 KB int8). In the figure the top row is the online model, the bottom row holds the targets of the same window (EMA latents, true distance labels) and every green vertical arrow is a loss between a prediction above and its target below; the frames, latent bars and probability cells are a real held-out window and the actual forward pass of model A.

![training pipeline]({pipe_fig})

## 3. The two test tasks

**pybullet pillar avoidance** (the original lesson's scoreboard, `eval_world_model_policy.py` / `speed_sweep.py`, conda env `drone_python`): {n_c} random courses ({ip_c} with a pillar directly ahead, {cl_c} safe to fly straight), 360 steps each with a 3 m finish line; the policy decides every 1/12 s from the 64×64 frame; a planar distance to a pillar axis below 0.22 m counts as a hit. A single-pillar speed sweep adds {any_s[1]} courses per speed, cruising at 0.8 → 1.6 m/s.

**hyperdrone indoor avoidance** (`hd_closed_loop.py --scene rooms`, conda env `dreamer`): {n_r} random starts in {k_h} ProcTHOR houses never used in training (catalog entries {h0}–{h1}; clearance ≥ 0.9 m, ≥ 2 m free ahead, random heading). The policy behaves as usual, flying forward unless it sees a threat and veering when it does, until the first contact with an obstacle (surface clearance from the distance field + 0.18 m < 0.22 m, the same test as for pillars) or until 7.5 s elapse. ProcTHOR houses contain almost no pillar-style courses (a free-standing obstacle, 3 m of straight path, room to pass on both sides), so the indoor task is a survival test.

In both tasks the reactive baseline (the same checkpoint's danger-now head + privileged veer direction) shares the encoder with the wm policy, so both change with the training set; the wm policy is the actual use of the world model.

## 4. Results

![results]({fig_rel})

### 4.1 pybullet pillar avoidance

| training set | crash rate reactive → wm (±95% CI) | min clearance | wm triggers earlier than reactive by | false triggers on clear courses | speed sweep, wm crash rate @0.8/1.0/1.2/1.4/1.6 m/s |
|---|---|---|---|---|---|
{chr(10).join(pb_rows)}

### 4.2 hyperdrone indoor avoidance (unseen houses)

| training set | survival reactive → wm (±95% CI) | time to first contact | mean flight time | mean distance flown | mean clearance | evaded at least once |
|---|---|---|---|---|---|---|
{chr(10).join(rm_rows)}

**Ceiling check (`hd_oracle.py`).** The same {n_or} starts, flown by a privileged policy that reads the house's distance field directly (no vision, no world model) but is restricted to the wm policy's own action menu: forward, or a 0.5 s lateral veer, yaw fixed. The wm policy's selector only ever picks between the two veers, so this is the best any vision model could do inside that menu. A second variant may also hover when every moving candidate would make contact within 0.3 m.

| policy | survival (±95% CI) | mean flight time | mean distance flown | evaded at least once |
|---|---|---|---|---|
{chr(10).join(or_rows)}

### 4.3 Supplement: offline prediction metrics (held-out frames; \\* = domain used in training)

Domains: pybullet pillars; hyperdrone pillars with random appearance (seen by no model); hyperdrone rooms, houses 0–15 (the training domain of B and C); hyperdrone rooms, houses 16–23 (unseen). Metrics: collision-foresight AUC (0.5 = chance), safe-veer ranking accuracy (0.5 = chance), and the prediction gain of the latent / decoded future frame over "future = now" (0 = no predictive power).

{offline}

### 4.4 Decoded future frames: predicted vs true

The world model predicts the next latent, never pixels. To see what each model imagines, a small decoder (64-d latent → 64×64 RGB) is fitted per model × domain on 80% of that domain's rollouts with the encoder frozen, and the predicted latent for the command actually flown is decoded at +8 and +32 steps (167 / 667 ms) on windows from the other 20%, next to the true future frames (`decode_compare.py`). All three models see the same windows. The number is the decoded-frame gain 1 − MSE(decode(ẑ₊ₖ), x₊ₖ) / MSE(xₜ, x₊ₖ): 0 means no better than copying the current frame, negative means worse (\\* = domain used in training). Caveat: the predictor was trained against the EMA target encoder, whose weights are not saved, so decoded predictions carry that mismatch on top of what a 64-d latent can hold. Expect blurry shapes in the right place, not photographs.

{dec_tab if dec_tab else "_(run decode_compare.py)_"}

![decoded future frames]({dec_fig})

## 5. Process videos

pybullet pillar forest, 4 rows (`forest_demo.py --rows 4`; left: top-down trajectory, right: onboard view):

{gif("forest_pb_ref_wm-mpc", A)}
{gif("forest_pb_rooms_wm-mpc", B)}
{gif("forest_pb_pb_plus_rooms_wm-mpc", Cm)}

hyperdrone indoor (house 17, seed 1017, the same start for all three; left: occupancy map + trajectory, right: the 64×64 frame the policy sees):

{gif("hd_room17_pybullet_ref", A)}
{gif("hd_room17_rooms", B)}
{gif("hd_room17_pb_plus_rooms", Cm)}

## 6. Conclusions

{conclusions}

## 7. Reproduce

{repro}
"""
    else:
        report = f"""# 三种训练集 × 两个避障任务：世界模型的表现对比

[English version / 英文版](REPORT_en.md)

同一套 world model（nanodrone-ai 第 29 课的 V-JEPA 风格潜变量模型 + 碰撞头 + MPC 避障策略，`train_world_model.py --epochs 80`，三个模型完全相同的训练脚本与超参）分别在三种数据集上训练，然后在两个避障任务里闭环测试。所有脚本和每个参数的说明在 [src/README_hyperdrone_dataset.md](src/README_hyperdrone_dataset.md)；更早的探索性实验（两个仿真器各自的柱子画风、随机外观、合并大数据集等）归档在 [src/output/report/REPORT_archive_2026-10-27.md](src/output/report/REPORT_archive_2026-10-27.md)。

## 1. 三种训练集

| 训练集 | 文件 | 规模 | 场景 | 大小 |
|---|---|---|---|---|
{chr(10).join(ds_rows)}

- **{A}**：原课程 `gen_wm_dataset.py` 生成的柱子 course（gym-pybullet-drones，CF2X + PID，64×64、60° 机体相机）。
- **{B}**：`gen_wm_dataset_hyperdrone.py --scene rooms`，ProcTHOR 目录 0–15 号共 16 栋家具房屋里各 16 条 rollout（hyperlux 光追，crazyflie + RAPTOR，同样的 64×64、60° 机体相机）；"危险"标签来自每栋房子在飞行高度的占据栅格 / 距离场（`wm_rooms.py`）。
- **{Cm}**：A 与 B 按 rollout 拼接（`merge_datasets.py`）。

三者共享同一套动作词表（forward / slow / veer_left / veer_right / climb / hover，约 1 s 的 held segment）、同样的 4/8/16/32 步（83–667 ms）碰撞标签和反事实标签格式。

## 2. 训练（各自 held-out rollout 上的离线指标）

| 训练集 | checkpoint | 训练 / 验证窗口 | latent MSE@32（no-op） | AUC@4/8/16/32 | now-AUC | veer-ranking |
|---|---|---|---|---|---|---|
{chr(10).join(tr_rows)}

训练流程（`train_world_model.py`）从数据集到 checkpoint：64×64 的画面经三层卷积编码器和保留左右方位的条带池化得到 64 维潜变量；动作条件预测器以残差形式预测 4/8/16/32 步后的潜变量；碰撞头从预测的潜变量读出"k 步内进入 0.7 m / 0.35 m"的概率，danger-now 头从当前潜变量读出"现在就危险"。预测目标来自编码器的 EMA 副本（JEPA 式，不用像素损失），碰撞头还用数据集里 6 个候选命令的反事实 oracle 监督。整套模型 83k 参数（int8 约 81 KB）。图中上排是在线模型，下排是同一窗口的目标（EMA 潜变量、真实距离标签），每条绿色竖向双箭头就是上面的预测与下面的目标之间的一项损失；画面、潜变量条和概率格子都来自一个 held-out 窗口和模型 A 的实际前向。

![训练流程]({pipe_fig})

## 3. 两个测试任务

**pybullet 柱子避障**（原课程的记分板，`eval_world_model_policy.py` / `speed_sweep.py`，conda env `drone_python`）：{n_c} 条随机 course（{ip_c} 条正前方有柱子，{cl_c} 条直飞安全），每条 360 步、3 m 终点线，策略每 1/12 s 从 64×64 画面决策；到柱心平面距离 < 0.22 m 记为穿越。另有单柱子速度扫描：每档速度 {any_s[1]} 条 course，巡航 0.8 → 1.6 m/s。

**hyperdrone 室内避障**（`hd_closed_loop.py --scene rooms`，conda env `dreamer`）：在训练从未用过的 ProcTHOR {h0}–{h1} 号共 {k_h} 栋房子里随机取 {n_r} 个起点（净空 ≥ 0.9 m、前方 ≥ 2 m 空、随机航向），策略照常"无威胁就前进、有威胁就绕"，直到第一次接触障碍（距离场表面净空 + 0.18 m < 0.22 m，与柱子同一判定）或 7.5 s 到时。ProcTHOR 房子里几乎不存在柱子式的 course（独立障碍 + 3 m 直路 + 两侧可绕），所以室内用生存测试。

两个任务里 reactive 基线（同一 checkpoint 的 danger-now 头 + 特权绕行方向）和 wm 策略共用编码器，所以都随训练集变化；wm 策略才是世界模型的用法。

## 4. 结果

![results]({fig_rel})

### 4.1 pybullet 柱子避障

| 训练集 | 撞击率 reactive → wm（±95% CI） | 最小净空 | wm 比 reactive 提前触发 | 清场 course 误触发 | 速度扫描 wm 撞击率 @0.8/1.0/1.2/1.4/1.6 m/s |
|---|---|---|---|---|---|
{chr(10).join(pb_rows)}

### 4.2 hyperdrone 室内避障（未见过的房子）

| 训练集 | 存活率 reactive → wm（±95% CI） | 首次接触时间 | 平均飞行时间 | 平均飞行距离 | 平均净空 | 至少绕行一次 |
|---|---|---|---|---|---|---|
{chr(10).join(rm_rows)}

**上限检验（`hd_oracle.py`）。** 同样的 {n_or} 个起点，换成一个直接读取房子距离场、不看图像也不用世界模型的特权策略，但限制在 wm 策略同样的动作菜单里：前进，或 0.5 s 的左右平移，偏航固定。wm 策略的选择器只会在两个平移之间挑，所以这就是任何视觉模型在这个菜单里能做到的最好成绩。第二个变体允许在所有移动候选都会在 0.3 m 内接触时悬停。

| 策略 | 存活率（±95% CI） | 平均飞行时间 | 平均飞行距离 | 至少绕行一次 |
|---|---|---|---|---|
{chr(10).join(or_rows)}

### 4.3 补充：离线预测指标（held-out 帧；\\* = 该域参与了训练）

数据域：pybullet 柱子；hyperdrone 柱子，随机外观（没有任何模型见过）；hyperdrone 房间 0–15（B、C 的训练域）；hyperdrone 房间 16–23（未见过）。指标：碰撞预判 AUC（0.5 = 随机）、安全绕行方向排序准确率（0.5 = 随机）、潜变量 / 解码未来帧相对"未来 = 现在"的预测增益（0 = 没有预测能力）。

{offline}

### 4.4 解码的未来帧：预测 vs 真实

世界模型预测的是下一步的潜变量，不是像素。为了看每个模型"想象"了什么，对每个模型 × 数据域各拟合一个小解码器（64 维潜变量 → 64×64 RGB，编码器冻结，用该域 80% 的 rollout），然后在剩下 20% 的窗口上，把实际飞行命令对应的预测潜变量在 +8 和 +32 步（167 / 667 ms）解码出来，与真实的未来帧并排（`decode_compare.py`）。三个模型看的是同一批窗口。表里的数字是解码未来帧增益 1 − MSE(decode(ẑ₊ₖ), x₊ₖ) / MSE(xₜ, x₊ₖ)：0 表示不比复制当前帧好，负数表示更差（\\* = 该域参与了训练）。注意：预测器训练时对齐的是 EMA 目标编码器，其权重没有保存，解码器只能用在线编码器的潜变量拟合，所以解码结果在 64 维潜变量本身的信息上限之外还带着这一点失配，看到的应是位置正确的模糊形状，而不是照片。

{dec_tab if dec_tab else "_(请先运行 decode_compare.py)_"}

![解码的未来帧]({dec_fig})

## 5. 过程视频

pybullet 4 行柱子森林（`forest_demo.py --rows 4`，左俯视轨迹 + 右机载画面）：

{gif("forest_pb_ref_wm-mpc", A)}
{gif("forest_pb_rooms_wm-mpc", B)}
{gif("forest_pb_pb_plus_rooms_wm-mpc", Cm)}

hyperdrone 室内（17 号房子，seed 1017，同一起点；左侧占据栅格地图 + 轨迹，右侧策略看到的 64×64 画面）：

{gif("hd_room17_pybullet_ref", A)}
{gif("hd_room17_rooms", B)}
{gif("hd_room17_pb_plus_rooms", Cm)}

## 6. 结论

{conclusions}

## 7. 复现

{repro}
"""
    out = os.path.join(ROOT, t["report"])
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"wrote {t['report']}; {fig_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lang", choices=["zh", "en", "both"], default="both")
    args = ap.parse_args()
    for lang in (["zh", "en"] if args.lang == "both" else [args.lang]):
        build(lang)


if __name__ == "__main__":
    main()
