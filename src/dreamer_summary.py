"""Collect the nano-Dreamer runs (PLAN_dreamerv3.md, M2-M4) next to the three one-frame models of
REPORT.md into src/output/report/dreamer_results.md: held-out training metrics, the offline
generalization grid, the two closed-loop tests and the open-loop figures.

  python src/dreamer_summary.py
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_report_three import EV, REP, cluttered, rooms, training  # noqa: E402

ROOT = os.path.dirname(HERE)
SETS = [("A", "纯 pybullet（64 rollouts）", "pybullet_ref", "dreamer_pybullet", "pybullet"),
        ("B", "纯 hyperdrone 室内（256 rollouts）", "rooms", "dreamer_rooms", "rooms"),
        ("C", "合并（320 rollouts）", "pb_plus_rooms", "dreamer_pb_plus_rooms", "pb_plus_rooms")]
# extra Dreamer variants per training set: (tag suffix, label)
VARIANTS = {"A": [("_mem", "nano-Dreamer，反事实损失不做视野 mask（8k 步）")],
            "B": [("6k", "nano-Dreamer，6k 步早停")],
            "C": [("6k", "nano-Dreamer，6k 步早停")]}
DOMAINS = [("pybullet", "pybullet 柱子"), ("rand_pillars", "hyperdrone 柱子·随机外观（未见）"), ("rooms_train", "室内房 0–15"), ("rooms_unseen", "室内房 16–23（未见）")]


def dreamer_train(tag):
    p = os.path.join(EV, f"dreamer_{tag}.log")
    if not os.path.exists(p):
        return None
    s = open(p, encoding="utf-8").read()
    m = re.search(r"DREAMER OK \[.*?\]: (\d+) steps, now-AUC=([\d.]+), open-loop AUC@4/8/16/32=([\d./]+), gain@8/@32=(-?[\d.]+)/(-?[\d.]+), "
                  r"dist-MAE post/imag@32=([\d.]+)/([\d.]+) m, veer-ranking=([\d.]+|nan) \(n=(\d+)\), params=([\d.]+)M", s)
    evals = re.findall(r"\[EVAL\s+(\d+)\] now-AUC=([\d.]+), open-loop AUC@4/8/16/32=[\d./]+/([\d.]+), gain@8/@32=(-?[\d.]+)/(-?[\d.]+)", s)
    return dict(m=m.groups() if m else None, evals=evals)


def fmt_pct(v):
    return "—" if v is None else f"{v}%"


def main():
    old_grid = json.load(open(os.path.join(REP, "three_way.json"))) if os.path.exists(os.path.join(REP, "three_way.json")) else {}
    new_grid = {}
    for fn in ("dreamer_three.json", "dreamer_variants.json"):
        if os.path.exists(os.path.join(REP, fn)):
            new_grid.update(json.load(open(os.path.join(REP, fn))))
    L = ["# nano-Dreamer（RSSM 世界模型）首轮训练结果", "",
         "分支 `dreamerv3-world-model`，方案见 [PLAN_dreamerv3.md](../../../PLAN_dreamerv3.md)。同一套三个训练集（A/B/C）、同一套测试，"
         "只把世界模型从单帧 JEPA 预测器换成 DreamerV3 风格的 RSSM（`train_dreamer.py`，1.26M 参数，12 Hz 建模，2 万步，三个模型同一超参）。"
         "闭环策略 `DreamerMPCPolicy` 的触发和代价与 `WMPolicy` 完全相同，只是候选命令的后果由想象 8 步得到，且信念状态跨决策保留。", "",
         "## 1. 训练集内 held-out 指标", "",
         "旧模型的 AUC 是单帧 posterior 直接读碰撞头；Dreamer 的 AUC 是 **open-loop**：观测 667 ms 后在飞过的命令下纯想象 8 步（= +32 帧）再读头，设定更难。解码增益 = 1 − MSE(预测帧, 真帧)/MSE(当前帧, 真帧)。", "",
         "| 训练集 | 模型 | AUC@32 | 解码增益 @8 / @32 | 想象 @32 的净空 MAE | veer ranking |", "|---|---|---|---|---|---|"]
    for key, label, old, new, tag in SETS:
        t = training(old)
        if t:
            L.append(f"| {key} {label} | JEPA 单帧（83k） | {t[6].split('/')[-1]} | （事后拟合解码器，见报告 4.4） | — | {t[8]} (n={t[9]}) |")
        for suffix, vlabel in [("", None)] + VARIANTS.get(key, []):
            d = dreamer_train(tag + suffix)
            if d and d["m"]:
                m = d["m"]
                L.append(f"| {key} {label} | {vlabel or f'nano-Dreamer（{m[9]}M，20k 步）'} | **{m[2].split('/')[-1]}** | {float(m[3]):+.2f} / {float(m[4]):+.2f} | {m[6]} m | {m[7]} (n={m[8]}) |")
    L += ["", "训练过程中每 4k 步的 held-out 评估（AUC@32 / 增益@32）："]
    for key, label, old, new, tag in SETS:
        d = dreamer_train(tag)
        if d and d["evals"]:
            L.append(f"- {key}: " + "，".join(f"{int(s)//1000}k: {a}/{float(g32):+.2f}" for s, _, a, _, g32 in d["evals"]))
    L += ["", "## 2. 离线泛化网格（held-out 帧，\\* = 训练时见过的数据域）", ""]
    for title, pick, fmt in (("碰撞预判 AUC@32（0.5 = 随机）", lambda m: m["auc"][-1], "{:.2f}"),
                             ("解码未来帧增益 @32（0 = 不比复制当前帧好）", lambda m: m["frame_gain"], "{:+.2f}"),
                             ("安全绕行方向排序准确率（0.5 = 随机）", lambda m: m["cf_rank"], "{:.2f}")):
        L += [f"**{title}**", "", "| 训练集 | 模型 | " + " | ".join(dl for _, dl in DOMAINS) + " |", "|---|---|" + "---|" * len(DOMAINS)]
        for key, label, old, new, tag in SETS:
            for name, g, mk in [(old, old_grid, "JEPA 单帧"), (new, new_grid, "nano-Dreamer")] + [(new + sfx, new_grid, vl) for sfx, vl in VARIANTS.get(key, [])]:
                if name in g:
                    cells = [fmt.format(pick(g[name][dk])) + ("\\*" if g[name][dk].get("seen") else "") if dk in g[name] else "—" for dk, _ in DOMAINS]
                    L.append(f"| {key} | {mk} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## 3. 闭环测试", "",
          "pybullet 柱子避障：300 条随机 course（210 条有柱子），记 wm 策略的穿越率；室内续航：ProcTHOR 房 16–23 共 300 个起点，1.0 m/s 飞 7.5 s，记无接触的生存率。reactive 基线用同一模型的 danger-now（Dreamer：posterior 上的 0.7 m 头）加特权方向。", "",
          "| 训练集 | 模型 | pybullet 穿越率 reactive → wm | 最小净空 | 误触发（无柱 course） | 室内生存率 reactive → wm | 室内平均飞行时间 | 室内生存率，菜单含 hover |", "|---|---|---|---|---|---|---|---|"]
    for key, label, old, new, tag in SETS:
        rows_ = [(old, "JEPA 单帧"), (new + "_v0", "nano-Dreamer，首版读数（单次采样想象，含当前状态）"), (new, "nano-Dreamer（8 条想象平均，只看未来）")]
        rows_ += [(new + suffix, vlabel) for suffix, vlabel in VARIANTS.get(key, [])]
        for name, mk in rows_:
            c, r = cluttered(name), rooms(name)
            if c is None and r is None:
                continue
            cl = f"{c['crash_r']}% → **{c['crash_w']}%**" if c else "未跑"
            mc = f"{c['clear_r']:.2f} → {c['clear_w']:.2f} m" if c else "—"
            fp = f"{c['fp_r']}% → {c['fp_w']}%" if c else "—"
            rm = f"{r['surv_r']}% → **{r['surv_w']}%**" if r else "未跑"
            ft = f"{r['ft_r']:.1f} → {r['ft_w']:.1f} s" if r else "—"
            rs = rooms(name + "_stop")
            st = f"**{rs['surv_w']}%**（飞行 {rs['ft_w']:.1f} s，{rs['dist_w']:.2f} m）" if rs else "未跑"
            L.append(f"| {key} | {mk} | {cl} | {mc} | {fp} | {rm} | {ft} | {st} |")
    L += ["", "参考：同菜单的特权 oracle 室内生存率 32%，允许悬停 89%（报告 4.2 节）。"
          "「菜单含 hover」= `--allow-stop`：两个绕行方向在 ~170 ms 内的 P(< 0.35 m) 都超过 0.5 时悬停 0.5 s（对应 oracle 的 +hover 规则），新旧模型同一规则。", "",
          "## 4. open-loop 想象的解码帧（held-out 窗口）", "",
          "上排真帧，下排：第 1 列是 posterior 重建，其后 8 列是观测 32 帧后在飞过的命令下想象 +4…+32 帧的解码结果；蓝字是想象状态读出的净空距离和 P(< 0.7 m)。", ""]
    for key, label, old, new, tag in SETS:
        p = os.path.join(REP, f"dreamer_openloop_{tag}.png")
        if os.path.exists(p):
            L += [f"**{key} {label}**", "", f"![open loop {tag}]({os.path.basename(p)})", ""]
    L += ["## 5. 复现", "", "```bash",
          "python src/train_dreamer.py --data src/output/wm_dataset.npz --out src/output/world_model_dreamer_pybullet.pth --tag pybullet --steps 20000",
          "python src/train_dreamer.py --data src/output/wm_dataset_rooms_train60.npz --out src/output/world_model_dreamer_rooms.pth --tag rooms --steps 20000",
          "python src/train_dreamer.py --data src/output/wm_dataset_pb_plus_rooms.npz --out src/output/world_model_dreamer_pb_plus_rooms.pth --tag pb_plus_rooms --steps 20000",
          "python src/eval_offline.py --models dreamer_pybullet dreamer_rooms dreamer_pb_plus_rooms --domains pybullet rand_pillars rooms_train rooms_unseen --out dreamer_three",
          "python src/eval_world_model_policy.py --seeds 300 --model src/output/world_model_dreamer_pybullet.pth   # conda env drone_python, per model",
          "python src/hd_closed_loop.py --scene rooms --houses 8 --house-first 16 --seeds 300 --sweep-seeds 0 --model src/output/world_model_dreamer_pybullet.pth",
          "python src/dreamer_summary.py", "```", ""]
    out = os.path.join(REP, "dreamer_results.md")
    open(out, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L[:60]))
    print("SUMMARY OK ->", out)


if __name__ == "__main__":
    main()
