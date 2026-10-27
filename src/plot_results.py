"""
Charts for the report: closed-loop results per training set and test environment
================================================================================
Reads the evaluation logs (src/output/evals/*.log, written by eval_world_model_policy.py,
speed_sweep.py and hd_closed_loop.py) and draws two figures:

  results_success.png   small multiples, one panel per test environment: success rate (courses
                        flown without a crash, wm policy, 100 cluttered courses) per training set,
                        and below it the false-trigger rate on the 30 clear courses
  results_speed.png     small multiples, one panel per training set: wm crash rate vs cruise
                        speed on 30 single-pillar courses, one line per test environment

  python src/plot_results.py            # writes src/output/report/results_*.png
"""

import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EVALS = os.path.join(HERE, "output", "evals")
OUT = os.path.join(HERE, "output", "report")

# the reference palette (light surface); validated with the dataviz skill's script
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, fixed order = test environments

CHECKPOINTS = [  # (log name, label, training data)
    ("pybullet_shipped", "pybullet 64\n(shipped weights)", "pybullet"),
    ("pybullet_ref", "pybullet 64\n(retrained)", "pybullet"),
    ("mixed_big", "hyperdrone 256\nhd look", "hyperdrone"),
    ("mixed_big_pbstyle", "hyperdrone 256\npb look", "hyperdrone"),
    ("pb_plus_big", "pybullet 64 +\nhyperdrone hd look", "merged"),
    ("pb_plus_big_pbstyle", "pybullet 64 +\nhyperdrone pb look", "merged"),
]
ENVS = [("", "pybullet"), ("@hdpb", "hyperdrone, pybullet look"), ("@hdhd", "hyperdrone, hyperdrone look")]
ENVS_ALL = ENVS + [("@hdrand", "hyperdrone, hd look,\nrandom appearance")]


def wilson_half(p, n, z=1.96):
    """Half-width (percentage points) of the 95% Wilson interval for rate p on n trials."""
    if not n:
        return 0.0
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    return 100 * z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)


def read(name, kind):
    path = os.path.join(EVALS, f"{name}_{kind}.log")
    return open(path, encoding="utf-8").read() if os.path.exists(path) else ""


def cluttered(name):
    log = read(name, "cluttered")
    crash = re.search(r"crash_rate:\s+reactive (\d+)% -> wm (\d+)%", log)
    fp = re.search(r"false_positive:\s+reactive (\d+)% -> wm (\d+)%", log)
    n = re.search(r"seeds=(\d+) \((\d+) in-path / (\d+) clear\)", log)
    return (int(crash.group(2)) if crash else None, int(fp.group(2)) if fp else None, int(crash.group(1)) if crash else None,
            int(n.group(2)) if n else None, int(n.group(3)) if n else None)


def sweep(name):
    rows = re.findall(r"v=([\d.]+) m/s: crash reactive (\d+)% / wm (\d+)%", read(name, "sweep"))
    return [(float(v), int(w), int(r)) for v, r, w in rows]


def rounded_bar(ax, y, width, height, color, xmax=112.0):
    """A horizontal bar: square at the baseline, 4px-rounded data end, <= 24px thick."""
    if width <= 0:
        return
    r = min(xmax * 0.011, width / 2)  # rounding in data units (~4px at this figure scale)
    ax.add_patch(FancyBboxPatch((0, y - height / 2), width, height, boxstyle=f"round,pad=0,rounding_size={r}", linewidth=0, facecolor=color, mutation_aspect=0.1))
    ax.add_patch(Rectangle((0, y - height / 2), min(r, width), height, linewidth=0, facecolor=color))  # square the baseline end


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def figure_success():
    envs = [e for e in ENVS_ALL if any(cluttered(c[0] + e[0])[0] is not None for c in CHECKPOINTS)]
    fig, axes = plt.subplots(2, len(envs), figsize=(2.9 * len(envs) + 2.8, 6.2), facecolor=SURFACE)
    labels = [c[1] for c in CHECKPOINTS]
    ys = list(range(len(CHECKPOINTS)))[::-1]
    n_ip = n_cl = None
    for col, (suffix, env) in enumerate(envs):
        vals = [cluttered(name + suffix) for name, _, _ in CHECKPOINTS]
        n_ip = n_ip or next((v[3] for v in vals if v[3]), None)
        n_cl = n_cl or next((v[4] for v in vals if v[4]), None)
        for row, (title, pick, nkey) in enumerate((
            ("Success rate (no crash on the in-path courses)", lambda v: None if v[0] is None else 100 - v[0], 3),
            ("False triggers on the clear courses (lower is better)", lambda v: v[1], 4),
        )):
            ax = axes[row, col]
            style(ax)
            for y, v in zip(ys, vals):
                x = pick(v)
                if x is None:
                    ax.text(1, y, "–", va="center", ha="left", fontsize=8, color=MUTED)
                    continue
                rounded_bar(ax, y, x, 0.52, SERIES[0])
                half = wilson_half(x / 100, v[nkey]) if v[nkey] else 0
                if half:
                    ax.plot([max(x - half, 0), min(x + half, 100)], [y, y], color=INK2, linewidth=1, solid_capstyle="butt", zorder=4)
                    ax.plot([max(x - half, 0)] * 2, [y - 0.12, y + 0.12], color=INK2, linewidth=1, zorder=4)
                    ax.plot([min(x + half, 100)] * 2, [y - 0.12, y + 0.12], color=INK2, linewidth=1, zorder=4)
                ax.text(min(x + half, 100) + 1.5, y, f"{x:.0f}%", va="center", ha="left", fontsize=8.5, color=INK)
            ax.set_xlim(0, 112)
            ax.set_ylim(-0.7, len(CHECKPOINTS) - 0.3)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_xticklabels(["0", "25", "50", "75", "100%"])
            if col == 0:
                ax.set_yticks(ys)
                ax.set_yticklabels(labels, fontsize=8, color=INK2)
            else:
                ax.set_yticks([])
            if row == 0:
                ax.set_title(f"tested in {env}", fontsize=10, color=INK, loc="left", pad=8)
            elif col == 0:
                ax.set_title(title, fontsize=9, color=INK2, loc="left", pad=8)
    fig.text(0.01, 0.985, f"Where does each world model fly safely? wm policy, the same {(n_ip or 0) + (n_cl or 0)} courses in every environment (rows = training data)",
             fontsize=11, color=INK, va="top")
    fig.text(0.01, 0.955, f"Top: success rate on the {n_ip} in-path courses with 95% Wilson intervals. Bottom: false-trigger rate on the {n_cl} clear courses (lower is better). Reactive baselines are in the tables.",
             fontsize=8.5, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.5)
    path = os.path.join(OUT, "results_success.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def figure_speed():
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.0), facecolor=SURFACE, sharex=True, sharey=True)
    for k, (name, label, _) in enumerate(CHECKPOINTS):
        ax = axes[k // 3, k % 3]
        style(ax)
        ax.yaxis.grid(True, color=GRID, linewidth=1)
        ax.xaxis.grid(False)
        ends = []
        for j, (suffix, env) in enumerate(ENVS):
            rows = sweep(name + suffix)
            if not rows:
                continue
            xs, ws = [r[0] for r in rows], [r[1] for r in rows]
            ax.plot(xs, ws, color=SERIES[j], linewidth=2, solid_capstyle="round", solid_joinstyle="round", label=env, zorder=3)
            ax.scatter(xs, ws, s=36, color=SERIES[j], edgecolor=SURFACE, linewidth=2, zorder=4)
            ends.append([xs[-1], ws[-1], ws[-1]])
        # end labels: nudge apart where they would overlap (>= 8 points of separation)
        ends.sort(key=lambda e: e[1])
        for a, b in zip(ends, ends[1:]):
            if b[2] - a[2] < 8:
                b[2] = a[2] + 8
        for x, w, y in ends:
            ax.text(x + 0.03, min(y, 100), f"{w:.0f}%", fontsize=7.5, color=INK2, va="center")
        ax.set_title(label.replace("\n", " "), fontsize=9.5, color=INK, loc="left", pad=6)
        ax.set_xlim(0.72, 1.78)
        ax.set_ylim(-4, 104)
        ax.set_xticks([0.8, 1.0, 1.2, 1.4, 1.6])
        ax.set_xticklabels(["0.8", "1.0", "1.2", "1.4", "1.6 m/s"])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(["0", "25", "50", "75", "100%"])
        ax.tick_params(labelleft=(k % 3 == 0), labelbottom=(k // 3 == 1))
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, [f"tested in {n}" for n in names], loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=3, frameon=False, fontsize=8.5, labelcolor=INK2)
    n_sw = next((len(sweep(c[0])) and re.search(r"SPEED-SWEEP OK: (\d+)", read(c[0], "sweep")) for c in CHECKPOINTS if read(c[0], "sweep")), None)
    n_sw = n_sw.group(1) if n_sw else "?"
    fig.text(0.01, 0.985, f"Crash rate vs cruise speed: wm policy on {n_sw} single-pillar courses per speed (lower is better)", fontsize=11, color=INK, va="top")
    fig.text(0.01, 0.955, "One panel per training set; each line is one test environment. Crashes stay low at speed only where the pictures match the training look.",
             fontsize=8.5, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    path = os.path.join(OUT, "results_speed.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def rooms_n(name):
    log = read(name + "@hdroom", "rooms")
    m = re.search(r"starts=(\d+) \[hyperdrone/rooms (\d+)-(\d+)", log)
    return (int(m.group(1)), int(m.group(3)) - int(m.group(2)) + 1) if m else (None, None)


def rooms(name):
    """(survival reactive, survival wm, ttc reactive, ttc wm, dist reactive, dist wm) from a ROOMS OK block."""
    log = read(name + "@hdroom", "rooms")
    surv = re.search(r"survival \(no contact in 7.5 s\): reactive (\d+)% -> wm (\d+)%", log)
    ttc = re.search(r"time to first contact \(contacted runs\): reactive ([\d.]+) s -> wm ([\d.]+) s", log)
    dist = re.search(r"mean distance flown: reactive ([\d.]+) m -> wm ([\d.]+) m", log)
    if not surv:
        return None
    return (int(surv.group(1)), int(surv.group(2)), float(ttc.group(1)) if ttc else None, float(ttc.group(2)) if ttc else None,
            float(dist.group(1)) if dist else None, float(dist.group(2)) if dist else None)


GRAY = "#b5b4ad"  # de-emphasis for the reactive baseline


def figure_rooms():
    data = [(c[1], rooms(c[0])) for c in CHECKPOINTS]
    if not any(d for _, d in data):
        return None
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), facecolor=SURFACE)
    panels = (
        ("Survival rate (no contact within 7.5 s)", 0, 1, "%", 112, [0, 25, 50, 75, 100]),
        ("Distance flown before contact / timeout", 4, 5, " m", 7.0, [0, 2, 4, 6]),
        ("Time to first contact, contacted runs", 2, 3, " s", 8.4, [0, 2, 4, 6, 8]),
    )
    ys = list(range(len(CHECKPOINTS)))[::-1]
    for ax, (title, ir, iw, unit, xmax, ticks) in zip(axes, panels):
        style(ax)
        for y, (label, d) in zip(ys, data):
            if d is None or d[iw] is None:
                ax.text(0.02 * xmax, y, "–", va="center", fontsize=8, color=MUTED)
                continue
            r, w = d[ir], d[iw]
            rounded_bar(ax, y + 0.17, r, 0.3, GRAY, xmax)
            rounded_bar(ax, y - 0.17, w, 0.3, SERIES[0], xmax)
            shift = 0.0
            if unit == "%":
                n_st = rooms_n(CHECKPOINTS[len(CHECKPOINTS) - 1 - y][0])[0]
                half = wilson_half(w / 100, n_st) if n_st else 0
                if half:
                    ax.plot([max(w - half, 0), w + half], [y - 0.17, y - 0.17], color=INK2, linewidth=1, zorder=4)
                    ax.plot([max(w - half, 0)] * 2, [y - 0.26, y - 0.08], color=INK2, linewidth=1, zorder=4)
                    ax.plot([w + half] * 2, [y - 0.26, y - 0.08], color=INK2, linewidth=1, zorder=4)
                    shift = half
            ax.text(w + shift + 0.012 * xmax, y - 0.17, f"{w:.0f}{unit}" if unit == "%" else f"{w:.1f}{unit}", va="center", ha="left", fontsize=8, color=INK)
            ax.text(r + 0.012 * xmax, y + 0.17, f"{r:.0f}{unit}" if unit == "%" else f"{r:.1f}{unit}", va="center", ha="left", fontsize=7, color=MUTED)
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.7, len(CHECKPOINTS) - 0.3)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t}{unit.strip()}" if t == ticks[-1] else str(t) for t in ticks])
        ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=8)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([c[1] for c in CHECKPOINTS], fontsize=8, color=INK2)
    for ax in axes[1:]:
        ax.set_yticks([])
    from matplotlib.patches import Patch

    fig.legend([Patch(color=SERIES[0]), Patch(color=GRAY)], ["wm policy (latent MPC)", "reactive baseline (danger-now head)"],
               loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2, frameon=False, fontsize=8.5, labelcolor=INK2)
    n_st, n_h = next((rooms_n(c[0]) for c in CHECKPOINTS if rooms_n(c[0])[0]), (None, None))
    fig.text(0.01, 0.975, f"Indoor clutter endurance: {n_st} random starts in {n_h} ProcTHOR houses never used for training (hyperdrone)", fontsize=11, color=INK, va="top")
    fig.text(0.01, 0.925, "Fly forward unless the policy evades, until the first contact or 7.5 s. Higher is better in every panel.", fontsize=8.5, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    path = os.path.join(OUT, "results_rooms.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams["font.family"] = "sans-serif"
    for p in (figure_success(), figure_speed(), figure_rooms()):
        if p:
            print("saved", p)


if __name__ == "__main__":
    main()
    sys.exit(0)
