"""
Lesson 29 (step 4c) — raise the speed until reaction breaks
===========================================================
Step 4b showed that at a gentle 0.8 m/s cruise, even the reactive baseline
stays crash-free — anticipation only buys *margin* there. This step turns the
margin into the metric that matters: **crash rate versus speed**.

The physics of why the curves must split: the reactive trigger fires at a
fixed visual *distance* (the danger-now head learned "planar < 0.7 m"), so as
speed rises, the time between trigger and impact shrinks — and with bounded
lateral authority, at some speed the evasion no longer fits in the time left.
The world model triggers at a fixed *time* (~0.7 s of look-ahead, because the
danger heads are conditioned on the commanded speed), which is a
speed-proportional distance. Reaction is a distance budget; anticipation is a
time budget. Speed spends the first and not the second.

Both policies fly the same threatened courses at each cruise speed (the same
seeds across speeds), with the whole command set scaled together — the same
model, trained across the 0.6–1.6 m/s envelope, powers every run.

Honest scope: these are *single-pillar* courses on purpose. The threat sits
where both policies can see it, so the sweep isolates the mechanism it claims
to measure. Step 4b's cluttered courses probe something else — the FOV blind
side — and there the hand-crafted planner still pays a measurable crash tail
(side pillars 60–90° off-axis are physically invisible to a fixed-yaw,
memoryless, forward camera; the same tail also grazes a few single-pillar
runs at walking pace). Those numbers are reported, not hidden, in the README —
and the roadmap's fix is learning the policy (Lesson 19) and adding memory,
not more cost-function tuning.

Run:
  python src/speed_sweep.py                # saves the plot
  python src/speed_sweep.py --seeds 30
  python src/speed_sweep.py --selftest     # 2 speeds, asserts
Needs output/world_model.pth (auto-trains a tiny one if missing).
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gen_wm_dataset import make_env  # noqa: E402
from wm_closed_loop import (  # noqa: E402
    ReactivePolicy,
    WMPolicy,
    load_or_train,
    run_episode,
)

SPEEDS = (1.0, 1.25, 1.5, 1.75, 2.0)  # x 0.8 m/s base -> 0.8..1.6 m/s cruise
BASE_V = 0.8  # forward command at speed factor 1.0 (m/s)
OUT = os.path.join(HERE, "output", "speed_sweep.png")


def _policies(enc, pred, cheads, nhead, meta, learned: bool = True):
    """Reactive + hand MPC always; learned variants join automatically when a
    trained .zip exists (step 6). Each factory takes the cruise-speed factor.
    `learned=False` keeps only the two hand policies (the PPO policies were
    trained on top of the pybullet world model's latents, so they only make
    sense with that checkpoint)."""
    mk = {
        "reactive": lambda s: ReactivePolicy(enc, nhead),
        "wm": lambda s: WMPolicy(enc, pred, cheads, meta, speed=s),
    }
    if not learned:
        return mk
    from learn_policy import LearnedPolicy, _load_policy, zip_path

    for name, rec, edge, curr in (
        ("learned", False, False, False),
        ("learned-rnn", True, False, False),
        ("learned-rnn-edge", True, True, False),
        ("learned-rnn-curr", True, False, True),
    ):
        path = zip_path(recurrent=rec, edge=edge, curr=curr)
        if os.path.exists(path):
            model = _load_policy(path)
            mk[name] = lambda s, m=model: LearnedPolicy(
                m, enc, pred, cheads, meta, speed=s
            )
    return mk


def sweep(speeds, n_seeds: int, seed0: int, path: str = None, learned: bool = True) -> list:
    """Fly `n_seeds` single-pillar courses per speed with every available
    policy. The same seeds repeat across speeds, so each speed step changes
    exactly one thing — and `solo` courses put the one threat where every
    policy can see it, so the sweep measures the anticipation mechanism, not
    the FOV limit."""
    enc, pred, cheads, nhead, meta = load_or_train(device="cpu", path=path)
    mk = _policies(enc, pred, cheads, nhead, meta, learned=learned)
    env = make_env()
    rows = []
    for s in speeds:
        row = {"speed": s, "v": s * BASE_V}
        report = []
        for name, factory in mk.items():
            crash, clear = 0, []
            for i in range(n_seeds):
                run = run_episode(
                    env, factory(s), seed0 + i, in_path=True, speed=s, solo=True
                )
                crash += int(run["crashed"])
                clear.append(run["min_clear"])
            row[f"crash_{name}"] = crash / n_seeds
            row[f"clear_{name}"] = float(np.mean(clear))
            report.append(f"{name} {row[f'crash_{name}']:.0%}")
        rows.append(row)
        print(f"  v={row['v']:.1f} m/s: crash " + " / ".join(report))
    env.close()
    rows[0]["_names"] = list(mk)
    return rows


COLORS = {
    "reactive": "tab:orange",
    "wm": "tab:green",
    "learned": "tab:blue",
    "learned-rnn": "tab:purple",
    "learned-rnn-edge": "tab:red",
    "learned-rnn-curr": "tab:brown",
}
LABELS = {
    "reactive": "reactive",
    "wm": "wm (hand MPC)",
    "learned": "learned (stacked memory)",
    "learned-rnn": "learned (LSTM memory)",
    "learned-rnn-edge": "learned (LSTM, edge-biased speeds)",
    "learned-rnn-curr": "learned (LSTM, mixed-diet curriculum)",
}


def _save_plot(rows, out: str = OUT) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = [r["v"] for r in rows]
    names = rows[0]["_names"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    for name in names:
        ax1.plot(
            v,
            [100 * r[f"crash_{name}"] for r in rows],
            "o-",
            color=COLORS.get(name, "tab:gray"),
            label=LABELS.get(name, name),
        )
        ax2.plot(
            v,
            [r[f"clear_{name}"] for r in rows],
            "o-",
            color=COLORS.get(name, "tab:gray"),
            label=LABELS.get(name, name),
        )
    ax1.set_xlabel("cruise speed (m/s)")
    ax1.set_ylabel("crash rate (%)")
    ax1.set_title("Reaction is a distance budget")
    ax1.legend(fontsize=7)
    ax2.set_xlabel("cruise speed (m/s)")
    ax2.set_ylabel("mean min clearance (m)")
    ax2.set_title("Anticipation is a time budget")
    ax2.legend(fontsize=7)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"[INFO] saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=3000)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default=None, help="checkpoint (default output/world_model.pth)")
    ap.add_argument("--no-learned", action="store_true", help="only the reactive and wm hand policies")
    ap.add_argument("--plot", default=OUT, help="where to save the figure")
    args = ap.parse_args()
    speeds, n = ((1.0, 2.0), 8) if args.selftest else (SPEEDS, args.seeds)

    rows = sweep(speeds, n, args.seed0, path=args.model, learned=not args.no_learned)
    _save_plot(rows, args.plot)
    lo, hi = rows[0], rows[-1]
    names = rows[0]["_names"]
    lo_s = "/".join(f"{lo[f'crash_{k}']:.0%}" for k in names)
    hi_s = "/".join(f"{hi[f'crash_{k}']:.0%}" for k in names)
    print(
        f"SPEED-SWEEP OK: {n} single-pillar courses/speed — crash "
        f"({'/'.join(names)}) at {lo['v']:.1f} m/s = {lo_s}; at "
        f"{hi['v']:.1f} m/s = {hi_s} — reaction pays a distance, "
        f"anticipation pays time"
    )
    if args.selftest:
        assert (
            hi["crash_wm"] < hi["crash_reactive"] - 0.25
        ), "the story did not hold: wm does not clearly out-survive reactive at speed"
        assert hi["clear_wm"] > hi["clear_reactive"], "no clearance edge at speed"
        # at walking pace both should be near-clean; the wm is allowed the
        # small planner tail the docstring's honest note documents
        assert (
            lo["crash_wm"] <= lo["crash_reactive"] + 2.0 / n + 1e-9
        ), "wm crashes at base speed beyond the documented planner tail"


if __name__ == "__main__":
    main()
    sys.exit(0)
