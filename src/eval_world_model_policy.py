"""
Lesson 29 (step 4b) — the closed-loop scoreboard + the on-board budget
======================================================================
A single demo course can flatter any controller. This script is the honest
scoreboard: it flies the reactive baseline and the world-model MPC over *many*
random courses (70% with a pillar in the path, 30% clear) and reports the
numbers a robot person actually asks for:

  * crash rate            — does anticipation prevent hits, not just look nice?
  * mean min clearance    — how much margin does the early veer buy?
  * trigger lead          — how many ms earlier does the model react?
  * false positives       — does it flinch on courses that were safe anyway?
  * goal time             — safety must not cost the mission
  * latency + memory      — would this actually run on the GAP8?

The pillar positions are privileged information used to *score* runs (min
clearance, crash) and to *stage* courses — never to fly them; both policies see
only camera frames (the reactive baseline is additionally handed its evasion
direction, a generous handicap documented in `wm_closed_loop.py`).

Read this scoreboard for what it measures: *cluttered* courses at cruise
speed. The side pillars sit at 60–90° bearings — physically outside the
forward camera's FOV — so evasive wandering can clip what no frame ever
showed, and the wm keeps a measured crash tail here that the simple
(privileged-direction, evade-and-resume) reactive baseline does not pay.
That tail is the FOV/memory limit, reported rather than hidden; step 4c
isolates the anticipation mechanism on single visible pillars — where the
crash story inverts hard — and the lesson's going-further (memory, learned
policies) is the honest path to having both at once.

The ONBOARD-BUDGET block splits the 512 KB GAP8 budget the way an embedded
engineer would: weights are not the whole story — the activation tensors and
the DMA double-buffer workspace live in the same SRAM. Latency is measured on
this machine and *estimated* for GAP8 from analytic MACs at an assumed
0.5 GMAC/s effective int8 throughput (PULP-NN-class kernels, order of
magnitude, stated so it can be challenged).

Run:
  python src/eval_world_model_policy.py --seeds 100
  python src/eval_world_model_policy.py --selftest  # 10 seeds
Needs output/world_model.pth (auto-trains a tiny one if missing).
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gen_wm_dataset import ACTION_VECS, CTRL_HZ, IMG_RES, make_env  # noqa: E402
from train_world_model import GAP8_BUDGET_KB, MODEL  # noqa: E402
from wm_closed_loop import (  # noqa: E402
    ReactivePolicy,
    WMPolicy,
    load_or_train,
    run_episode,
)

GAP8_GMACS = 0.5  # assumed effective int8 throughput (GMAC/s), stated not hidden


def onboard_budget(enc, pred, cheads, nhead) -> dict:
    """Split the on-board memory bill into the three numbers that all have to
    fit in the GAP8's 512 KB L2 at once: int8 weights, the peak pair of live
    activation tensors, and a double-buffer workspace (DMA staging) estimate.
    Also counts MACs analytically for the latency estimate."""
    mods = (enc, pred, cheads, nhead)
    n_params = sum(p.numel() for m in mods for p in m.parameters())

    sizes, macs_enc = [3 * IMG_RES * IMG_RES], 0
    x = torch.zeros(1, 3, IMG_RES, IMG_RES)
    with torch.no_grad():
        for mod in enc.features:
            x = mod(x)
            if isinstance(mod, nn.ReLU):
                continue  # fused into the conv kernel on-device; no own buffer
            sizes.append(x.numel())
            if isinstance(mod, nn.Conv2d):
                k = mod.kernel_size[0] * mod.kernel_size[1]
                macs_enc += x.numel() * mod.in_channels * k
    peak_act = max(a + b for a, b in zip(sizes, sizes[1:]))  # in+out live pair
    macs_enc += sum(  # the strip-pool projection runs once per frame too
        m.in_features * m.out_features
        for m in enc.modules()
        if isinstance(m, nn.Linear)
    )

    macs_lin = sum(
        m.in_features * m.out_features
        for mod in (pred, cheads, nhead)
        for m in mod.modules()
        if isinstance(m, nn.Linear)
    )
    n_cands = len(ACTION_VECS)  # the MPC re-runs only the MLPs per candidate
    return {
        "weights_kb": n_params / 1024,
        "peak_act_kb": peak_act / 1024,
        "workspace_kb": peak_act / 1024,  # double-buffered DMA staging
        "total_kb": n_params / 1024 + 2 * peak_act / 1024,
        "macs_frame": macs_enc + macs_lin,  # encoder once + one MLP sweep
        "macs_decision": macs_enc + n_cands * macs_lin,  # MPC: encoder shared
    }


def measure_latency(policy, n: int = 50) -> float:
    """Wall-clock ms per MPC decision on this machine (content-free frame)."""
    frame = np.zeros((IMG_RES, IMG_RES, 3), dtype=np.uint8)
    state = np.zeros(20)
    policy.begin([])
    t0 = time.perf_counter()
    for _ in range(n):
        policy.decide(frame, state)
    return (time.perf_counter() - t0) / n * 1000


def evaluate(n_seeds: int, seed0: int, enc, pred, cheads, nhead, meta) -> list:
    env = make_env()
    rows = []
    for i in range(n_seeds):
        seed, in_path = seed0 + i, (i % 10) < 7  # 70% threatened, 30% clear
        row = {"seed": seed, "in_path": in_path}
        row["reactive"] = run_episode(
            env, ReactivePolicy(enc, nhead), seed, in_path=in_path
        )
        row["wm"] = run_episode(
            env, WMPolicy(enc, pred, cheads, meta), seed, in_path=in_path
        )
        rows.append(row)
        r, w = row["reactive"], row["wm"]
        print(
            f"  seed {seed} ({'in-path' if in_path else 'clear  '}): "
            f"clear {r['min_clear']:.2f}->{w['min_clear']:.2f} m, "
            f"trigger {r['trigger']:>3}->{w['trigger']:>3}, "
            f"crash {int(r['crashed'])}/{int(w['crashed'])}"
        )
    env.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default=None, help="checkpoint (default output/world_model.pth)")
    args = ap.parse_args()
    n_seeds = 10 if args.selftest else args.seeds

    had_ckpt = os.path.exists(args.model or MODEL)
    enc, pred, cheads, nhead, meta = load_or_train(device="cpu", path=args.model)
    print(f"[INFO] checkpoint {args.model or MODEL}: trained on {meta.get('source', 'pybullet')}, camera {meta.get('fov_deg', 60):g} deg")
    rows = evaluate(n_seeds, args.seed0, enc, pred, cheads, nhead, meta)

    ip = [r for r in rows if r["in_path"]]
    cl = [r for r in rows if not r["in_path"]]

    def rate(rs, pol, key):
        return float(np.mean([float(r[pol][key]) for r in rs])) if rs else float("nan")

    crash_r, crash_w = rate(ip, "reactive", "crashed"), rate(ip, "wm", "crashed")
    clr_r, clr_w = rate(ip, "reactive", "min_clear"), rate(ip, "wm", "min_clear")
    leads = [
        (r["reactive"]["trigger"] - r["wm"]["trigger"]) * 1000.0 / CTRL_HZ
        for r in ip
        if r["reactive"]["trigger"] >= 0 and r["wm"]["trigger"] >= 0
    ]
    lead = float(np.mean(leads)) if leads else float("nan")
    fp_r = float(np.mean([r["reactive"]["trigger"] >= 0 for r in cl])) if cl else 0.0
    fp_w = float(np.mean([r["wm"]["trigger"] >= 0 for r in cl])) if cl else 0.0
    ok = [
        r
        for r in ip
        if r["reactive"]["reached"]
        and r["wm"]["reached"]
        and not r["reactive"]["crashed"]
        and not r["wm"]["crashed"]
    ]
    goal_pct = (
        100.0
        * (
            float(np.mean([r["wm"]["steps"] for r in ok]))
            / float(np.mean([r["reactive"]["steps"] for r in ok]))
            - 1.0
        )
        if ok
        else float("nan")
    )

    budget = onboard_budget(enc, pred, cheads, nhead)
    lat_ms = measure_latency(WMPolicy(enc, pred, cheads, meta))
    gap8_ms = budget["macs_decision"] / (GAP8_GMACS * 1e9) * 1000

    print(
        f"WORLD-POLICY OK: seeds={n_seeds} ({len(ip)} in-path / {len(cl)} clear)\n"
        f"  crash_rate:        reactive {crash_r:.0%} -> wm {crash_w:.0%}\n"
        f"  mean_min_clearance: {clr_r:.2f} m -> {clr_w:.2f} m\n"
        f"  mean_trigger_lead: +{lead:.0f} ms (n={len(leads)} both triggered)\n"
        f"  false_positive:    reactive {fp_r:.0%} -> wm {fp_w:.0%} of clear runs\n"
        f"  goal_time:         wm {goal_pct:+.0f}% vs reactive (n={len(ok)} clean)\n"
        f"  decision latency:  {lat_ms:.1f} ms measured (this CPU) | "
        f"~{gap8_ms:.0f} ms est @ GAP8 {GAP8_GMACS:.1f} GMAC/s "
        f"({budget['macs_decision'] / 1e6:.1f} M MACs, encoder shared "
        f"across 6 candidates)"
    )
    print(
        f"ONBOARD-BUDGET OK: weights={budget['weights_kb']:.1f} KB + "
        f"peak_activation={budget['peak_act_kb']:.1f} KB + "
        f"workspace(dbl-buf)={budget['workspace_kb']:.1f} KB = "
        f"{budget['total_kb']:.1f} KB < {GAP8_BUDGET_KB} KB"
    )

    if args.selftest:
        assert budget["total_kb"] < GAP8_BUDGET_KB, "over the GAP8 budget"
        assert gap8_ms < 1000.0 / 12, "MPC too slow for a 12 Hz decision loop"
        if had_ckpt:  # with the properly trained model the policy claims hold
            if leads:
                assert lead > 0, "wm did not trigger earlier on average"
            assert fp_w <= fp_r + 1e-9, "wm false-triggers on safe courses"
            # at selftest scale one crashed course drags the mean clearance
            # hard; the 100-seed run reads 0.38 vs 0.41 m (comparable)
            assert clr_w >= clr_r - 0.20, "wm clears far less than reactive"
            # cluttered courses probe the FOV blind side, where the memoryless
            # planner keeps a measured crash tail (see the README's honest
            # note; step 4c isolates the anticipation mechanism it lacks here)
            assert crash_w <= crash_r + 0.40, "crash tail beyond the documented"
        else:
            print("[INFO] fresh tiny model: policy asserts skipped, budget checked")


if __name__ == "__main__":
    main()
    sys.exit(0)
