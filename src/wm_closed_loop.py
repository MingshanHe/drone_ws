"""
Lesson 29 (step 4) — close the loop: the world model flies from vision alone
============================================================================
Step 3 (`proactive_avoid.py`) isolated the *timing* benefit of anticipation
with the simulator's privileged future geometry standing in for the model.
This step removes the crutch. Here the danger signal in the control loop comes
from the **camera alone**: frame -> encoder -> predictor -> collision heads.
The pillar positions are used to *score* the flight afterwards (min clearance,
crashes) — never to fly it.

Two controllers fly the same course (same seed, same pillars):

  * **reactive** — the danger-*now* head (same encoder, no look-ahead) says
    "too close!", and the drone evades. Honest note: we hand this baseline the
    evasion *direction* from privileged pillar positions — a deliberately
    generous opponent. If anticipation still wins on clearance while choosing
    its own direction from vision, the win is real.
  * **wm (proactive)** — a tiny **latent MPC**: an anticipatory *trigger* plus
    a safest-veer *chooser*, at ~12 Hz (an on-board-honest decision rate).
    Each decision encodes the frame once, then asks the predictor "and if I
    held *this* command?" for every candidate — the expensive encoder is
    shared, so on a GAP8 the whole deliberation is nearly free. The trigger is
    *relative*: evade when going straight is predicted meaningfully more
    warn-dangerous than the better veer (an absolute threshold inherits the
    heads' course-dependent probability floor and false-triggers — measured),
    with an absolute near-term *crit* backstop for the moment everything
    saturates together. On trigger it commits ~0.5 s to the veer with the
    cheapest predicted future:

      cost = 6*(0.25*warn + 0.75*crit) + heading + 1.5*|y_after| - 1.2*progress

    The *critical* ring (0.35 m) keeps that choice sighted inside the warn
    ring (where every moving action correctly "warns"), and the |y_after| term
    is a corridor-centering prior from the drone's own odometry — because the
    camera cannot see 60° to the side, the planner prefers not to wander
    there. Anticipation means this cascade starts ~500 ms before the reactive
    trigger at cruise — and at speed it is the difference between dodging and
    hitting (step 4c).

Run:
  python src/wm_closed_loop.py             # saves the plot
  python src/wm_closed_loop.py --gui       # watch it fly
  python src/wm_closed_loop.py --selftest  # asserts (local)
Needs output/world_model.pth (auto-trains a tiny one if missing).
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gen_wm_dataset import (  # noqa: E402
    ACTION_NAMES,
    ACTION_VECS,
    COLLISION_R,
    CTRL_HZ,
    DANGER_R,
    FORWARD,
    VelCommander,
    gen,
    grab_frame,
    make_ctrl,
    make_env,
    nearest_planar,
    spawn_pillars,
)
from train_world_model import MODEL, load_model, train  # noqa: E402

GOAL_X = 3.0  # finish line (m)
TMAX = 360  # step budget (7.5 s @ 48 Hz)
DECIDE_EVERY = 4  # decide at 12 Hz — an honest on-board rate; PID stays 48 Hz
THETA_NOW = 0.5  # reactive trigger on P(too close now)
EVADE_HOLD = 24  # reactive holds its evasion ~0.5 s before re-checking
W_DANGER, W_HEAD, W_SWITCH, W_PROG = 6.0, 1.0, 0.5, 1.2  # MPC cost weights
W_CENTER = 1.5  # corridor-centering prior. The camera cannot see a pillar
# 60 deg to the side (the honest FOV limit), so all else near-equal the
# planner prefers motion that ends nearer the corridor centre — computed from
# the drone's OWN odometry (its y), never from pillar positions. Measured
# without it: blind-side clips on courses whose side pillars sat exactly
# where the evasion wandered.
# Urgency weights over the horizons (83/167/333/667 ms). Imminent danger
# counts fully; distant danger is an early warning. A flat max-over-horizons
# would treat "will cross 0.7 m within 667 ms" as a veto — but passing a
# pillar at a safe 0.5 m *is* such a crossing, so far horizons must warn,
# not forbid.
W_HORIZON = (1.0, 0.6, 0.35, 0.2)
MARGIN_WM = 0.4  # evade when going straight is predicted this much more
# warn-dangerous (urgency-weighted) than the better veer — a *relative*
# trigger, immune to the course-dependent probability floor that defeats any
# absolute threshold (measured: fixed thresholds false-trigger on clear
# courses and under-trigger on cluttered ones)
SCENARIO_SEED = 11  # the demo course (step-4 eval sweeps many seeds)
OUT = os.path.join(HERE, "output", "wm_closed_loop.png")


def _frame_tensor(frame: np.ndarray) -> torch.Tensor:
    x = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0
    return x.unsqueeze(0)  # (1, 3, 64, 64)


class WMPolicy:
    """Latent MPC from vision alone: an anticipatory trigger plus a
    safest-veer chooser. Never sees the pillar positions.

    The trigger is *relative*, not a fixed threshold: evade when continuing
    forward is predicted MARGIN_WM more warn-dangerous than the better veer.
    Relative, because the heads carry a course-dependent probability floor
    (open-space haze, side-pillar clutter) that shifts every candidate
    together — an absolute threshold tuned on one course false-triggers on
    another (measured: 100% false positives on clear courses), while the
    *difference* only opens when something actually blocks the way ahead.
    On trigger, commit ~0.5 s to the veer with the cheapest predicted future
    (crit-weighted — inside the warn ring only the bad actions are *about to
    hit*), with a corridor-centering prior from the drone's own odometry."""

    def __init__(self, enc, pred, cheads, meta, speed: float = 1.0):
        self.enc, self.pred, self.cheads = enc, pred, cheads
        names = list(meta["action_names"])
        vecs = float(speed) * np.array(meta["action_vecs"], dtype=np.float32)
        # The planner's menu. The corridor task is planar and `climb` games the
        # planar danger label (a slower planar approach scores "safer" without
        # ever engaging the visual task — traced: the MPC climbed over the
        # course), so the planner leaves it off the menu; the model itself
        # still knows the full six-command vocabulary.
        self.ids = [i for i, n in enumerate(names) if n != "climb"]
        self.i_fwd = self.ids.index(FORWARD)
        self.i_veers = [
            self.ids.index(names.index(n)) for n in ("veer_left", "veer_right")
        ]
        sub = vecs[self.ids]
        self.cands = torch.tensor(sub / np.array(meta["a_norm"], dtype=np.float32))
        # per-candidate cost terms that never change: heading error vs the +x
        # goal direction, and forward progress normalised to the menu's fastest
        # candidate — so the cost trade-offs are identical at every cruise speed
        xy = sub[:, :2]
        spd = np.linalg.norm(xy, axis=1)
        self.heading = torch.tensor(
            np.where(spd > 1e-6, 1.0 - xy[:, 0] / np.maximum(spd, 1e-6), 1.0),
            dtype=torch.float32,
        )
        self.progress = torch.tensor(sub[:, 0] / max(float(sub[:, 0].max()), 1e-6))
        self.vy = sub[:, 1]  # for the centering prior (own odometry only)
        h_w = list(W_HORIZON)[: len(meta["horizons"])]
        self.h_w = torch.tensor(h_w, dtype=torch.float32)
        self.hold, self.evade = 0, FORWARD

    def begin(self, pillars) -> None:
        del pillars  # vision only — the whole point
        self.hold, self.evade = 0, FORWARD

    def decide(self, frame: np.ndarray, state: np.ndarray) -> int:
        # `state` supplies the drone's OWN odometry (y, for the centering
        # prior) — its knowledge of itself, never of the pillars
        if self.hold > 0:  # fly the chosen maneuver through
            self.hold -= DECIDE_EVERY
            return self.evade
        with torch.no_grad():
            z = self.enc(_frame_tensor(frame))  # encoder runs ONCE
            z_hat = self.pred(z.expand(len(self.cands), -1), self.cands)
            p = torch.sigmoid(self.cheads(z_hat))  # (n_cands, horizons, 2 rings)
        warn = p[:, :, 0] @ self.h_w  # urgency-weighted "close soon"
        crit = p[:, :, 1] @ self.h_w  # urgency-weighted "about to hit"
        edge = float(warn[self.i_fwd]) - min(float(warn[i]) for i in self.i_veers)
        # the relative margin has a blind spot: when the drone is already deep
        # in trouble every candidate saturates together and the difference
        # *collapses* — so an absolute near-term backstop ("straight ahead hits
        # within ~170 ms") forces the evasion the margin can no longer see
        imminent = float(p[self.i_fwd, :2, 1].max())
        if edge < MARGIN_WM and imminent < 0.5:
            self.evade = FORWARD  # ahead is no worse than aside — keep flying
            return FORWARD
        danger = 0.25 * warn + 0.75 * crit  # crit carries the in-ring gradient
        y_after = float(state[1]) + self.vy * (EVADE_HOLD / CTRL_HZ)
        cost = (
            W_DANGER * danger
            + W_HEAD * self.heading
            + W_CENTER * torch.tensor(np.abs(y_after), dtype=torch.float32)
            - W_PROG * self.progress
        )
        j = min(self.i_veers, key=lambda i: float(cost[i]))
        self.evade = self.ids[j]
        self.hold = EVADE_HOLD
        return self.evade


class ReactivePolicy:
    """The honest baseline: same encoder, but only the danger-*now* head — no
    look-ahead. Generous handicap: when it does trigger, we hand it the correct
    evasion direction from privileged pillar positions (a real reactive stack
    would need L17's depth net for that). It can only lose on *timing*."""

    def __init__(self, enc, nhead):
        self.enc, self.nhead = enc, nhead
        self.pillars = []
        self.hold = 0
        self.evade = FORWARD

    def begin(self, pillars) -> None:
        self.pillars = [np.array(q) for q in pillars]
        self.hold, self.evade = 0, FORWARD

    def decide(self, frame: np.ndarray, state: np.ndarray) -> int:
        if self.hold > 0:
            self.hold -= DECIDE_EVERY
            return self.evade
        with torch.no_grad():
            p_now = float(torch.sigmoid(self.nhead(self.enc(_frame_tensor(frame)))))
        if p_now > THETA_NOW:
            q = min(self.pillars, key=lambda q: np.linalg.norm(state[0:2] - q))
            away_left = state[1] > q[1]  # privileged direction (see class note)
            self.evade = ACTION_NAMES.index("veer_left" if away_left else "veer_right")
            self.hold = EVADE_HOLD
            return self.evade
        return FORWARD


def run_episode(
    env,
    policy,
    scenario_seed: int,
    tmax: int = TMAX,
    in_path: bool = True,
    speed: float = 1.0,
    solo: bool = False,
    randomize: bool = False,
) -> dict:
    """Fly START -> GOAL_X once under `policy`. The same seed reproduces the
    same pillar course, so two policies can fly literally the same test.
    `in_path=False` gives a course that is safe if flown straight — the
    step-4b eval uses those to count false-positive evasions. `speed` scales
    the whole command set and `solo` strips the side clutter (the step-4c
    sweep raises speed on single-pillar courses until reaction breaks).
    `randomize=True` is step 5's unseen world: random pillar shape/colour,
    0-2 steps of command latency, ±8 % actuation noise, and Lesson 20's fixed
    appearance shift on every frame the *policy* sees (scoring keeps the true
    geometry)."""
    rng = np.random.default_rng(scenario_seed)
    obs, _ = env.reset(seed=int(scenario_seed))
    cmd = VelCommander(make_ctrl(), env.CTRL_TIMESTEP)
    cmd.reset(obs[0][0:3])
    pillars = spawn_pillars(env, rng, in_path=in_path, solo=solo, randomize=randomize)
    policy.begin(pillars)
    vecs = float(speed) * ACTION_VECS
    lat = int(rng.integers(0, 3)) if randomize else 0
    pending = [FORWARD] * max(lat, 1)  # executed command lags the decision
    if randomize:
        from randomize import shift_appearance

    state, a_id, trigger = obs[0], FORWARD, -1
    path, min_clear = [state[0:3].copy()], 9.0
    for t in range(tmax):
        if t % DECIDE_EVERY == 0:
            frame = grab_frame(env)
            if randomize:  # the unseen camera: dimmer, noisier (Lesson 20)
                shifted = shift_appearance(frame[None].astype(np.float32) / 255.0)
                frame = (shifted[0] * 255.0).astype(np.uint8)
            a_id = policy.decide(frame, state)
            if a_id != FORWARD and trigger < 0:
                trigger = t
        pending.append(a_id)
        a_exec = pending.pop(0) if lat else pending.pop()
        v = vecs[a_exec]
        if randomize:
            v = v * (1.0 + rng.normal(0.0, 0.08, size=4))
        obs, _, _, _, _ = env.step(cmd.rpm(state, v).reshape(1, 4))
        state = obs[0]
        path.append(state[0:3].copy())
        min_clear = min(min_clear, nearest_planar(state[0:2], pillars))
        if state[0] >= GOAL_X:
            break
    return {
        "path": np.array(path),
        "pillars": pillars,
        "min_clear": min_clear,
        "trigger": trigger,
        "crashed": min_clear < COLLISION_R,
        "steps": len(path) - 1,
        "reached": bool(state[0] >= GOAL_X),
    }


def load_or_train(device: str = "cpu", path: str = None):
    """Course convention: every step runs on its own. No checkpoint -> train a
    tiny one right here (slower but self-contained). `path` evaluates another
    checkpoint (e.g. one trained on hyperdrone data) instead of MODEL."""
    path = path or MODEL
    if not os.path.exists(path):
        if path != MODEL:
            raise SystemExit(f"no checkpoint at {path}")
        print(f"[INFO] no model at {MODEL}; training a tiny one first ...")
        ckpt, _ = train(gen(12, 100), epochs=40)
        os.makedirs(os.path.dirname(MODEL), exist_ok=True)
        torch.save(ckpt, MODEL)
    return load_model(path, device=device)


def _save_plot(reactive: dict, wm: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    runs = ((reactive, "tab:orange", "reactive"), (wm, "tab:green", "wm (latent MPC)"))
    for run, color, name in runs:
        px = run["path"]
        ax.plot(px[:, 0], px[:, 1], "-", color=color, label=name)
        if run["trigger"] >= 0:
            ax.plot(*px[run["trigger"], :2], "o", color=color, markersize=6)
    ax.plot(0, 0, "ko", markersize=8, label="start")
    ax.axvline(GOAL_X, color="k", linestyle=":", linewidth=1)
    for q in wm["pillars"]:
        ax.add_patch(plt.Circle(q, COLLISION_R, color="red", alpha=0.6))
        ax.add_patch(plt.Circle(q, DANGER_R, color="red", alpha=0.10))
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Closed loop from vision: latent-MPC world model vs reactive")
    ax.legend(loc="best", fontsize=8)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110)
    print(f"[INFO] saved {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--seed", type=int, default=SCENARIO_SEED)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default=None, help="checkpoint (default output/world_model.pth)")
    args = ap.parse_args()

    enc, pred, cheads, nhead, meta = load_or_train(device="cpu", path=args.model)
    env = make_env(gui=args.gui)
    reactive = run_episode(env, ReactivePolicy(enc, nhead), args.seed)
    wm = run_episode(env, WMPolicy(enc, pred, cheads, meta), args.seed)
    env.close()
    _save_plot(reactive, wm)

    lead = reactive["trigger"] - wm["trigger"]
    print(
        f"WM-CLOSED-LOOP OK: reactive min-clear={reactive['min_clear']:.2f} m "
        f"(trigger@{reactive['trigger']}), wm min-clear={wm['min_clear']:.2f} m "
        f"(trigger@{wm['trigger']}), lead=+{lead} steps "
        f"(~{lead * 1000 / CTRL_HZ:.0f} ms earlier), crashes reactive/wm = "
        f"{int(reactive['crashed'])}/{int(wm['crashed'])}, goal steps = "
        f"{reactive['steps']}/{wm['steps']} — danger signal from camera alone "
        f"(no privileged look-ahead in control)"
    )
    if args.selftest:
        assert wm["trigger"] >= 0, "wm never deviated from cruise"
        assert reactive["trigger"] >= 0, "reactive never triggered"
        assert lead > 0, "wm did not trigger earlier than reactive"
        assert wm["min_clear"] > reactive["min_clear"], "no clearance gain"
        assert not wm["crashed"], "wm crashed"
        assert wm["reached"], "wm never reached the goal line"


if __name__ == "__main__":
    main()
    sys.exit(0)
