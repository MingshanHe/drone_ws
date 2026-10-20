"""
Lesson 29 (step 6) — learn the policy: RL over the world model's eyes
=====================================================================
Step 4 ended on a measured verdict: the model's rankings are perfect
(veer-ranking 1.00) and the hand-crafted cost function is the bottleneck —
eight tuned configurations each fixed one failure mode and exposed the next,
and the cluttered courses kept a 16 % crash tail. This step does what that
verdict asks: **stop tuning the cost, learn it.**

The recipe is Lesson 19's PPO, pointed at Lesson 29's world model:

  * **Observation** — not pixels, and not privileged pillar positions: the
    world model's own outputs. For each of the five candidate commands, the
    8 collision probabilities (4 horizons x warn/crit rings), plus the drone's
    own y and cruise speed and the previous command — 47 numbers per decision.
  * **Memory, two honest flavours** — default: the last 12 decisions stacked
    (~1 s at 12 Hz), the same trick Lesson 19's base env uses for actions.
    `--recurrent`: sb3-contrib's RecurrentPPO instead — the observation is a
    single decision and an LSTM carries the memory, so it can remember
    *longer* than any fixed stack at lower cost. (The one new dependency in
    the whole lesson, and it is optional.)
  * **Actions** — the same five-command menu the hand planner used
    (climb stays off: it games the planar labels).
  * **Reward** — Lesson 19's shape, unchanged in spirit: progress toward the
    goal line, a small time cost, a crash penalty, a goal bonus. No hand-tuned
    danger weights anywhere — that is the point.

Training randomizes cruise speed (0.6–1.6 m/s) and course layout every
episode. `--randomize` goes further and trains inside step 5's storm: random
pillar shapes/colours, 0–2 control steps of command latency, ±8 % actuation
noise, and appearance jitter on every frame — so the policy learns on the
degraded probabilities it will actually see. `--edge-bias` re-weights the
per-episode speed draw: half the episodes come from the fast edge of the
envelope (1.2–1.6 m/s), because uniform sampling starves the edge twice over
— the top band is a sliver of the range, and fast episodes end sooner, so
their share of *decisions* is smaller still. `--curriculum` spends the same
budget across three diets in sequence — natural first, then the fast edge,
ending mixed — the schedule aimed at holding both bands with one memory,
after `--edge-bias` measurably traded one band for the other. The learned
policy then drops
into the *same* harnesses as every other policy in this lesson
(`run_episode`, the 4b scoreboard, the 4c sweep), so comparisons are apples
to apples.

Run:
  python src/learn_policy.py --timesteps 300000   # train (~16 min)
  python src/learn_policy.py --recurrent          # LSTM memory
  python src/learn_policy.py --recurrent --edge-bias  # + fast edge
  python src/learn_policy.py --curriculum         # 3-diet schedule
  python src/learn_policy.py --randomize          # in the storm
  python src/learn_policy.py --eval               # compare policies
  python src/learn_policy.py --selftest           # tiny, asserts
Saves output/ppo_wm_policy[_recurrent][_rand][_edge][_curr].zip (git-ignored).
Like Lesson 19, the real training runs are manual jobs, not CI smoke tests.
"""

import argparse
import os
import sys
from collections import deque

import gymnasium as gym
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gen_wm_dataset import (  # noqa: E402
    ACTION_VECS,
    COLLISION_R,
    FORWARD,
    VelCommander,
    grab_frame,
    make_ctrl,
    make_env,
    nearest_planar,
    spawn_pillars,
)
from randomize import jitter  # noqa: E402  (Lesson 20's appearance DR)
from train_world_model import load_model  # noqa: E402
from wm_closed_loop import (  # noqa: E402
    DECIDE_EVERY,
    GOAL_X,
    TMAX,
    ReactivePolicy,
    WMPolicy,
    _frame_tensor,
    load_or_train,
    run_episode,
)

HISTORY = 12  # stacked-memory depth (~1 s @ 12 Hz); --recurrent uses 1 + LSTM
SPEED_RANGE = (0.75, 2.0)  # per-episode cruise factor, same envelope as the data
# --edge-bias: uniform sampling starves the envelope edge twice over — the top
# band is a sliver of the range AND fast episodes end sooner, so the edge's
# share of *decisions* is smaller still. Bias half the episodes into the edge.
EDGE_RANGE = (1.5, 2.0)  # the envelope edge: 1.2–1.6 m/s cruise
EDGE_P = 0.5  # with --edge-bias, this fraction of episodes trains at the edge
# --curriculum: one budget, three diets — learn the base skill on the natural
# distribution, drill the starved fast edge, then consolidate on a mixed diet
# so neither band is forgotten. (edge_p, share of the budget) per phase.
CURRICULUM = ((0.0, 1 / 3), (0.5, 1 / 3), (0.25, 1 / 3))
POLICY_ZIP = os.path.join(HERE, "output", "ppo_wm_policy.zip")


def zip_path(
    recurrent: bool = False,
    randomize: bool = False,
    edge: bool = False,
    curr: bool = False,
) -> str:
    suffix = (
        ("_recurrent" if recurrent else "")
        + ("_rand" if randomize else "")
        + ("_edge" if edge else "")
        + ("_curr" if curr else "")
    )
    return os.path.join(HERE, "output", f"ppo_wm_policy{suffix}.zip")


def _menu(meta):
    """The five-command menu (climb off, as in WMPolicy) -> original ids."""
    names = list(meta["action_names"])
    return [i for i, n in enumerate(names) if n != "climb"]


class ObsBuilder:
    """One decision's observation: the world model's 8 collision probabilities
    per candidate command + own y + cruise factor + previous command one-hot,
    stacked over the last `history` decisions (history=1 for the recurrent
    policy — its LSTM is the memory). Shared verbatim by the training env and
    the deployed LearnedPolicy, so train and fly see the same thing."""

    def __init__(self, enc, pred, cheads, meta, speed: float, history: int = HISTORY):
        self.enc, self.pred, self.cheads = enc, pred, cheads
        self.ids = _menu(meta)
        vecs = float(speed) * np.array(meta["action_vecs"], dtype=np.float32)
        a_norm = np.array(meta["a_norm"], dtype=np.float32)
        self.cands = torch.tensor(vecs[self.ids] / a_norm)
        self.speed = float(speed)
        self.n_act = len(self.ids)
        self.per_step = self.n_act * 8 + 2 + self.n_act  # probs + y,speed + prev
        self.history = int(history)
        self.hist = deque(maxlen=self.history)
        self.reset()

    def reset(self) -> None:
        self.hist.clear()
        for _ in range(self.history):
            self.hist.append(np.zeros(self.per_step, dtype=np.float32))

    def push(self, frame: np.ndarray, y: float, prev_menu_idx: int) -> np.ndarray:
        with torch.no_grad():
            z = self.enc(_frame_tensor(frame))
            z_hat = self.pred(z.expand(len(self.cands), -1), self.cands)
            p = torch.sigmoid(self.cheads(z_hat)).numpy()  # (n_act, H, 2)
        prev = np.zeros(self.n_act, dtype=np.float32)
        prev[prev_menu_idx] = 1.0
        step = np.concatenate(
            [
                p.reshape(-1),
                np.array([y / 2.5, self.speed / SPEED_RANGE[1]], dtype=np.float32),
                prev,
            ]
        ).astype(np.float32)
        self.hist.append(step)
        return np.concatenate(self.hist)


class WMPolicyEnv(gym.Env):
    """Gymnasium wrapper around the lesson's own closed-loop harness: the sim
    steps at 48 Hz under the PID VelCommander, the agent decides at 12 Hz, and
    everything the agent sees comes from the camera through the world model —
    pillar positions only stage and score, exactly like `run_episode`.
    `randomize=True` trains inside step 5's storm (random pillar shapes,
    command latency, actuation noise, appearance jitter)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        seed0: int = 0,
        history: int = HISTORY,
        randomize: bool = False,
        edge_bias: bool = False,
    ):
        super().__init__()
        self.env = make_env()
        self.enc, self.pred, self.cheads, self.nhead, self.meta = load_model()
        self.rng = np.random.default_rng(seed0)
        self.history = int(history)
        self.randomize = bool(randomize)
        self.edge_p = EDGE_P if edge_bias else 0.0
        probe = ObsBuilder(
            self.enc, self.pred, self.cheads, self.meta, 1.0, history=self.history
        )
        self.obs_dim = probe.per_step * self.history
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(probe.n_act)
        self.max_decisions = TMAX // DECIDE_EVERY

    def set_edge_p(self, p: float) -> None:
        """Curriculum hook: change the speed diet between training phases
        (called through the VecEnv, so it reaches past the Monitor wrapper)."""
        self.edge_p = float(p)

    def _frame(self) -> np.ndarray:
        frame = grab_frame(self.env)
        if self.randomize:  # Lesson 20's appearance DR, per frame
            out = jitter(frame[None].astype(np.float32) / 255.0, self.rng)
            frame = (out[0] * 255.0).astype(np.uint8)
        return frame

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        scenario = int(self.rng.integers(2**31 - 1))
        obs, _ = self.env.reset(seed=scenario)
        self.cmd = VelCommander(make_ctrl(), self.env.CTRL_TIMESTEP)
        self.cmd.reset(obs[0][0:3])
        rng = np.random.default_rng(scenario)
        # mostly threatened, some clear — the distribution the tail lives in
        self.pillars = spawn_pillars(
            self.env,
            rng,
            in_path=bool(self.rng.random() < 0.8),
            randomize=self.randomize,
        )
        band = SPEED_RANGE
        if self.edge_p > 0.0 and self.rng.random() < self.edge_p:
            band = EDGE_RANGE
        speed = float(self.rng.uniform(*band))
        self.ob = ObsBuilder(
            self.enc, self.pred, self.cheads, self.meta, speed, history=self.history
        )
        self.vecs = speed * ACTION_VECS
        self.lat = int(self.rng.integers(0, 3)) if self.randomize else 0
        self.pending = [FORWARD] * max(self.lat, 1)
        self.state = obs[0]
        self.prev_x = float(self.state[0])
        self.decisions = 0
        return self.ob.push(self._frame(), float(self.state[1]), 0), {}

    def step(self, action: int):
        a_id = self.ob.ids[int(action)]
        self.pending.append(a_id)
        a_exec = self.pending.pop(0) if self.lat else self.pending.pop()
        for _ in range(DECIDE_EVERY):  # the PID flies 4 control steps per decision
            v = self.vecs[a_exec]
            if self.randomize:  # actuation wobbles; the intent stays clean
                v = v * (1.0 + self.rng.normal(0.0, 0.08, size=4))
            rpm = self.cmd.rpm(self.state, v)
            obs, _, _, _, _ = self.env.step(rpm.reshape(1, 4))
            self.state = obs[0]
        self.decisions += 1
        x, y = float(self.state[0]), float(self.state[1])
        d = nearest_planar(self.state[0:2], self.pillars)

        # Lesson 19's reward shape: progress, small time cost, crash, goal —
        # and deliberately no danger-shaping term (that is what we learn)
        reward = 25.0 * (x - self.prev_x) - 0.02
        self.prev_x = x
        terminated, truncated = False, False
        if d < COLLISION_R:
            reward -= 30.0
            terminated = True
        elif x >= GOAL_X:
            reward += 50.0
            terminated = True
        elif abs(y) > 2.4 or self.decisions >= self.max_decisions:
            truncated = True

        next_obs = self.ob.push(self._frame(), y, int(action))
        return next_obs, float(reward), terminated, truncated, {}

    def close(self):
        self.env.close()


class LearnedPolicy:
    """The trained policy behind the same interface as WMPolicy /
    ReactivePolicy, so `run_episode`, the 4b scoreboard and the 4c sweep can
    fly it unchanged. Vision only: frames -> world model -> memory (stacked
    history or the LSTM's hidden state) -> network -> one of five commands.
    The memory flavour and stack depth are inferred from the loaded model."""

    def __init__(self, model, enc, pred, cheads, meta, speed: float = 1.0):
        self.model = model
        self.recurrent = model.__class__.__name__ == "RecurrentPPO"
        probe = ObsBuilder(enc, pred, cheads, meta, speed, history=1)
        history = int(model.observation_space.shape[0]) // probe.per_step
        self.ob = ObsBuilder(enc, pred, cheads, meta, speed, history=history)
        self.i_fwd = self.ob.ids.index(FORWARD)
        self.prev = self.i_fwd
        self.lstm_state = None
        self.first = True

    def begin(self, pillars) -> None:
        del pillars  # vision only — same rule as WMPolicy
        self.ob.reset()
        self.prev = self.i_fwd
        self.lstm_state = None
        self.first = True

    def decide(self, frame: np.ndarray, state: np.ndarray) -> int:
        obs = self.ob.push(frame, float(state[1]), self.prev)
        if self.recurrent:
            action, self.lstm_state = self.model.predict(
                obs,
                state=self.lstm_state,
                episode_start=np.array([self.first]),
                deterministic=True,
            )
            self.first = False
        else:
            action, _ = self.model.predict(obs, deterministic=True)
        self.prev = int(action)
        return self.ob.ids[self.prev]


def _load_policy(path: str):
    if "_recurrent" in os.path.basename(path):
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO.load(path)
    from stable_baselines3 import PPO

    return PPO.load(path)


def train(
    timesteps: int,
    seed0: int = 0,
    recurrent: bool = False,
    randomize: bool = False,
    edge_bias: bool = False,
    out: str = None,
    n_steps: int = 256,
    lstm_size: int = 64,
):
    from stable_baselines3.common.env_util import make_vec_env

    history = 1 if recurrent else HISTORY
    env = make_vec_env(
        lambda: WMPolicyEnv(
            seed0=seed0, history=history, randomize=randomize, edge_bias=edge_bias
        ),
        n_envs=1,
    )
    if recurrent:
        from sb3_contrib import RecurrentPPO

        # A right-sized LSTM: the observation is 47 numbers, so the default
        # 256-wide hidden state is mostly empty capacity that slows learning.
        # n_steps=256 gives backprop-through-time a window longer than the
        # stacked variant's 12 decisions.
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            ent_coef=0.01,
            n_steps=n_steps,
            policy_kwargs=dict(lstm_hidden_size=lstm_size),
            verbose=0,
        )
    else:
        from stable_baselines3 import PPO

        model = PPO("MlpPolicy", env, ent_coef=0.01, verbose=0)
    model.learn(total_timesteps=timesteps)
    out = out or zip_path(recurrent, randomize, edge_bias)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    model.save(out)
    env.close()
    return model


def train_curriculum(
    timesteps: int,
    seed0: int = 0,
    out: str = None,
    n_steps: int = 256,
    lstm_size: int = 64,
):
    """The mixed-diet curriculum (recurrent only — the stack never needed it):
    one model, one total budget, three diets in sequence per `CURRICULUM`.
    Everything else matches `train(recurrent=True)`, so against the uniform
    and edge-biased runs the *order of the data* is the only variable."""
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.env_util import make_vec_env

    env = make_vec_env(lambda: WMPolicyEnv(seed0=seed0, history=1), n_envs=1)
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        ent_coef=0.01,
        n_steps=n_steps,
        policy_kwargs=dict(lstm_hidden_size=lstm_size),
        verbose=0,
    )
    done = 0
    for i, (edge_p, share) in enumerate(CURRICULUM):
        last = i == len(CURRICULUM) - 1
        chunk = timesteps - done if last else int(round(timesteps * share))
        env.env_method("set_edge_p", edge_p)
        print(f"[INFO] curriculum phase {i + 1}: edge_p={edge_p}, {chunk} steps")
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        done += chunk
    out = out or zip_path(recurrent=True, curr=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    model.save(out)
    env.close()
    return model


def compare(n_seeds: int, seed0: int = 1000) -> dict:
    """Fly every available policy on identical cluttered courses (the 4b
    distribution, where the 16 % tail lives), plus the 4c speed endpoint on
    single-pillar courses. Learned variants join automatically when their
    trained .zip exists."""
    enc, pred, cheads, nhead, meta = load_or_train(device="cpu")
    env = make_env()

    mk = {
        "reactive": lambda s: ReactivePolicy(enc, nhead),
        "wm-mpc": lambda s: WMPolicy(enc, pred, cheads, meta, speed=s),
    }
    for name, rec, rnd, edge, curr in (
        ("learned", False, False, False, False),
        ("learned-rnn", True, False, False, False),
        ("learned-rnn-edge", True, False, True, False),
        ("learned-rnn-curr", True, False, False, True),
        ("learned-rand", False, True, False, False),
        ("learned-rnn-rand", True, True, False, False),
    ):
        path = zip_path(rec, rnd, edge, curr)
        if os.path.exists(path):
            model = _load_policy(path)
            mk[name] = lambda s, m=model: LearnedPolicy(
                m, enc, pred, cheads, meta, speed=s
            )

    def fly(policy_fn, **kw):
        crash, clear = 0, []
        for i in range(n_seeds):
            run = run_episode(env, policy_fn(kw.get("speed", 1.0)), seed0 + i, **kw)
            crash += int(run["crashed"])
            clear.append(run["min_clear"])
        return crash / n_seeds, float(np.mean(clear))

    out = {}
    for name, fn in mk.items():
        out[name] = {"cluttered": fly(fn, in_path=True)}
    for name, fn in mk.items():
        out[name]["fast"] = fly(fn, in_path=True, solo=True, speed=2.0)
    env.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--recurrent", action="store_true")
    ap.add_argument("--n-steps", type=int, default=256)  # BPTT window (recurrent)
    ap.add_argument("--lstm-size", type=int, default=64)  # hidden width (recurrent)
    ap.add_argument("--edge-bias", action="store_true")  # oversample the fast edge
    ap.add_argument("--curriculum", action="store_true")  # 3-diet schedule (LSTM)
    ap.add_argument("--randomize", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        env = WMPolicyEnv(seed0=7)
        obs, _ = env.reset()
        n_act = env.action_space.n
        per = env.obs_dim // HISTORY
        assert n_act == 5, f"menu should be 5 commands, got {n_act}"
        assert per == 5 * 8 + 2 + 5, f"per-step obs dim off ({per})"
        assert obs.shape == (env.obs_dim,), "stacked obs shape off"
        obs2, r, term, trunc, _ = env.step(0)
        assert obs2.shape == obs.shape and np.isfinite(r), "step broken"
        env.close()
        env = WMPolicyEnv(seed0=7, history=1, randomize=True)  # the storm
        obs, _ = env.reset()
        assert obs.shape == (env.obs_dim,) and env.obs_dim == per, "randomized env off"
        env.step(0)
        env.close()
        paths = {
            zip_path(),
            zip_path(True),
            zip_path(True, True),
            zip_path(True, False, True),
            zip_path(True, False, False, True),
        }
        assert len(paths) == 5, "zip suffixes clash"
        # smoke-train both flavours (wiring, not skill) — into _selftest zips,
        # so a real trained policy is never clobbered by a selftest
        st = os.path.join(HERE, "output", "ppo_wm_policy_selftest.zip")
        st_r = os.path.join(HERE, "output", "ppo_wm_policy_selftest_rnn.zip")
        st_c = os.path.join(HERE, "output", "ppo_wm_policy_selftest_curr.zip")
        train(1500, seed0=7, out=st)
        train(1024, seed0=7, recurrent=True, out=st_r)
        train_curriculum(768, seed0=7, out=st_c)  # one 256-step rollout per diet
        assert os.path.exists(st), "policy zip not saved"
        assert os.path.exists(st_r), "recurrent zip not saved"
        assert os.path.exists(st_c), "curriculum zip not saved"
        print(
            f"LEARN-POLICY OK: obs={HISTORY}x{per} stacked (or 1x{per} + LSTM), "
            f"5 actions, smoke-trained stacked/LSTM/curriculum, randomized env "
            f"steps, saved {st}"
        )
        return

    if not args.eval:
        if args.curriculum:
            print(f"[INFO] RecurrentPPO mixed-diet curriculum, {args.timesteps} steps")
            train_curriculum(
                args.timesteps, n_steps=args.n_steps, lstm_size=args.lstm_size
            )
            print(f"[INFO] saved {zip_path(recurrent=True, curr=True)}")
        else:
            tag = (
                ("recurrent " if args.recurrent else "stacked ")
                + ("+ randomized" if args.randomize else "clean")
                + (" + edge-bias" if args.edge_bias else "")
            )
            print(
                f"[INFO] PPO over world-model outputs ({tag}), {args.timesteps} steps"
            )
            train(
                args.timesteps,
                recurrent=args.recurrent,
                randomize=args.randomize,
                edge_bias=args.edge_bias,
                n_steps=args.n_steps,
                lstm_size=args.lstm_size,
            )
            print(
                f"[INFO] saved "
                f"{zip_path(args.recurrent, args.randomize, args.edge_bias)}"
            )

    res = compare(args.seeds)
    order = [k for k in res]
    clut = " / ".join(f"{k} {res[k]['cluttered'][0]:.0%}" for k in order)
    clr = " / ".join(f"{res[k]['cluttered'][1]:.2f}" for k in order)
    fast = " / ".join(f"{k} {res[k]['fast'][0]:.0%}" for k in order)
    print(
        f"LEARNED-POLICY OK: {args.seeds} cluttered courses @ 0.8 m/s — crash "
        f"{clut} (clearance {clr} m)\n"
        f"  single-pillar @ 1.6 m/s — crash {fast} — the cost function is "
        f"learned, the world model is the same"
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
