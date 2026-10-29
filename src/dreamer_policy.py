"""Closed-loop policies on the nano-Dreamer world model (PLAN_dreamerv3.md, M4 / route B1).

DreamerMPCPolicy keeps the RSSM belief across decisions (memory), updates it with every frame at
12 Hz, and asks the prior "what if I held this command for 8 steps?" for every menu candidate.
The trigger and the cost are WMPolicy's, untouched, so a comparison against the one-frame models
is a comparison of world models only. DreamerReactivePolicy is the matching reactive baseline:
the col head's "inside 0.7 m now" read off the posterior belief.
"""

import numpy as np
import torch

from train_dreamer import H_IMAG, MODEL_STEPS
from wm_closed_loop import DECIDE_EVERY, ReactivePolicy, WMPolicy, _frame_tensor

N_SAMPLES = 8  # imagined rollouts per candidate, averaged: one sampled rollout is too noisy to rank veers on


class _Belief:
    """RSSM state handling shared by both policies."""

    def _reset_belief(self, meta, speed: float) -> None:
        vecs = float(speed) * np.array(meta["action_vecs"], dtype=np.float32)
        self.full = torch.tensor(vecs / np.array(meta["a_norm"], dtype=np.float32))  # every command, normalised
        self.state, self.a_prev = None, torch.zeros(1, self.full.shape[1])
        torch.manual_seed(0)  # imagination samples latents: keep an episode reproducible

    @torch.no_grad()
    def _observe(self, frame: np.ndarray) -> None:
        if self.state is None:
            self.state = self.model.rssm.initial(1)
        self.state, _ = self.model.rssm.obs_step(self.state, self.a_prev, self.model.enc(_frame_tensor(frame)))


class DreamerMPCPolicy(WMPolicy, _Belief):
    def __init__(self, model, meta, speed: float = 1.0, allow_stop: bool = False):
        super().__init__(None, None, None, meta, speed=speed, allow_stop=allow_stop)
        self.model, self.speed = model, float(speed)
        self._reset_belief(meta, speed)

    def begin(self, pillars) -> None:
        super().begin(pillars)
        self._reset_belief(self.meta, self.speed)

    @torch.no_grad()
    def probs(self, frame: np.ndarray) -> torch.Tensor:
        """(n_cands, horizons, 2 rings): P(inside the ring within k FUTURE steps) for every menu candidate,
        averaged over N_SAMPLES imagined rollouts. Future-only on purpose: the trained "within k" also
        counts the current state, so inside the warn ring every candidate would read 1.0 and the planner
        could no longer tell the veers apart (traced: it then picked by the centering prior and hit)."""
        K, A = N_SAMPLES, len(self.cands)
        st = {k: v.repeat_interleave(K * A, 0) for k, v in self.state.items()}
        acts = self.cands.repeat(K, 1)[:, None].expand(-1, H_IMAG, -1)
        im = self.model.rssm.imagine(st, acts)
        safe = torch.cumprod(1.0 - self.model.p_col(self.model.rssm.feat(im)), dim=1)  # (K*A, H, 2)
        within = torch.stack([1.0 - safe[:, j - 1] for j in MODEL_STEPS], 1)  # (K*A, horizons, 2)
        return within.view(K, A, len(MODEL_STEPS), -1).mean(0)

    def decide(self, frame: np.ndarray, state: np.ndarray) -> int:
        self._observe(frame)  # the belief follows every frame, also while a manoeuvre is held
        if self.hold > 0:
            self.hold -= DECIDE_EVERY
            a = self.evade
        else:
            a = self._choose(self.probs(frame), state)
        self.a_prev = self.full[a][None]
        return a


class DreamerReactivePolicy(ReactivePolicy, _Belief):
    def __init__(self, model, meta):
        super().__init__(None, None)
        self.model, self.meta = model, meta
        self._reset_belief(meta, 1.0)

    def begin(self, pillars) -> None:
        super().begin(pillars)
        self._reset_belief(self.meta, 1.0)

    @torch.no_grad()
    def p_now(self, frame: np.ndarray) -> float:
        return float(self.model.p_col(self.model.rssm.feat(self.state))[0, 0])

    def decide(self, frame: np.ndarray, state: np.ndarray) -> int:
        self._observe(frame)
        if self.hold > 0:
            self.hold -= DECIDE_EVERY
            a = self.evade
        else:
            a = self._react(self.p_now(frame), state)
        self.a_prev = self.full[a][None]
        return a
