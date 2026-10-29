"""Sequence sampler over the project's rollout datasets for the RSSM.

The datasets are 48 Hz; the model steps at the 12 Hz decision rate (STRIDE = 4 frames per step),
so a 120-frame rollout is a 30-step sequence and each of the 4 phase offsets is one more sequence.
All arrays stay on the training device (uint8 frames)."""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from gen_wm_dataset import A_NORM, counterfactual_labels  # noqa: E402

STRIDE = 4  # 48 Hz frames per 12 Hz model step


class SeqData:
    def __init__(self, data: dict, device: str):
        self.device = device
        frames = np.asarray(data["frames"])
        self.R, self.L = frames.shape[:2]
        self.T = self.L // STRIDE
        self.frames = torch.tensor(frames, device=device)  # (R,L,64,64,3) uint8
        self.actions = torch.tensor(np.asarray(data["actions"], dtype=np.float32) / A_NORM, device=device)
        self.dists = torch.tensor(np.asarray(data["dists"], dtype=np.float32), device=device)
        cf, vis = counterfactual_labels(data)
        self.cf = torch.tensor(cf, dtype=torch.float32, device=device)  # (R,L,6,4,2)
        self.vis = torch.tensor(vis, dtype=torch.float32, device=device)  # (R,L,6)

    def _gather(self, r: torch.Tensor, t_abs: torch.Tensor) -> dict:
        """r (B,), t_abs (B,T) frame indices -> batch dict."""
        rr = r[:, None].expand_as(t_abs)
        t_prev = (t_abs - STRIDE).clamp_min(0)
        a_prev = self.actions[rr, t_prev] * (t_abs >= STRIDE).float()[..., None]  # zero before the first step
        return dict(
            frames=self.frames[rr, t_abs].permute(0, 1, 4, 2, 3).float() / 255.0,  # (B,T,3,64,64)
            a_prev=a_prev, actions=self.actions[rr, t_abs], dists=self.dists[rr, t_abs],
            cf=self.cf[rr, t_abs], vis=self.vis[rr, t_abs], r=r, t_abs=t_abs,
        )

    def sample(self, rolls: np.ndarray, B: int, T: int, rng: np.random.Generator) -> dict:
        r = torch.tensor(rng.choice(rolls, B), device=self.device)
        phase = torch.tensor(rng.integers(0, STRIDE, B), device=self.device)
        start = torch.tensor(rng.integers(0, self.T - T + 1, B), device=self.device)
        t_abs = phase[:, None] + STRIDE * (start[:, None] + torch.arange(T, device=self.device)[None])
        return self._gather(r, t_abs)

    def episodes(self, rolls, phases=range(STRIDE)) -> dict:
        """Every (rollout, phase) as a full-length sequence (N, T)."""
        r = torch.tensor([int(x) for x in rolls for _ in phases], device=self.device)
        ph = torch.tensor([int(p) for _ in rolls for p in phases], device=self.device)
        t_abs = ph[:, None] + STRIDE * torch.arange(self.T, device=self.device)[None]
        return self._gather(r, t_abs)

    def future_min(self, r: torch.Tensor, t_abs: torch.Tensor, k_frames: int) -> torch.Tensor:
        """min clearance over the flown frames [t, t+k] (48 Hz), the dataset's 'within k' label base."""
        idx = (t_abs[:, None] + torch.arange(k_frames + 1, device=self.device)[None]).clamp_max(self.L - 1)
        return self.dists[r[:, None].expand_as(idx), idx].min(1).values
