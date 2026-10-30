"""Recurrent State-Space Model (DreamerV3): deterministic GRU state h_t plus a discrete
stochastic latent z_t (stoch groups x classes). Prior p(z_t | h_t) for imagination, posterior
q(z_t | h_t, embed_t) when a frame is observed; KL balancing (dyn / rep) with free bits."""

import torch
import torch.nn as nn

from .nets import MLP, cat_kl, cat_probs, cat_sample


class NormGRU(nn.Module):
    """DreamerV3's GRU cell: one linear map to (reset, cand, update) with LayerNorm, update gate biased closed."""

    def __init__(self, inp: int, deter: int):
        super().__init__()
        self.lin = nn.Linear(inp + deter, 3 * deter, bias=False)
        self.ln = nn.LayerNorm(3 * deter)

    def forward(self, h, x):
        reset, cand, update = self.ln(self.lin(torch.cat([x, h], -1))).chunk(3, -1)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1.0)
        return update * cand + (1.0 - update) * h


class BlockGRU(nn.Module):
    """DreamerV3's large-model GRU: full input projection, block-diagonal recurrent weights
    (`blocks` independent groups of deter/blocks units), LayerNorm on the gate pre-activations."""

    def __init__(self, inp: int, deter: int, blocks: int):
        super().__init__()
        assert deter % blocks == 0
        self.blocks, self.g = blocks, deter // blocks
        self.inp = nn.Linear(inp, 3 * deter, bias=False)
        self.rec = nn.Parameter(torch.randn(blocks, self.g, 3 * self.g) / self.g ** 0.5)
        self.ln = nn.LayerNorm(3 * deter)

    def forward(self, h, x):
        B = h.shape[0]
        hr = torch.einsum("bkg,kgo->bko", h.view(B, self.blocks, self.g), self.rec.to(h.dtype))  # (B, blocks, 3g)
        hr = hr.view(B, self.blocks, 3, self.g).transpose(1, 2).reshape(B, -1)  # gate-major like inp
        reset, cand, update = self.ln(self.inp(x) + hr).chunk(3, -1)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1.0)
        return update * cand + (1.0 - update) * h


class RSSM(nn.Module):
    def __init__(self, embed_dim: int, action_dim: int, deter: int = 256, stoch: int = 16, classes: int = 16,
                 hidden: int = 128, unimix: float = 0.01, free: float = 1.0, blocks: int = 1):
        super().__init__()
        self.deter, self.stoch, self.classes, self.unimix, self.free = deter, stoch, classes, unimix, free
        self.img_in = nn.Sequential(nn.Linear(stoch * classes + action_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.gru = NormGRU(hidden, deter) if blocks <= 1 else BlockGRU(hidden, deter, blocks)
        self.img_out = MLP(deter, hidden, 1, stoch * classes)  # prior logits
        self.obs_out = MLP(deter + embed_dim, hidden, 1, stoch * classes)  # posterior logits
        self.h0 = nn.Parameter(torch.zeros(deter))

    @property
    def feat_dim(self) -> int:
        return self.deter + self.stoch * self.classes

    def feat(self, state: dict) -> torch.Tensor:
        return torch.cat([state["h"], state["z"].flatten(-2)], -1)

    def initial(self, B: int) -> dict:
        h = torch.tanh(self.h0).expand(B, -1)
        logits = self.img_out(h).view(B, self.stoch, self.classes)
        probs = cat_probs(logits, self.unimix)
        return dict(h=h, z=probs, logits=logits)  # expected one-hot as the initial stochastic state

    def img_step(self, state: dict, action: torch.Tensor) -> dict:
        x = self.img_in(torch.cat([state["z"].flatten(1), action], -1))
        h = self.gru(state["h"], x)
        logits = self.img_out(h).view(-1, self.stoch, self.classes)
        z = cat_sample(cat_probs(logits, self.unimix))
        return dict(h=h, z=z, logits=logits)

    def obs_step(self, state: dict, action: torch.Tensor, embed: torch.Tensor) -> tuple:
        prior = self.img_step(state, action)
        logits = self.obs_out(torch.cat([prior["h"], embed], -1)).view(-1, self.stoch, self.classes)
        z = cat_sample(cat_probs(logits, self.unimix))
        return dict(h=prior["h"], z=z, logits=logits), prior

    @staticmethod
    def _stack(states: list) -> dict:
        return {k: torch.stack([s[k] for s in states], 1) for k in states[0]}

    @staticmethod
    def index(states: dict, b, t=None) -> dict:
        """Pick a sub-batch (and optionally one time step) out of stacked states."""
        return {k: (v[b] if t is None else v[b, t]) for k, v in states.items()}

    def observe(self, embeds: torch.Tensor, prev_actions: torch.Tensor, state: dict | None = None) -> tuple:
        """embeds (B,T,E), prev_actions (B,T,A) = action that led INTO step t. Returns stacked (posts, priors)."""
        B, T = embeds.shape[:2]
        state = state or self.initial(B)
        posts, priors = [], []
        for t in range(T):
            state, prior = self.obs_step(state, prev_actions[:, t], embeds[:, t])
            posts.append(state)
            priors.append(prior)
        return self._stack(posts), self._stack(priors)

    def imagine(self, state: dict, actions: torch.Tensor) -> dict:
        """Roll the prior forward from `state` under actions (B,H,A). Returns stacked states (B,H,...)."""
        out = []
        for t in range(actions.shape[1]):
            state = self.img_step(state, actions[:, t])
            out.append(state)
        return self._stack(out)

    def kl(self, post_logits: torch.Tensor, prior_logits: torch.Tensor) -> tuple:
        """KL balancing: dyn trains the prior towards sg(post), rep trains the posterior towards sg(prior);
        free bits clip each term at `free` nats before averaging."""
        post, prior = cat_probs(post_logits, self.unimix), cat_probs(prior_logits, self.unimix)
        dyn = cat_kl(post.detach(), prior).clamp_min(self.free).mean()
        rep = cat_kl(post, prior.detach()).clamp_min(self.free).mean()
        return dyn, rep
