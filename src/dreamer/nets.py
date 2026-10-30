"""Building blocks of the nano DreamerV3-style world model (PyTorch).

Kept as in DreamerV3 (danijar/dreamerv3): LayerNorm + SiLU everywhere, symlog / symexp,
two-hot regression over symlog bins, categorical latents with 1% uniform mixing (unimix)
and straight-through gradients, image decoder trained with a summed-over-pixels MSE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class MLP(nn.Module):
    def __init__(self, inp: int, units: int, layers: int, out: int | None = None):
        super().__init__()
        seq, d = [], inp
        for _ in range(layers):
            seq += [nn.Linear(d, units), nn.LayerNorm(units), nn.SiLU()]
            d = units
        if out is not None:
            seq.append(nn.Linear(d, out))
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


class ChanLayerNorm(nn.Module):
    """LayerNorm over the channel axis of an NCHW tensor (DreamerV3's conv norm)."""

    def __init__(self, c: int):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvEncoder(nn.Module):
    """64x64x3 -> 4x4x(4*depth) -> flat embed. Kernel 4 stride 2, four stages."""

    def __init__(self, depth: int = 16, mults=None):
        super().__init__()
        chans = [depth * m for m in mults] if mults else [depth, 2 * depth, 4 * depth, 4 * depth]
        layers, c_in = [], 3
        for c in chans:
            layers += [nn.Conv2d(c_in, c, 4, 2, 1), ChanLayerNorm(c), nn.SiLU()]
            c_in = c
        self.net = nn.Sequential(*layers)
        self.out_dim = c_in * 4 * 4

    def forward(self, x):  # (B,3,64,64) in [0,1]
        return self.net(x - 0.5).flatten(1)


class ConvDecoder(nn.Module):
    """feat -> 4x4x(2*depth) -> 64x64x3 through four transposed convs (last one linear)."""

    def __init__(self, feat_dim: int, depth: int = 16, mults=None):
        super().__init__()
        if mults:  # DreamerV3 layout: 4x4x(depth*mults[-1]) -> mirrored channels -> 3
            self.c0 = depth * mults[-1]
            chans = [depth * m for m in reversed(mults[:-1])] + [3]
        else:
            self.c0 = 2 * depth
            chans = [4 * depth, 2 * depth, depth, 3]
        self.lin = nn.Linear(feat_dim, self.c0 * 16)
        layers, c_in = [], self.c0
        for i, c in enumerate(chans):
            layers.append(nn.ConvTranspose2d(c_in, c, 4, 2, 1))
            if i < len(chans) - 1:
                layers += [ChanLayerNorm(c), nn.SiLU()]
            c_in = c
        self.net = nn.Sequential(*layers)

    def forward(self, feat):  # -> (B,3,64,64), image units
        return self.net(self.lin(feat).view(-1, self.c0, 4, 4)) + 0.5


class TwoHot(nn.Module):
    """Two-hot regression over fixed bins in symlog space (DreamerV3 reward/critic heads)."""

    def __init__(self, lo: float, hi: float, bins: int = 64):
        super().__init__()
        self.register_buffer("bins", torch.linspace(symlog(torch.tensor(lo)).item(), symlog(torch.tensor(hi)).item(), bins))

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # logits (...,K), target raw (...) -> (...)
        bins = self.bins
        y = symlog(target.float()).clamp(bins[0], bins[-1])
        hi = torch.bucketize(y, bins).clamp(1, len(bins) - 1)
        lo = hi - 1
        w_hi = ((y - bins[lo]) / (bins[hi] - bins[lo])).clamp(0.0, 1.0)
        dist = torch.zeros_like(logits, dtype=torch.float32)
        dist.scatter_add_(-1, lo.unsqueeze(-1), (1.0 - w_hi).unsqueeze(-1))
        dist.scatter_add_(-1, hi.unsqueeze(-1), w_hi.unsqueeze(-1))
        return -(dist * F.log_softmax(logits.float(), -1)).sum(-1)

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        return symexp((F.softmax(logits.float(), -1) * self.bins).sum(-1))


def cat_probs(logits: torch.Tensor, unimix: float) -> torch.Tensor:
    """Softmax over the last axis mixed with `unimix` of uniform (keeps every class alive)."""
    p = F.softmax(logits.float(), -1)
    return (1.0 - unimix) * p + unimix / logits.shape[-1]


def cat_sample(probs: torch.Tensor) -> torch.Tensor:
    """One-hot sample with straight-through gradient: value = sample, gradient = probs."""
    flat = probs.reshape(-1, probs.shape[-1])
    idx = torch.multinomial(flat, 1).squeeze(-1)
    onehot = F.one_hot(idx, probs.shape[-1]).float().view_as(probs)
    return onehot + probs - probs.detach()


def cat_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL[p || q] for categorical probs (..., S, C): summed over classes and stoch groups -> (...)."""
    return (p * (torch.log(p) - torch.log(q))).sum(-1).sum(-1)
