"""
Lesson 20 — domain randomization primitives
============================================
Reusable image-space augmentations. Training a perception model on randomized
appearance (brightness, noise, blur) forces it to key on *shape*, not the exact
colours of one clean sim render — so it survives an appearance it never saw. The
same primitives are reused by Lesson 25 to mimic a real camera.
"""

import numpy as np


def jitter(imgs, rng):
    """Per-image random brightness + heavy Gaussian noise (training-time DR). The
    noise range is wide on purpose so it covers the appearance shifts a deployed
    model will actually hit."""
    out = imgs.copy()
    b = rng.uniform(0.5, 1.5, size=(len(out), 1, 1, 1)).astype(np.float32)
    sigma = rng.uniform(0.0, 0.18, size=(len(out), 1, 1, 1)).astype(np.float32)
    out = out * b + rng.standard_normal(out.shape).astype(np.float32) * sigma
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def shift_appearance(imgs, brightness: float = 0.7, noise: float = 0.15, seed: int = 7):
    """A FIXED appearance shift (dimmer + heavily noisier) — stands in for a
    different camera / lighting the model never trained on (a sim-to-real gap)."""
    rng = np.random.default_rng(seed)
    out = imgs * brightness + rng.normal(0, noise, size=imgs.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
