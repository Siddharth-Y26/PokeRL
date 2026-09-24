"""Seeding.

Reproducibility is a stated requirement of the study (same seed twice must give the same
learning curve), so every source of randomness is seeded from one place.

Note the honest caveat: the Showdown server itself has its own RNG for damage rolls,
critical hits, accuracy, and team generation, and it is *not* seeded from the Python side.
Runs are therefore reproducible in the agent's decisions but not bit-identical in battle
outcomes. This is exactly why the protocol requires 3 seeds and 500 evaluation battles.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_torch_threads(n: int) -> None:
    """Cap torch's intra-op parallelism. ``n <= 0`` leaves torch at its default.

    Must be called before any tensor work. See ``ExperimentConfig.torch_threads`` for why
    this matters: torch defaults to every core, which on a 64x64 MLP costs more in thread
    synchronisation than it saves, while starving the env workers that are actually the
    critical path.
    """
    if n <= 0:
        return
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(n)
    except RuntimeError:
        # Only settable once, and only before any parallel work has started. If torch has
        # already initialised its inter-op pool, the intra-op cap above is the part that
        # matters anyway.
        pass


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
