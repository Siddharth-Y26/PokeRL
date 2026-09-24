"""Feature extractor for the Dict observation space.

The observation is ``{"observation": Box, "action_mask": MultiBinary}``. Only the former is
network input; the mask is consumed by the policy head. This extractor pulls out the
observation tensor and declares the right ``features_dim`` so SB3 sizes the MLP correctly.

Unlike poke-env's example, the dimension is read from the observation space rather than a
module-level constant, so the same class serves both the 12-feature v0 and 289-feature v1
encoders without edits.
"""

from __future__ import annotations

import torch
from gymnasium.spaces import Dict as DictSpace
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DictFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: DictSpace):
        super().__init__(
            observation_space, features_dim=observation_space["observation"].shape[0]
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return obs["observation"]
