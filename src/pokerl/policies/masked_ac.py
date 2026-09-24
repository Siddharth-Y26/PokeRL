"""Actor-critic policies for PPO and A2C, with and without action masking.

Masking works by adding ``-inf`` to the logits of illegal actions before the categorical
distribution is built, so illegal actions receive exactly zero probability and contribute no
gradient. The mask is captured in ``forward``/``evaluate_actions`` because SB3 does not thread
the observation through to ``_get_action_dist_from_latent``.

The unmasked variant exists purely as an experimental control: comparing the two isolates how
much of the agent's performance comes from masking versus from learning (experiment axis A).
"""

from __future__ import annotations

from typing import Any

import torch
from stable_baselines3.common.policies import ActorCriticPolicy

from pokerl.policies.extractors import DictFeaturesExtractor


class UnmaskedActorCriticPolicy(ActorCriticPolicy):
    """Standard actor-critic over the Dict observation. Illegal actions stay selectable.

    poke-env's env is constructed with ``strict=False`` for this policy, which converts an
    illegal choice into a default move rather than raising.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("net_arch", [64, 64])
        kwargs.setdefault("features_extractor_class", DictFeaturesExtractor)
        super().__init__(*args, **kwargs)


class MaskedActorCriticPolicy(UnmaskedActorCriticPolicy):
    """Actor-critic that assigns zero probability to illegal actions."""

    def __init__(self, *args: Any, **kwargs: Any):
        self._mask: torch.Tensor | None = None
        super().__init__(*args, **kwargs)

    def forward(self, obs, deterministic: bool = False):
        self._mask = obs["action_mask"]
        return super().forward(obs, deterministic)

    def evaluate_actions(self, obs, actions):
        self._mask = obs["action_mask"]
        return super().evaluate_actions(obs, actions)

    def predict_values(self, obs):
        self._mask = obs["action_mask"]
        return super().predict_values(obs)

    def _predict(self, observation, deterministic: bool = False):
        self._mask = observation["action_mask"]
        return super()._predict(observation, deterministic)

    def get_distribution(self, obs):
        # Not used by training or by predict(), both of which arrive via forward/_predict.
        # It is the entry point for inference code that wants the action *distribution*
        # rather than a sampled action — the coach's explanations read it directly. Without
        # this override the call falls through to _get_action_dist_from_latent carrying
        # whatever mask the previous call happened to leave behind (or None on the first
        # call), which silently puts probability mass on illegal actions.
        self._mask = obs["action_mask"]
        return super().get_distribution(obs)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        logits = self.action_net(latent_pi)
        if self._mask is None:
            return self.action_dist.proba_distribution(logits)
        mask = self._mask
        # A state with no legal action would make every logit -inf and the softmax NaN.
        # Fall back to leaving that row unmasked; poke-env resolves it to a default order.
        legal = mask.sum(dim=-1, keepdim=True) > 0
        mask = torch.where(legal, mask, torch.ones_like(mask))
        penalty = torch.where(
            mask.bool(), torch.zeros_like(logits), torch.full_like(logits, float("-inf"))
        )
        return self.action_dist.proba_distribution(logits + penalty)
