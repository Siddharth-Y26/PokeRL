"""Action-masked DQN.

Neither Stable-Baselines3 nor sb3-contrib ships a maskable DQN (``MaskablePPO`` exists;
there is no ``MaskableDQN``). This module supplies one, because comparing an unmasked DQN
against a masked PPO would confound the algorithm with the masking and invalidate axis A of
the study.

Masking is applied in two places, and both are necessary:

1. ``MaskedQNetwork.forward`` adds a large negative constant to illegal Q-values. Because
   DQN calls the Q-network for *both* action selection and the bootstrapped target
   (``next_q_values.max(dim=1)``), masking here fixes the target as well — an unmasked target
   would let the agent bootstrap from Q-values of moves it can never take.
2. ``MaskedDQN.predict`` samples epsilon-greedy exploration from the *legal* actions only.
   Without this, uniform sampling over the 26-action space would pick an illegal action the
   large majority of the time, and the exploration branch would carry almost no information.

A finite ``-1e8`` is used rather than ``-inf``: DQN takes a max over Q-values and computes a
Huber loss on them, and ``-inf`` propagates into NaN gradients if a state ever has no legal
action.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork

from pokerl.policies.extractors import DictFeaturesExtractor

ILLEGAL_Q = -1e8


def _sanitised_mask(mask: torch.Tensor) -> torch.Tensor:
    """Treat an all-illegal row as all-legal to keep max/argmax well defined."""
    legal = mask.sum(dim=-1, keepdim=True) > 0
    return torch.where(legal, mask, torch.ones_like(mask))


class MaskedQNetwork(QNetwork):
    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        q_values = super().forward(obs)
        mask = _sanitised_mask(obs["action_mask"]).to(q_values.dtype)
        return q_values + (1.0 - mask) * ILLEGAL_Q


class MaskedDQNPolicy(DQNPolicy):
    def __init__(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("net_arch", [64, 64])
        kwargs.setdefault("features_extractor_class", DictFeaturesExtractor)
        super().__init__(*args, **kwargs)

    def make_q_net(self) -> MaskedQNetwork:
        # Mirrors DQNPolicy.make_q_net, but builds the masked network. The features
        # extractor is rebuilt per network, as SB3 does, so the two Q-nets stay independent.
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return MaskedQNetwork(**net_args).to(self.device)


def _random_legal(masks: np.ndarray) -> np.ndarray:
    """Uniformly sample one legal action per row of a batch of masks."""
    actions = []
    for mask in np.atleast_2d(masks):
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            legal = np.arange(mask.size)
        actions.append(np.random.choice(legal))
    return np.array(actions)


class MaskedDQN(DQN):
    """DQN whose random exploration only ever proposes legal actions.

    SB3 samples randomly in two distinct places, and both must be masked:

    - the initial ``learning_starts`` warmup, which calls ``action_space.sample()`` inside
      ``_sample_action`` and never goes through ``predict`` at all;
    - the epsilon-greedy branch of ``predict``.

    Missing the first is not a subtle inefficiency. Of the 26 actions in the gen-9 space only
    a handful are ever legal, so unmasked warmup fills the replay buffer almost entirely with
    illegal transitions, and under ``strict=True`` poke-env raises on the first one.
    """

    def _sample_action(
        self,
        learning_starts: int,
        action_noise=None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.num_timesteps < learning_starts:
            action = _random_legal(self._last_obs["action_mask"])
            return action, action
        return super()._sample_action(learning_starts, action_noise, n_envs)

    def predict(
        self,
        observation: dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ):
        if not deterministic and np.random.rand() < self.exploration_rate:
            action = _random_legal(np.asarray(observation["action_mask"]))
            if not self.policy.is_vectorized_observation(observation):
                action = action[0]
            return action, state
        return self.policy.predict(observation, state, episode_start, deterministic)
