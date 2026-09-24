"""Wraps a trained SB3 policy back into a poke-env ``Player``.

Evaluation runs through poke-env's own machinery (``battle_against``, ``cross_evaluate``)
rather than through the Gym env, so a trained policy has to look like a ``Player``. This
class is that adapter, and it is deliberately algorithm-agnostic: it calls
``policy.predict``, which is defined on every SB3 policy, so the same class evaluates PPO,
A2C, and DQN agents without branching on the algorithm.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from poke_env.battle import AbstractBattle
from poke_env.environment import SinglesEnv
from poke_env.player import BattleOrder, DefaultBattleOrder, Player

from pokerl.env.encoders import Encoder, get_encoder


class PolicyPlayer(Player):
    def __init__(
        self,
        policy: Any,
        encoder: Encoder | str = "v0",
        deterministic: bool = True,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.encoder = get_encoder(encoder) if isinstance(encoder, str) else encoder
        self.deterministic = deterministic

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if battle.wait:
            return DefaultBattleOrder()

        mask = np.array(SinglesEnv.get_action_mask(battle), dtype=np.int8)
        obs = {
            "observation": self.encoder(battle)[None, :],
            "action_mask": mask[None, :],
        }
        action, _ = self.policy.predict(obs, deterministic=self.deterministic)
        action = int(np.asarray(action).flatten()[0])

        # The unmasked control policy can propose an illegal action. Rather than let
        # poke-env raise, fall back to a default order so the battle completes and the
        # illegal choice is reflected honestly in the win rate.
        if mask.sum() > 0 and not mask[action]:
            return DefaultBattleOrder()
        return SinglesEnv.action_to_order(np.int64(action), battle, strict=False)
