"""The training environment.

``PokeRLEnv`` subclasses poke-env's ``SinglesEnv`` (a PettingZoo-style parallel env) and takes
its encoder and reward as constructor arguments. That single decision is what keeps the
experiment matrix from becoming a directory of near-identical copy-pasted classes: every cell
of the study is a different set of arguments to the same class.

Observations are a Dict space of ``observation`` and ``action_mask``. The mask is carried
inside the observation because SB3's vectorised envs give policies no other channel to
receive per-step action legality, and every policy in this study masks illegal actions.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from gymnasium.spaces import Box
from poke_env import AccountConfiguration
from poke_env.battle import AbstractBattle
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import Player
from stable_baselines3.common.monitor import Monitor

from pokerl.env.encoders import Encoder, get_encoder
from pokerl.env.rewards import RewardConfig, get_reward

DEFAULT_FORMAT = "gen9randombattle"


class PokeRLEnv(SinglesEnv):
    """Singles battle environment with a swappable encoder and reward."""

    def __init__(
        self,
        encoder: Encoder | str = "v0",
        reward: RewardConfig | str = "shaped",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.encoder = get_encoder(encoder) if isinstance(encoder, str) else encoder
        self.reward = get_reward(reward) if isinstance(reward, str) else reward

        # Only the raw observation space is declared here. PokeEnv intercepts assignment to
        # ``observation_spaces`` and wraps each entry into
        # ``Dict({"observation": raw, "action_mask": Box})``, populating the mask itself on
        # every step. Declaring the Dict ourselves would nest it twice.
        self.observation_spaces = {
            agent: Box(
                self.encoder.low,
                self.encoder.high,
                shape=(self.encoder.size,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

    def calc_reward(self, battle: AbstractBattle) -> float:
        return self.reward_computing_helper(battle, **self.reward.kwargs())

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        return self.encoder(battle)


def _unique_accounts() -> tuple[AccountConfiguration, AccountConfiguration]:
    """Mint a fresh username pair for one environment.

    poke-env would otherwise generate these with the ``random`` module, which the study seeds
    for reproducibility. Seeding makes the generated names *identical on every run*, so a new
    process reconnects under a name the previous run's socket still holds on the server; the
    sessions cross and battles never close out, surfacing later as "Can not reset player's
    battles while they are still running".

    ``uuid4`` draws from ``os.urandom`` and so is deliberately immune to ``random.seed`` —
    reproducibility must cover the agent's decisions, not its network identity. Showdown caps
    usernames at 18 characters, hence the short token.
    """
    token = uuid.uuid4().hex[:8]
    return (
        AccountConfiguration(f"pkrl{token}a", None),
        AccountConfiguration(f"pkrl{token}b", None),
    )


def make_env(
    opponent: Player,
    encoder: str = "v0",
    reward: str = "shaped",
    battle_format: str = DEFAULT_FORMAT,
    monitor_path: str | None = None,
    **kwargs: Any,
) -> Monitor:
    """Build a single-agent, Monitor-wrapped env ready for Stable-Baselines3.

    ``SingleAgentWrapper`` drives the second agent from ``opponent.choose_move``, turning the
    two-player env into the single-agent one SB3 expects. ``Monitor`` records episode returns
    and lengths, which is where the learning curves come from.
    """
    account1, account2 = _unique_accounts()
    env = PokeRLEnv(
        encoder=encoder,
        reward=reward,
        battle_format=battle_format,
        account_configuration1=account1,
        account_configuration2=account2,
        log_level=40,
        open_timeout=None,
        **kwargs,
    )
    return Monitor(SingleAgentWrapper(env, opponent), filename=monitor_path)
