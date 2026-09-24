"""Reward functions (experiment axis C).

All variants delegate to poke-env's ``PokeEnv.reward_computing_helper``, which scores the
battle state and returns the *difference* from the previous call. That potential-based form
keeps the optimal policy unchanged under shaping while densifying the learning signal.

``sparse``     Win/loss only. The true objective, and the hardest to learn from.
``shaped``     The example's defaults: HP, faints, status, victory.
``aggressive`` Weights damage dealt more heavily than survival.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class RewardConfig:
    """Keyword arguments forwarded to ``PokeEnv.reward_computing_helper``."""

    name: str
    fainted_value: float = 0.0
    hp_value: float = 0.0
    status_value: float = 0.0
    victory_value: float = 1.0
    number_of_pokemons: int = 6
    starting_value: float = 0.0

    def kwargs(self) -> dict[str, float | int]:
        d = asdict(self)
        d.pop("name")
        return d


REWARDS: dict[str, RewardConfig] = {
    # Win/loss only. Included to quantify how much shaping actually buys.
    "sparse": RewardConfig(name="sparse", victory_value=30.0),
    # Defaults from poke-env's official example; the study's reference setting.
    "shaped": RewardConfig(
        name="shaped",
        fainted_value=2.0,
        hp_value=1.0,
        status_value=0.5,
        victory_value=30.0,
    ),
    # Rewards trading damage; deliberately undervalues status and survival.
    "aggressive": RewardConfig(
        name="aggressive",
        fainted_value=3.0,
        hp_value=2.0,
        status_value=0.1,
        victory_value=30.0,
    ),
}


def get_reward(name: str) -> RewardConfig:
    if name not in REWARDS:
        raise KeyError(f"Unknown reward {name!r}. Available: {sorted(REWARDS)}")
    return REWARDS[name]
