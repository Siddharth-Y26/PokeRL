"""Tabular Q-learning — the classical-RL arm of the benchmark.

Deep methods need a baseline that is not a neural network, both to show what function
approximation actually buys and to connect this study to the tabular results reported by
poke_RL (SBGames 2024). Continuous features are discretised into a table of ~11.5k states,
small enough to fill with the episode counts available here.

The agent trains against the same Gymnasium env as the SB3 agents, so the comparison is
like-for-like: same opponent, same reward, same episode budget.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from poke_env.battle import AbstractBattle
from poke_env.environment import SinglesEnv
from poke_env.player import BattleOrder, DefaultBattleOrder, Player

from pokerl.env.encoders import EncoderV0, _damage_multiplier, _hp

# Bucket edges. Multipliers are the discrete set Pokemon type effectiveness can take.
MULTIPLIER_BUCKETS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
HP_BUCKETS = [0.0, 0.25, 0.5, 0.75]
POWER_BUCKETS = [0.0, 60.0, 90.0, 120.0]


def _bucket(value: float, edges: list[float]) -> int:
    return int(np.searchsorted(edges, value, side="right") - 1)


def discretise(battle: AbstractBattle) -> tuple[int, ...]:
    """Map a battle to a small discrete state.

    Captures the decision-relevant essentials: how hard our best move hits, how strong it
    is, both sides' health, and how far through the match we are.
    """
    best_mult, best_power = 0.0, 0.0
    for move in battle.available_moves:
        mult = _damage_multiplier(move, battle)
        if mult * move.base_power >= best_mult * best_power:
            best_mult, best_power = mult, move.base_power

    return (
        _bucket(best_mult, MULTIPLIER_BUCKETS),
        _bucket(best_power, POWER_BUCKETS),
        _bucket(_hp(battle.active_pokemon), HP_BUCKETS),
        _bucket(_hp(battle.opponent_active_pokemon), HP_BUCKETS),
        len([m for m in battle.team.values() if m.fainted]),
        len([m for m in battle.opponent_team.values() if m.fainted]),
    )


class TabularQAgent:
    """Epsilon-greedy Q-learning over the discretised state."""

    def __init__(
        self,
        n_actions: int,
        learning_rate: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        exploration_fraction: float = 0.5,
        seed: int = 0,
    ):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.exploration_fraction = exploration_fraction
        self.rng = np.random.default_rng(seed)
        self.q: dict[tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float32)
        )

    def act(self, state: tuple[int, ...], mask: np.ndarray, epsilon: float = 0.0) -> int:
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            return 0
        if self.rng.random() < epsilon:
            return int(self.rng.choice(legal))
        q = self.q[state].copy()
        q[~mask.astype(bool)] = -np.inf
        # Break ties uniformly at random, never by index order. Most state-action pairs are
        # never visited, so ties are the common case rather than an edge case, and
        # ``np.argmax`` would resolve every one of them to the lowest legal index. Actions
        # 0-5 are switches, so that bias makes an undertrained agent switch almost every
        # turn, take free damage, and lose to opponents it should beat.
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def update(
        self,
        state: tuple[int, ...],
        action: int,
        reward: float,
        next_state: tuple[int, ...] | None,
        next_mask: np.ndarray | None,
    ) -> None:
        target = reward
        if next_state is not None and next_mask is not None and next_mask.sum() > 0:
            next_q = self.q[next_state].copy()
            next_q[~next_mask.astype(bool)] = -np.inf
            target += self.gamma * float(np.max(next_q))
        self.q[state][action] += self.lr * (target - self.q[state][action])

    def epsilon_at(self, progress: float) -> float:
        """Linear decay over the first ``exploration_fraction`` of training."""
        frac = min(progress / self.exploration_fraction, 1.0)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"n_actions": self.n_actions, "q": dict(self.q)}, f)

    @classmethod
    def load(cls, path: str | Path) -> TabularQAgent:
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(n_actions=data["n_actions"])
        agent.q.update(data["q"])
        return agent


class TabularQPlayer(Player):
    """Evaluation adapter, mirroring ``PolicyPlayer`` for the tabular agent."""

    def __init__(self, agent: TabularQAgent, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.agent = agent

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if battle.wait:
            return DefaultBattleOrder()
        mask = np.array(SinglesEnv.get_action_mask(battle), dtype=np.int8)
        action = self.agent.act(discretise(battle), mask, epsilon=0.0)
        return SinglesEnv.action_to_order(np.int64(action), battle, strict=False)
