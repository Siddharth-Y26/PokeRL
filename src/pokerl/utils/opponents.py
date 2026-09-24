"""Training and evaluation opponents (experiment axis D).

The three built-in poke-env baselines form a natural difficulty ladder, so they serve as both
the training opponents and the fixed evaluation panel:

``random``      picks uniformly among legal actions
``max_power``   always uses the highest-base-power move
``heuristic``   matchup-aware: switches on bad matchups, sets hazards, tracks speed tiers

``mixed`` cycles through all three during training. It tests whether exposure to varied
opposition produces an agent that generalises better than one trained against a single style
— which is the interesting question axis D exists to answer.

``snapshot`` is self-play against a lagging copy of the agent itself, added for the coach
work. The three scripted baselines are all the opponent an agent ever sees in the original
study, and only ``heuristic`` switches at all; an agent whose opponent never switches has
little reason to learn how to punish or anticipate one. Written ``snapshot:<path>`` to point
at a specific checkpoint.
"""

from __future__ import annotations

import itertools
import threading
from pathlib import Path

from poke_env.battle import AbstractBattle
from poke_env.player import (
    BattleOrder,
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)

OPPONENTS: dict[str, type[Player]] = {
    "random": RandomPlayer,
    "max_power": MaxBasePowerPlayer,
    "heuristic": SimpleHeuristicsPlayer,
}


class MixedOpponent(Player):
    """Delegates each battle to one of the three baselines in round-robin order.

    Rotation is per battle rather than per turn: switching strategy mid-battle would produce
    incoherent play that no real opponent exhibits, and would teach the agent to model an
    opponent that does not exist.
    """

    def __init__(self, battle_format: str, **kwargs):
        super().__init__(battle_format=battle_format, **kwargs)
        self._delegates = [
            cls(battle_format=battle_format, start_listening=False)
            for cls in OPPONENTS.values()
        ]
        self._cycle = itertools.cycle(range(len(self._delegates)))
        self._assignment: dict[str, Player] = {}

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        delegate = self._assignment.get(battle.battle_tag)
        if delegate is None:
            delegate = self._delegates[next(self._cycle)]
            self._assignment[battle.battle_tag] = delegate
        return delegate.choose_move(battle)


# A directory of versioned snapshots, not a single file. Overwriting one path in place is
# not viable on Windows: every SubprocVecEnv worker reloads the moment the mtime changes, so
# a writer's os.replace lands while readers hold the zip open and fails with Access Denied
# (observed on the second write of a 2-worker smoke run). Versioned names mean the writer
# never touches a file a reader might have open.
DEFAULT_SNAPSHOT = Path("results") / "_snapshot"
SNAPSHOT_GLOB = "opponent_*.zip"


class SnapshotOpponent(Player):
    """Self-play against a lagging copy of the learner, reloaded from disk as it improves.

    A *frozen* snapshot would be a poor training opponent here: the best v1 agent wins 26.5%
    against SimpleHeuristicsPlayer, so freezing it hands the learner an opponent weaker than
    the scripted one it already trains against. Instead the trainer periodically writes its
    current policy to ``path`` (see the snapshot callback in ``pokerl.train``) and every
    opponent instance picks it up, so the opposition tracks the learner.

    Reload picks the highest-numbered snapshot in the directory and happens between battles
    rather than mid-battle: swapping policies inside a battle would produce an incoherent
    opponent, the same reason ``MixedOpponent`` rotates per battle. Until the first snapshot
    exists the opponent falls back to ``SimpleHeuristicsPlayer``, so early training is not
    spent against a randomly-initialised network.

    A snapshot being pruned while this instance is loading it is tolerated — the previously
    loaded policy simply stays in use until the next battle.
    """

    def __init__(self, battle_format: str, path: str | Path = DEFAULT_SNAPSHOT, **kwargs):
        super().__init__(battle_format=battle_format, **kwargs)
        self._dir = Path(path)
        self._encoder = None
        self._policy = None
        self._loaded: str | None = None
        self._lock = threading.Lock()
        self._fallback = SimpleHeuristicsPlayer(
            battle_format=battle_format, start_listening=False
        )

    def _maybe_reload(self) -> None:
        try:
            # Zero-padded step counts, so the lexicographic max is the newest.
            latest = max(q.name for q in self._dir.glob(SNAPSHOT_GLOB))
        except (OSError, ValueError):
            return
        if latest == self._loaded:
            return

        # Imported lazily: this module is imported by evaluate.py and the config tooling,
        # neither of which should pay for torch/SB3 just to name an opponent.
        from stable_baselines3 import PPO

        from pokerl.env.encoders import get_encoder

        with self._lock:
            if latest == self._loaded:
                return
            try:
                model = PPO.load(self._dir / latest, device="cpu")
            except (OSError, EOFError):
                return  # pruned or still being written; keep the current policy
            self._policy = model.policy
            self._policy.set_training_mode(False)
            size = model.observation_space["observation"].shape[0]
            self._encoder = get_encoder("v1" if size > 12 else "v0")
            self._loaded = latest

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        self._maybe_reload()
        if self._policy is None:
            return self._fallback.choose_move(battle)

        import numpy as np
        from poke_env.environment import SinglesEnv

        mask = np.array(SinglesEnv.get_action_mask(battle), dtype=np.int8)
        action, _ = self._policy.predict(
            {
                "observation": self._encoder(battle)[None, :],
                "action_mask": mask[None, :],
            },
            deterministic=False,  # a deterministic opponent is trivially exploitable
        )
        action = int(np.asarray(action).flatten()[0])
        if mask.sum() > 0 and not mask[action]:
            return self._fallback.choose_move(battle)
        return SinglesEnv.action_to_order(np.int64(action), battle, strict=False)


def make_opponent(name: str, battle_format: str, **kwargs) -> Player:
    """Build an opponent by name.

    ``start_listening=False`` is the default because opponents driven by
    ``SingleAgentWrapper`` are called directly rather than over a websocket; opening a
    connection for them would consume a server slot for nothing.
    """
    kwargs.setdefault("start_listening", False)
    if name == "mixed":
        return MixedOpponent(battle_format=battle_format, **kwargs)
    if name == "snapshot" or name.startswith("snapshot:"):
        _, _, path = name.partition(":")
        return SnapshotOpponent(
            battle_format=battle_format, path=path or DEFAULT_SNAPSHOT, **kwargs
        )
    if name not in OPPONENTS:
        raise KeyError(
            f"Unknown opponent {name!r}. "
            f"Available: {sorted(OPPONENTS) + ['mixed', 'snapshot[:path]']}"
        )
    return OPPONENTS[name](battle_format=battle_format, **kwargs)
