"""State encoders.

An encoder turns a poke-env ``AbstractBattle`` into a flat float32 vector. Encoders are the
primary independent variable of this study (experiment axis B), so they are swappable objects
rather than a method on the environment.

Two are provided:

``v0``  12 features. The encoding used by poke-env's official example and by poke_RL
        (SBGames 2024). Exists to reproduce prior work and validate the pipeline.
``v1``  ~289 features. Adds status, types, stat boosts, per-move detail, full team state,
        and field conditions. This is the study's contribution over prior work.

All features are normalised into roughly ``[-1, 4]`` to match the observation ``Box``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from poke_env.battle import (
    AbstractBattle,
    Field,
    Move,
    MoveCategory,
    Pokemon,
    PokemonType,
    SideCondition,
    Status,
    Weather,
)
from poke_env.data import GenData

# Stable, ordered vocabularies. Sorting by name keeps the feature layout reproducible across
# runs and poke-env versions (enum declaration order is not guaranteed stable).
TYPES: list[PokemonType] = sorted(PokemonType, key=lambda t: t.name)
STATUSES: list[Status] = sorted(Status, key=lambda s: s.name)
WEATHERS: list[Weather] = sorted(Weather, key=lambda w: w.name)
BOOST_KEYS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")

TERRAIN_FIELDS = [
    Field.ELECTRIC_TERRAIN,
    Field.GRASSY_TERRAIN,
    Field.MISTY_TERRAIN,
    Field.PSYCHIC_TERRAIN,
    Field.TRICK_ROOM,
]
HAZARDS = [
    SideCondition.STEALTH_ROCK,
    SideCondition.SPIKES,
    SideCondition.TOXIC_SPIKES,
    SideCondition.STICKY_WEB,
]

N_TYPES = len(TYPES)
N_STATUS = len(STATUSES)


class Encoder(ABC):
    """Maps a battle state to a fixed-length float32 vector."""

    name: str
    size: int
    low: float = -1.0
    high: float = 4.0

    @abstractmethod
    def __call__(self, battle: AbstractBattle) -> np.ndarray: ...


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _type_multihot(mon: Pokemon | None) -> np.ndarray:
    """Multi-hot over types. A dual-type Pokemon sets two entries."""
    vec = np.zeros(N_TYPES, dtype=np.float32)
    if mon is None:
        return vec
    for t in (mon.type_1, mon.type_2):
        if t is not None:
            vec[TYPES.index(t)] = 1.0
    return vec


def _status_onehot(mon: Pokemon | None) -> np.ndarray:
    vec = np.zeros(N_STATUS, dtype=np.float32)
    if mon is not None and mon.status is not None:
        vec[STATUSES.index(mon.status)] = 1.0
    return vec


def _boosts(mon: Pokemon | None) -> np.ndarray:
    """Stat boosts scaled from [-6, +6] to [-1, +1]."""
    if mon is None:
        return np.zeros(len(BOOST_KEYS), dtype=np.float32)
    return np.array(
        [mon.boosts.get(k, 0) / 6.0 for k in BOOST_KEYS], dtype=np.float32
    )


def _hp(mon: Pokemon | None) -> float:
    if mon is None or mon.current_hp_fraction is None:
        return 0.0
    return float(mon.current_hp_fraction)


def _damage_multiplier(move: Move, battle: AbstractBattle) -> float:
    """Type effectiveness of ``move`` against the opponent's active Pokemon."""
    opp = battle.opponent_active_pokemon
    move_type = _move_attr(move, "type", None)
    if opp is None or move_type is None:
        return 1.0
    return float(
        move_type.damage_multiplier(
            opp.type_1, opp.type_2, type_chart=GenData.from_gen(battle.gen).type_chart
        )
    )


def _move_attr(move: Move, name: str, default: Any) -> Any:
    """Read a Move property, tolerating gaps in poke-env's move data.

    Several ``Move`` properties index straight into the raw data entry
    (``self.entry["priority"]``), and not every entry carries every key — Z-moves, Max moves,
    and a few generated forms are missing some. The lookup then raises ``KeyError`` from deep
    inside a property access, which kills the SubprocVecEnv worker and surfaces in the parent
    as an uninformative ``EOFError`` on a broken pipe.
    """
    try:
        return getattr(move, name)
    except (KeyError, AttributeError, TypeError):
        return default


def _accuracy(move: Move) -> float:
    """poke-env reports never-miss moves as True; normalise everything to a float."""
    acc = _move_attr(move, "accuracy", 1.0)
    if acc is True or acc is None or acc is False:
        return 1.0
    return float(acc)


# --------------------------------------------------------------------------------------
# v0 — 12-feature baseline
# --------------------------------------------------------------------------------------


class EncoderV0(Encoder):
    """The 12-feature encoding from poke-env's example and poke_RL.

    Four move base powers, four damage multipliers, fainted counts on both sides, and both
    active Pokemon's HP fractions. Deliberately identical to prior work so that the ``v0``
    results are directly comparable.
    """

    name = "v0"
    size = 12

    def __call__(self, battle: AbstractBattle) -> np.ndarray:
        base_power = -np.ones(4, dtype=np.float32)
        multiplier = np.ones(4, dtype=np.float32)
        for i, move in enumerate(battle.available_moves[:4]):
            base_power[i] = _move_attr(move, "base_power", 0) / 100.0
            multiplier[i] = _damage_multiplier(move, battle)

        fainted_ours = len([m for m in battle.team.values() if m.fainted]) / 6.0
        fainted_theirs = len([m for m in battle.opponent_team.values() if m.fainted]) / 6.0

        return np.concatenate(
            [
                base_power,
                multiplier,
                [fainted_ours, fainted_theirs],
                [_hp(battle.active_pokemon), _hp(battle.opponent_active_pokemon)],
            ],
            dtype=np.float32,
        )


# --------------------------------------------------------------------------------------
# v1 — rich encoding
# --------------------------------------------------------------------------------------


class EncoderV1(Encoder):
    """Rich encoding: active Pokemon detail, per-move detail, full team, and field state.

    Hidden information is handled explicitly. The opponent's team is only partially
    revealed in a real battle, so unrevealed slots encode as zeros with a ``revealed``
    flag rather than being silently treated as absent.
    """

    name = "v1"
    # active(35) + opp_active(36) + moves(36) + team(138) + opp_team(18) + field(26)
    size = 289

    def __call__(self, battle: AbstractBattle) -> np.ndarray:
        parts: list[np.ndarray] = []

        # --- our active Pokemon (35) ---
        me = battle.active_pokemon
        parts += [
            np.array([_hp(me)], dtype=np.float32),
            _status_onehot(me),
            _type_multihot(me),
            _boosts(me),
        ]

        # --- opponent active Pokemon (36) ---
        opp = battle.opponent_active_pokemon
        parts += [
            np.array([_hp(opp), 1.0 if opp is not None else 0.0], dtype=np.float32),
            _status_onehot(opp),
            _type_multihot(opp),
            _boosts(opp),
        ]

        # --- our four available moves (4 x 9 = 36) ---
        moves = np.zeros((4, 9), dtype=np.float32)
        moves[:, 0] = -1.0  # unavailable slots read as base power -1, matching v0
        for i, move in enumerate(battle.available_moves[:4]):
            cat = np.zeros(3, dtype=np.float32)
            category = _move_attr(move, "category", None)
            if category == MoveCategory.PHYSICAL:
                cat[0] = 1.0
            elif category == MoveCategory.SPECIAL:
                cat[1] = 1.0
            else:
                cat[2] = 1.0
            max_pp = _move_attr(move, "max_pp", 0)
            pp_frac = (_move_attr(move, "current_pp", 0) / max_pp) if max_pp else 0.0
            stab = 1.0 if (me is not None and move.type in me.types) else 0.0
            moves[i] = [
                _move_attr(move, "base_power", 0) / 100.0,
                _accuracy(move),
                *cat,
                _damage_multiplier(move, battle),
                pp_frac,
                _move_attr(move, "priority", 0) / 5.0,
                stab,
            ]
        parts.append(moves.ravel())

        # --- our team, 6 slots (6 x 23 = 138) ---
        team = np.zeros((6, 23), dtype=np.float32)
        for i, mon in enumerate(list(battle.team.values())[:6]):
            team[i] = [
                _hp(mon),
                1.0 if mon.fainted else 0.0,
                1.0 if mon.status is not None else 0.0,
                *_type_multihot(mon),
            ]
        parts.append(team.ravel())

        # --- opponent team, 6 slots (6 x 3 = 18) ---
        opp_team = np.zeros((6, 3), dtype=np.float32)
        for i, mon in enumerate(list(battle.opponent_team.values())[:6]):
            opp_team[i] = [_hp(mon), 1.0 if mon.fainted else 0.0, 1.0]
        parts.append(opp_team.ravel())

        # --- field and side conditions (26) ---
        weather = np.zeros(len(WEATHERS), dtype=np.float32)
        for w in battle.weather:
            weather[WEATHERS.index(w)] = 1.0

        terrain = np.array(
            [1.0 if f in battle.fields else 0.0 for f in TERRAIN_FIELDS],
            dtype=np.float32,
        )
        our_hazards = np.array(
            [float(battle.side_conditions.get(h, 0)) for h in HAZARDS], dtype=np.float32
        )
        their_hazards = np.array(
            [float(battle.opponent_side_conditions.get(h, 0)) for h in HAZARDS],
            dtype=np.float32,
        )
        misc = np.array(
            [
                min(battle.turn / 100.0, 1.0),
                1.0 if battle.can_tera else 0.0,
                1.0 if battle.opponent_used_tera else 0.0,
                1.0 if battle.force_switch else 0.0,
            ],
            dtype=np.float32,
        )
        parts += [weather, terrain, our_hazards, their_hazards, misc]

        obs = np.concatenate(parts, dtype=np.float32)
        # Spikes/toxic spikes stack to 3/2 layers; clip so nothing escapes the Box bounds.
        return np.clip(obs, self.low, self.high)


ENCODERS: dict[str, type[Encoder]] = {"v0": EncoderV0, "v1": EncoderV1}


def get_encoder(name: str) -> Encoder:
    if name not in ENCODERS:
        raise KeyError(f"Unknown encoder {name!r}. Available: {sorted(ENCODERS)}")
    return ENCODERS[name]()
