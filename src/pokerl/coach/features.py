"""Named view of the ``v1`` observation vector.

Explanations have to name what drove a decision, and the encoder emits an anonymous 289-wide
float array. This module derives the segment boundaries from the encoder's own constants
rather than hard-coding them, so adding a status or a hazard cannot silently misalign every
label downstream.

Only ``v1`` is mapped. ``v0``'s 12 features carry nothing about the bench — no teammate HP,
types or matchups — so a counterfactual like "what if my switch-in were healthy" has no
feature to perturb, and an attribution over it could never name a reason to switch.
"""

from __future__ import annotations

from dataclasses import dataclass

from pokerl.env.encoders import (
    BOOST_KEYS,
    HAZARDS,
    N_STATUS,
    N_TYPES,
    TERRAIN_FIELDS,
    WEATHERS,
)

N_BOOSTS = len(BOOST_KEYS)
TEAM_SLOT_WIDTH = 3 + N_TYPES   # hp, fainted, status, then the type multi-hot
MOVE_WIDTH = 9


@dataclass(frozen=True)
class Segment:
    name: str
    start: int
    stop: int

    def __contains__(self, index: int) -> bool:
        return self.start <= index < self.stop


def _build() -> dict[str, Segment]:
    segments: dict[str, Segment] = {}
    offset = 0
    for name, size in (
        ("own_active", 1 + N_STATUS + N_TYPES + N_BOOSTS),
        ("opp_active", 2 + N_STATUS + N_TYPES + N_BOOSTS),
        ("moves", 4 * MOVE_WIDTH),
        ("team", 6 * TEAM_SLOT_WIDTH),
        ("opp_team", 6 * 3),
        ("field", len(WEATHERS) + len(TERRAIN_FIELDS) + 2 * len(HAZARDS) + 4),
    ):
        segments[name] = Segment(name, offset, offset + size)
        offset += size
    return segments


SEGMENTS = _build()
SIZE = SEGMENTS["field"].stop

# Individual indices the counterfactual probes need.
OWN_HP = SEGMENTS["own_active"].start
OPP_HP = SEGMENTS["opp_active"].start
OWN_HAZARDS = Segment(
    "own_hazards",
    SEGMENTS["field"].start + len(WEATHERS) + len(TERRAIN_FIELDS),
    SEGMENTS["field"].start + len(WEATHERS) + len(TERRAIN_FIELDS) + len(HAZARDS),
)


def team_slot_hp(slot: int) -> int:
    """Index of teammate ``slot``'s HP fraction."""
    return SEGMENTS["team"].start + slot * TEAM_SLOT_WIDTH


# Human-readable bucket per segment, used to group attribution scores.
BUCKET_LABELS = {
    "own_active": "your active Pokemon",
    "opp_active": "their active Pokemon",
    "moves": "your move options",
    "team": "your bench",
    "opp_team": "their remaining team",
    "field": "field and hazards",
}


def bucket_of(index: int) -> str:
    for name, seg in SEGMENTS.items():
        if index in seg:
            return BUCKET_LABELS[name]
    return "unknown"
