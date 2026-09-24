"""Rebuild a poke-env ``Battle`` from Showdown protocol traffic.

The coach has to see the battle from the *human's* side. A spectator cannot: Showdown only
sends the private ``|request|`` payload — your team, your moves, your PP — to the player
themselves. So the browser client forwards its own traffic here and this module replays it
into a ``Battle``, which is the same object the training environment builds and therefore the
only thing the encoders know how to read.

Both entry points are poke-env's public API (``parse_message`` / ``parse_request``); nothing
here reaches into private state. Verified against a live position: feeding a ``|request|`` and
a ``|switch|`` line reproduces ``active_pokemon``, ``available_moves`` and
``available_switches``, after which ``SinglesEnv.get_action_mask`` yields the correct legal
set and both encoders emit in-range vectors.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from poke_env.battle import Battle

LOGGER = logging.getLogger("pokerl.coach")

# battle-gen9randombattle-12345  ->  gen9randombattle
_TAG_FORMAT = re.compile(r"^battle-([a-z0-9]+)-")

# The encoders and the trained policy are built around this format: team size, the action
# space, and the type chart generation all assume it. Coaching another format would silently
# produce advice from a model that has never seen it.
SUPPORTED_FORMAT = "gen9randombattle"


class UnsupportedFormat(ValueError):
    pass


def battle_format(battle_tag: str) -> str | None:
    match = _TAG_FORMAT.match(battle_tag)
    return match.group(1) if match else None


class BattleTracker:
    """Keeps one reconstructed ``Battle`` per battle tag.

    A browser may have several battle rooms open at once, and each carries an independent
    protocol stream, so state is keyed by tag rather than held as a single current battle.
    """

    def __init__(self, username: str = "player", gen: int = 9):
        self.username = username
        self.gen = gen
        self._battles: dict[str, Battle] = {}

    def get(self, battle_tag: str) -> Battle | None:
        return self._battles.get(battle_tag)

    def forget(self, battle_tag: str) -> None:
        self._battles.pop(battle_tag, None)

    def _battle(self, battle_tag: str) -> Battle:
        battle = self._battles.get(battle_tag)
        if battle is None:
            fmt = battle_format(battle_tag)
            if fmt is not None and fmt != SUPPORTED_FORMAT:
                raise UnsupportedFormat(
                    f"battle format {fmt!r} is not supported; the trained policy and both "
                    f"encoders assume {SUPPORTED_FORMAT!r}"
                )
            battle = Battle(battle_tag, self.username, LOGGER, gen=self.gen)
            self._battles[battle_tag] = battle
        return battle

    def handle(self, battle_tag: str, lines: list[str]) -> Battle:
        """Feed raw protocol lines for one battle room and return the updated battle.

        ``lines`` are the client's raw strings, e.g. ``|switch|p2a: Dragapult|Dragapult, M|100/100``
        or ``|request|{...}``. Unparseable lines are logged and skipped rather than raised:
        a coach that dies on one unexpected message is worse than one that misses a turn.
        """
        battle = self._battle(battle_tag)
        for line in lines:
            if not line or not line.startswith("|"):
                continue
            split = line.split("|")
            # Protocol lines start with a leading '|', so split()[0] is the empty string and
            # split()[1] is the message type -- the shape poke-env's parser expects.
            message_type = split[1] if len(split) > 1 else ""
            try:
                if message_type == "request":
                    payload = line.split("|", 2)[2]
                    if payload.strip():
                        battle.parse_request(json.loads(payload))
                elif message_type in ("init", "title", "j", "l", "c", "raw", "html"):
                    continue  # room chatter, nothing to do with battle state
                else:
                    battle.parse_message(split)
            except UnsupportedFormat:
                raise
            except Exception as exc:  # noqa: BLE001 - see docstring
                LOGGER.debug("skipped %r: %s", line[:120], exc)
        return battle

    def handle_request(self, battle_tag: str, request: dict[str, Any]) -> Battle:
        """Feed an already-decoded ``|request|`` payload."""
        battle = self._battle(battle_tag)
        battle.parse_request(request)
        return battle
