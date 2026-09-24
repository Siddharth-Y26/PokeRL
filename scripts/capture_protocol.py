"""Capture a real Showdown protocol stream and check that BattleTracker replays it faithfully.

``pokerl.coach.reconstruct`` rebuilds a battle from the raw protocol the browser client
receives. Testing that against a hand-written ``|request|`` only proves the happy path; real
traffic carries teampreview, forced switches, fainted Pokemon, weather, and the message types
poke-env quietly tolerates. This script records one genuine battle, replays it through
``BattleTracker``, and asserts the reconstruction matches poke-env's own battle state at every
decision point.

That comparison is the real test: poke-env's internal ``Battle`` is built from the same
messages by the library itself, so any divergence is a bug in our replay rather than in the
protocol.

Usage:
    python scripts/capture_protocol.py --battles 1 --out scripts/_capture.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from poke_env.battle import AbstractBattle  # noqa: E402
from poke_env.player import RandomPlayer  # noqa: E402

from pokerl.coach.reconstruct import BattleTracker  # noqa: E402


class CapturingPlayer(RandomPlayer):
    """Records the raw protocol alongside poke-env's own view of the battle.

    ``checkpoints`` pairs the message stream so far with what poke-env believed the state was
    at that moment, which is what lets the replay be verified rather than merely run.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lines: list[str] = []
        self.checkpoints: list[dict] = []
        self.tag: str | None = None

    async def _handle_battle_message(self, split_messages):
        for split in split_messages:
            if len(split) > 1:
                self.lines.append("|".join(split))
        return await super()._handle_battle_message(split_messages)

    def choose_move(self, battle: AbstractBattle):
        self.tag = battle.battle_tag
        self.checkpoints.append(
            {
                "n_lines": len(self.lines),
                "request": battle.last_request,
                "turn": battle.turn,
                "active": battle.active_pokemon.species if battle.active_pokemon else None,
                "moves": sorted(m.id for m in battle.available_moves),
                "switches": sorted(p.species for p in battle.available_switches),
            }
        )
        return super().choose_move(battle)


def verify(capture: dict) -> tuple[int, int]:
    """Replay the capture through BattleTracker and compare against poke-env at each point."""
    tracker = BattleTracker(username="capturer")
    tag = capture["tag"]
    checked = failed = 0

    for checkpoint in capture["checkpoints"]:
        tracker.forget(tag)
        battle = tracker.handle(tag, capture["lines"][: checkpoint["n_lines"]])
        if checkpoint["request"]:
            battle = tracker.handle_request(tag, checkpoint["request"])

        active = battle.active_pokemon.species if battle.active_pokemon else None
        moves = sorted(m.id for m in battle.available_moves)
        switches = sorted(p.species for p in battle.available_switches)

        checked += 1
        problems = []
        if active != checkpoint["active"]:
            problems.append(f"active {active!r} != {checkpoint['active']!r}")
        if moves != checkpoint["moves"]:
            problems.append(f"moves {moves} != {checkpoint['moves']}")
        if switches != checkpoint["switches"]:
            problems.append(f"switches {switches} != {checkpoint['switches']}")
        if problems:
            failed += 1
            print(f"  turn {checkpoint['turn']}: " + "; ".join(problems))

    return checked, failed


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=1)
    ap.add_argument("--out", default="scripts/_capture.json")
    args = ap.parse_args()

    player = CapturingPlayer(battle_format="gen9randombattle", max_concurrent_battles=1)
    opponent = RandomPlayer(battle_format="gen9randombattle", max_concurrent_battles=1)
    await player.battle_against(opponent, n_battles=args.battles)

    capture = {
        "tag": player.tag,
        "lines": player.lines,
        "checkpoints": player.checkpoints,
    }
    Path(args.out).write_text(json.dumps(capture), encoding="utf-8")
    print(
        f"captured {len(capture['lines'])} lines, "
        f"{len(capture['checkpoints'])} decision points -> {args.out}"
    )

    print("\nreplaying through BattleTracker:")
    checked, failed = verify(capture)
    print(f"\n{checked - failed}/{checked} decision points reconstructed exactly")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
