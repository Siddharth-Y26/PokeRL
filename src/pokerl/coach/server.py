"""WebSocket server between the browser client and the coach.

The browser userscript forwards the protocol traffic of the human's own battle rooms here;
this process reconstructs the battle, asks the policy what it would play, and sends back an
``Advice`` payload for the panel to render.

It is deliberately advice-only. Nothing in this process ever sends a ``/choose`` back to
Showdown: the human always makes the move, which is the difference between a teaching tool
and a bot playing on someone's account.

Usage:
    python -m pokerl.coach.server --run results/ppo_masked_v1_seed0
    python -m pokerl.coach.server --run results/ppo_v1_selfplay_seed0 --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import websockets

from pokerl.coach.explain import Coach
from pokerl.coach.reconstruct import SUPPORTED_FORMAT, BattleTracker, UnsupportedFormat
from pokerl.config import ExperimentConfig

LOGGER = logging.getLogger("pokerl.coach.server")

DEFAULT_RUN = Path("results") / "ppo_masked_v1_seed0"


def load_coach(run_dir: Path) -> tuple[Coach, ExperimentConfig]:
    """Build a Coach from a training run directory."""
    from pokerl.evaluate import ALGO_CLASSES

    cfg = ExperimentConfig.from_yaml(run_dir / "config.yaml")
    if cfg.encoder != "v1":
        raise SystemExit(
            f"{run_dir.name} uses the {cfg.encoder} encoder. The coach needs v1: v0's 12 "
            "features contain nothing about the bench, so it cannot explain a switch."
        )
    if cfg.battle_format != SUPPORTED_FORMAT:
        raise SystemExit(
            f"{run_dir.name} was trained on {cfg.battle_format}, not {SUPPORTED_FORMAT}."
        )
    model = ALGO_CLASSES[cfg.algo].load(run_dir / "model.zip", device="cpu")
    return Coach(model.policy, encoder=cfg.encoder), cfg


class CoachServer:
    def __init__(self, coach: Coach, username: str = "player"):
        self.coach = coach
        self.username = username
        # One tracker per connection would lose battle state across reconnects, and one
        # global tracker would mix two browser tabs together; keyed by connection is the
        # honest middle -- a reload starts the room's stream again from the client anyway.
        self._trackers: dict[int, BattleTracker] = {}

    def _tracker(self, connection_id: int) -> BattleTracker:
        if connection_id not in self._trackers:
            self._trackers[connection_id] = BattleTracker(username=self.username)
        return self._trackers[connection_id]

    def _advise(self, tracker: BattleTracker, payload: dict[str, Any]) -> dict[str, Any]:
        battle_tag = payload.get("battle_tag")
        if not battle_tag:
            return {"error": "missing battle_tag"}

        lines = payload.get("lines") or []
        if lines:
            tracker.handle(battle_tag, lines)
        request = payload.get("request")
        if request:
            tracker.handle_request(battle_tag, request)

        battle = tracker.get(battle_tag)
        if battle is None:
            return {"error": f"no state for {battle_tag}"}
        if battle.finished:
            tracker.forget(battle_tag)
            return {"battle_tag": battle_tag, "finished": True}
        if battle.active_pokemon is None:
            return {"battle_tag": battle_tag, "waiting": True}

        started = time.perf_counter()
        advice = self.coach.advise(battle).to_dict()
        advice["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return advice

    async def handle(self, websocket) -> None:
        connection_id = id(websocket)
        tracker = self._tracker(connection_id)
        LOGGER.info("client connected (%s)", connection_id)
        try:
            async for raw in websocket:
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"error": "invalid JSON"}))
                    continue

                try:
                    # advise() is CPU-bound torch work; off-thread so one slow turn cannot
                    # stall the event loop and back up other rooms.
                    response = await asyncio.to_thread(self._advise, tracker, payload)
                except UnsupportedFormat as exc:
                    response = {"error": str(exc)}
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("advice failed")
                    response = {"error": f"{type(exc).__name__}: {exc}"}
                await websocket.send(json.dumps(response))
        except websockets.ConnectionClosed:
            pass
        finally:
            self._trackers.pop(connection_id, None)
            LOGGER.info("client disconnected (%s)", connection_id)


async def serve(run_dir: Path, host: str, port: int, username: str) -> None:
    coach, cfg = load_coach(run_dir)
    server = CoachServer(coach, username=username)
    print(f"coach model : {run_dir.name} (encoder {cfg.encoder}, {cfg.total_timesteps:,} steps)")
    print(f"listening   : ws://{host}:{port}")
    print("advice only -- this process never sends a move to Showdown")
    async with websockets.serve(server.handle, host, port, max_size=2**22):
        await asyncio.Future()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(DEFAULT_RUN))
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--username", default="player")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    asyncio.run(serve(Path(args.run), args.host, args.port, args.username))


if __name__ == "__main__":
    main()
