"""End-to-end check of the coach server, driven by a real captured battle.

Replays ``scripts/_capture.json`` over the websocket exactly as the browser userscript will:
protocol lines first, then the ``|request|`` payload, once per decision point. Asserts that
every turn produces a usable recommendation and reports the latency distribution, since the
panel has to keep up with live play.

Run the server first:
    python -m pokerl.coach.server --run results/ppo_masked_v1_seed0

Then:
    python scripts/test_coach_server.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import websockets  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default="scripts/_capture.json")
    ap.add_argument("--url", default="ws://localhost:8765")
    ap.add_argument("--limit", type=int, default=25)
    args = ap.parse_args()

    capture = json.loads(Path(args.capture).read_text(encoding="utf-8"))
    tag = capture["tag"]
    checkpoints = capture["checkpoints"][: args.limit]

    latencies: list[float] = []
    advised = errors = skipped = 0
    switch_recs = 0
    shown = 0

    async with websockets.connect(args.url) as ws:
        sent = 0
        for checkpoint in checkpoints:
            # Only the lines that are new since the last decision, which is what the client
            # forwards: it streams as messages arrive rather than resending the whole log.
            new_lines = capture["lines"][sent : checkpoint["n_lines"]]
            sent = checkpoint["n_lines"]

            await ws.send(json.dumps({
                "battle_tag": tag,
                "lines": new_lines,
                "request": checkpoint["request"],
            }))
            reply = json.loads(await ws.recv())

            if "error" in reply:
                errors += 1
                print(f"  turn {checkpoint['turn']}: ERROR {reply['error']}")
                continue
            if reply.get("waiting") or reply.get("finished"):
                skipped += 1
                continue

            advised += 1
            latencies.append(reply["latency_ms"])
            rec = reply["recommendation"]
            if rec["kind"] == "switch":
                switch_recs += 1

            if shown < 3:
                shown += 1
                print(f"\n--- turn {reply['turn']} ---")
                print(f"  play      : {rec['label']} ({rec['probability']:.0%}, {reply['confidence']})")
                print(f"  split     : attack {reply['p_attack']:.0%} / switch {reply['p_switch']:.0%}")
                print(f"  eval      : {reply['value']:+.2f}   ({reply['latency_ms']} ms)")
                for cf in reply["counterfactuals"]:
                    print(f"  why       : {cf['sentence']}")
                for fact in reply["facts"][:2]:
                    print(f"  fact      : {fact}")

    print(f"\n{advised} advised, {skipped} waiting/finished, {errors} errors")
    print(f"switch recommended in {switch_recs}/{advised} positions")
    if latencies:
        print(
            f"latency: median {statistics.median(latencies):.0f} ms, "
            f"p95 {sorted(latencies)[int(len(latencies) * 0.95) - 1]:.0f} ms, "
            f"max {max(latencies):.0f} ms"
        )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
