"""Play a battle against a trained agent yourself.

Logs the agent into the local Showdown server under a fixed, known username and waits for you
to challenge it. Everything the study does is agent-vs-scripted-bot; this is the only way to
face one directly, and it is also how you check whether the coach's advice matches what the
model actually plays.

The agent uses deterministic action selection by default, matching the evaluation protocol --
so it plays exactly the moves its win rate was measured on. Pass ``--sampled`` to let it
sample instead, which makes it less predictable across repeated games.

Usage:
    python scripts/play_vs_agent.py --run results/ppo_masked_v1_seed0
    python scripts/play_vs_agent.py --run results/ppo_v1_selfplay_seed0 --games 5 --name Rival

Then in the Showdown client: find the username it prints, and challenge it to a
gen9randombattle.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from poke_env import AccountConfiguration  # noqa: E402

from pokerl.agents.policy_player import PolicyPlayer  # noqa: E402
from pokerl.agents.tabular_q import TabularQAgent, TabularQPlayer  # noqa: E402
from pokerl.config import ExperimentConfig  # noqa: E402
from pokerl.evaluate import ALGO_CLASSES  # noqa: E402


def build_player(run_dir: Path, username: str, deterministic: bool):
    """Load a trained run as a Player that listens on the server under a fixed name.

    Unlike training and evaluation, this player needs a *predictable* username so a human can
    find and challenge it -- the env's uuid4-based accounts exist to avoid cross-run collisions
    and would be unguessable here. It also needs ``start_listening`` left on (the default), the
    opposite of ``make_opponent``, because nothing else is driving it.
    """
    cfg = ExperimentConfig.from_yaml(run_dir / "config.yaml")
    account = AccountConfiguration(username, None)

    kwargs = dict(
        battle_format=cfg.battle_format,
        account_configuration=account,
        max_concurrent_battles=1,
    )

    if cfg.algo == "tabular_q":
        agent = TabularQAgent.load(run_dir / "qtable.pkl")
        return TabularQPlayer(agent=agent, **kwargs), cfg

    model = ALGO_CLASSES[cfg.algo].load(run_dir / "model.zip", device="cpu")
    player = PolicyPlayer(
        policy=model.policy,
        encoder=cfg.encoder,
        deterministic=deterministic,
        **kwargs,
    )
    return player, cfg


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/ppo_masked_v1_seed0")
    ap.add_argument("--name", default="PokeRLAgent", help="username to challenge (max 18 chars)")
    ap.add_argument("--games", type=int, default=1, help="how many challenges to accept")
    ap.add_argument("--sampled", action="store_true", help="sample actions instead of argmax")
    ap.add_argument("--from-user", default=None, help="only accept challenges from this user")
    args = ap.parse_args()

    if len(args.name) > 18:
        raise SystemExit("Showdown caps usernames at 18 characters")

    run_dir = Path(args.run)
    if not (run_dir / "model.zip").exists() and not (run_dir / "qtable.pkl").exists():
        raise SystemExit(f"no trained model in {run_dir}")

    player, cfg = build_player(run_dir, args.name, not args.sampled)

    print(f"agent      : {run_dir.name}")
    print(f"encoder    : {cfg.encoder}   format: {cfg.battle_format}")
    print(f"actions    : {'sampled' if args.sampled else 'deterministic'}")
    print()
    print(f"  Challenge  {args.name}  to a {cfg.battle_format} in the Showdown client.")
    print(f"  Waiting for {args.games} challenge(s)...")
    print()

    await player.accept_challenges(args.from_user, args.games)

    won = player.n_won_battles
    print(f"\ndone: the agent won {won} of {player.n_finished_battles} battle(s)")


if __name__ == "__main__":
    asyncio.run(main())
