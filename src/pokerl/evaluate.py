"""Evaluation.

Two modes, both writing CSV so that every figure regenerates from committed data alone:

``--run``        evaluate one trained agent against the three baseline opponents
``--cross``      run poke-env's ``cross_evaluate`` over several trained agents plus the
                 baselines, producing the full pairwise win-rate matrix

The evaluation protocol is deliberately identical for every agent: same battle count, same
opponent panel, deterministic action selection. Anything that varies between arms other than
the trained policy would confound the comparison.

Usage:
    python -m pokerl.evaluate --run results/ppo_masked_v0_seed0
    python -m pokerl.evaluate --agent random --opponent random --n 10
    python -m pokerl.evaluate --cross results/ppo_masked_v0_seed0 results/dqn_masked_v0_seed0
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
import uuid
from pathlib import Path

from poke_env import AccountConfiguration
from poke_env.player import Player, cross_evaluate
from stable_baselines3 import A2C, PPO

from pokerl.agents.policy_player import PolicyPlayer
from pokerl.agents.tabular_q import TabularQAgent, TabularQPlayer
from pokerl.config import RESULTS_DIR, ExperimentConfig
from pokerl.policies.masked_dqn import MaskedDQN
from pokerl.utils.opponents import OPPONENTS

ALGO_CLASSES = {
    "ppo_masked": PPO,
    "ppo_unmasked": PPO,
    "a2c_masked": A2C,
    "dqn_masked": MaskedDQN,
}


def _unique_account(prefix: str) -> AccountConfiguration:
    """Mint a username no other process can already be holding.

    Without this, poke-env derives a name from the class -- "PolicyPlayer 1", "RandomPlayer 1"
    -- which is unique within one process but identical across processes. Evaluating several
    runs concurrently then has every process claim the same names; the server bounces the
    duplicates and the battles never start, so the run hangs with no error and no battles.

    uuid4 draws from os.urandom and so is immune to ``random.seed``, for the same reason
    ``_unique_accounts`` in env/battle_env.py uses it: reproducibility must cover the agent's
    decisions, not its network identity. Showdown caps usernames at 18 characters.
    """
    return AccountConfiguration(f"{prefix}{uuid.uuid4().hex[:12]}", None)


def load_agent(run_dir: str | Path, **player_kwargs) -> tuple[Player, ExperimentConfig]:
    """Reconstruct a trained agent as a poke-env Player from its run directory."""
    run_dir = Path(run_dir)
    cfg = ExperimentConfig.from_yaml(run_dir / "config.yaml")
    player_kwargs.setdefault("battle_format", cfg.battle_format)
    player_kwargs.setdefault("max_concurrent_battles", 10)
    player_kwargs.setdefault("account_configuration", _unique_account("ag"))

    if cfg.algo == "tabular_q":
        agent = TabularQAgent.load(run_dir / "qtable.pkl")
        return TabularQPlayer(agent=agent, **player_kwargs), cfg

    model = ALGO_CLASSES[cfg.algo].load(run_dir / "model.zip", device=cfg.device)
    player = PolicyPlayer(policy=model.policy, encoder=cfg.encoder, **player_kwargs)
    return player, cfg


def evaluate_run(run_dir: str | Path, n_battles: int | None = None) -> dict[str, float]:
    """Win rate of one trained agent against each baseline opponent."""
    run_dir = Path(run_dir)
    agent, cfg = load_agent(run_dir)
    n = n_battles or cfg.eval_battles

    results: dict[str, float] = {}
    for name, cls in OPPONENTS.items():
        opponent = cls(
            battle_format=cfg.battle_format,
            max_concurrent_battles=10,
            account_configuration=_unique_account("op"),
        )
        asyncio.run(agent.battle_against(opponent, n_battles=n))
        # Counters accumulate across opponents, so read the win rate off the opponent,
        # which is freshly constructed each time.
        win_rate = 100 * opponent.n_lost_battles / max(opponent.n_finished_battles, 1)
        results[name] = round(win_rate, 1)
        print(f"  vs {name:<10} {win_rate:5.1f}%  ({opponent.n_finished_battles} battles)")

    out = run_dir / "eval.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "algo", "encoder", "reward", "train_opponent", "seed",
                         "opponent", "win_rate", "n_battles"])
        for opponent_name, rate in results.items():
            writer.writerow([cfg.run_id, cfg.algo, cfg.encoder, cfg.reward,
                             cfg.train_opponent, cfg.seed, opponent_name, rate, n])
    print(f"  -> {out}")
    return results


def evaluate_cross(run_dirs: list[str], n_battles: int = 100) -> None:
    """Pairwise win-rate matrix among trained agents and the baselines."""
    players: list[Player] = []
    battle_format = "gen9randombattle"

    # poke-env derives usernames from the account configuration, so every trained agent
    # would otherwise appear as "PolicyPlayer 1..N" and the matrix would be unreadable.
    # Usernames are capped at 18 characters by Showdown and must stay unique, so rather than
    # fight that limit we keep a username -> configuration mapping and relabel afterwards.
    label_of: dict[str, str] = {}

    for run_dir in run_dirs:
        player, cfg = load_agent(run_dir, max_concurrent_battles=5)
        battle_format = cfg.battle_format
        label_of[player.username] = cfg.run_id
        players.append(player)

    for name, cls in OPPONENTS.items():
        player = cls(
            battle_format=battle_format,
            max_concurrent_battles=5,
            account_configuration=_unique_account("op"),
        )
        label_of[player.username] = name
        players.append(player)

    matrix = asyncio.run(cross_evaluate(players, n_challenges=n_battles))

    usernames = [p.username for p in players]
    labels = [label_of[u] for u in usernames]
    out = RESULTS_DIR / "cross_evaluation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["player"] + labels)
        for row_user, row_label in zip(usernames, labels):
            row = matrix.get(row_user, {})
            writer.writerow(
                [row_label]
                + [("" if row.get(c) is None else round(100 * row[c], 1)) for c in usernames]
            )
    print(f"Cross-evaluation matrix -> {out}")


def evaluate_baseline(agent: str, opponent: str, n: int, battle_format: str) -> None:
    """Baseline-vs-baseline sanity check; used by the smoke test."""
    a = OPPONENTS[agent](battle_format=battle_format, max_concurrent_battles=5)
    b = OPPONENTS[opponent](battle_format=battle_format, max_concurrent_battles=5)
    asyncio.run(a.battle_against(b, n_battles=n))
    print(f"{agent} vs {opponent}: {100 * a.n_won_battles / n:.1f}% over {n} battles")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PokeRL agents.")
    parser.add_argument("--run", help="run directory of a trained agent")
    parser.add_argument("--cross", nargs="+", help="run directories for a cross-evaluation")
    parser.add_argument("--agent", choices=sorted(OPPONENTS), help="baseline-vs-baseline check")
    parser.add_argument("--opponent", choices=sorted(OPPONENTS), default="random")
    parser.add_argument("--n", type=int, default=None, help="number of battles")
    parser.add_argument("--format", default="gen9randombattle")
    args = parser.parse_args()

    if args.run:
        print(f"=== evaluating {args.run} ===")
        evaluate_run(args.run, args.n)
    elif args.cross:
        evaluate_cross(args.cross, args.n or 100)
    elif args.agent:
        evaluate_baseline(args.agent, args.opponent, args.n or 100, args.format)
    else:
        parser.error("one of --run, --cross, or --agent is required")


if __name__ == "__main__":
    main()
