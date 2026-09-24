"""How often does the model pick a move that another legal move strictly beats?

Written after the coach recommended Air Slash (Flying, resisted, effective power 56) over Giga
Drain (Grass, super effective, 150) against a Rampardos at 38% HP -- see
report/COACH_FINDINGS.md. One position proves nothing; this measures the rate.

"Effective power" is base power x type multiplier x STAB. That is a crude damage proxy -- it
ignores Attack/Defense stats, items, abilities and secondary effects -- so it is only used to
flag *dominated* choices, where an alternative is better by a wide margin. A narrow gap can
easily be correct play (priority, pivoting, chip, status); a 2x gap on a checkable type matchup
is the error a learner would catch.

Usage:
    python scripts/measure_type_choice.py --run results/ppo_v1_selfplay_seed0 --battles 100
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from poke_env.battle import AbstractBattle, Move  # noqa: E402
from poke_env.data import GenData  # noqa: E402
from poke_env.environment import SinglesEnv  # noqa: E402
from poke_env.player import BattleOrder  # noqa: E402

from pokerl.agents.policy_player import PolicyPlayer  # noqa: E402
from pokerl.config import ExperimentConfig  # noqa: E402
from pokerl.env.encoders import _move_attr  # noqa: E402
from pokerl.evaluate import ALGO_CLASSES, OPPONENTS, _unique_account  # noqa: E402

# A choice counts as dominated only if an alternative is at least this many times stronger.
DOMINATION_RATIO = 1.5


def effective_power(move: Move, battle: AbstractBattle) -> float:
    """base power x type multiplier x STAB. Status moves score 0."""
    opp = battle.opponent_active_pokemon
    me = battle.active_pokemon
    base = _move_attr(move, "base_power", 0) or 0
    if not base or move.type is None or opp is None:
        return 0.0
    chart = GenData.from_gen(battle.gen).type_chart
    multiplier = move.type.damage_multiplier(opp.type_1, opp.type_2, type_chart=chart)
    stab = 1.5 if (me is not None and move.type in me.types) else 1.0
    return float(base) * float(multiplier) * stab


class TypeAuditPlayer(PolicyPlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows: list[dict] = []

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        order = super().choose_move(battle)

        moves = battle.available_moves
        if len(moves) >= 2 and battle.opponent_active_pokemon is not None:
            mask = np.array(SinglesEnv.get_action_mask(battle), dtype=np.int8)
            legal = np.flatnonzero(mask)
            tensors = self.policy.obs_to_tensor(
                {"observation": self.encoder(battle)[None, :], "action_mask": mask[None, :]}
            )[0]
            with torch.no_grad():
                probs = self.policy.get_distribution(tensors).distribution.probs[0].cpu().numpy()

            move_actions = [a for a in legal if 6 <= a < 10]
            if move_actions:
                # Restrict to plain move actions, whose index maps onto available_moves.
                best_action = int(max(move_actions, key=lambda a: probs[a]))
                chosen_idx = best_action - 6
                if chosen_idx < len(moves):
                    powers = [effective_power(m, battle) for m in moves]
                    chosen_power = powers[chosen_idx]
                    best_power = max(powers)
                    best_idx = int(np.argmax(powers))
                    dominated = (
                        best_power > 0
                        and chosen_power < best_power / DOMINATION_RATIO
                    )
                    self.rows.append({
                        "turn": battle.turn,
                        "chosen": moves[chosen_idx].id,
                        "chosen_power": round(chosen_power, 1),
                        "chosen_prob": round(float(probs[best_action]), 3),
                        "best": moves[best_idx].id,
                        "best_power": round(best_power, 1),
                        "ratio": round(best_power / chosen_power, 2) if chosen_power else None,
                        "dominated": int(dominated),
                    })
        return order


def summarise(rows: list[dict], label: str) -> None:
    """Report attacking choices separately from status moves.

    Status moves score 0 effective power, so counting them as "dominated" would brand every
    Swords Dance and Roost an error and inflate the rate (40% vs 18% on the first sample).
    Only a damaging move beaten by a better damaging move is evidence of the failure this
    script exists to measure.
    """
    if not rows:
        print(f"{label}: no comparable decisions")
        return

    chosen = np.array([r["chosen_power"] for r in rows], dtype=float)
    dominated = np.array([r["dominated"] for r in rows])
    attacks = chosen > 0

    print(label)
    print(f"  move choices compared     : {len(rows)}")
    print(f"  status / 0-power choices  : {int((~attacks).sum())}  (excluded -- often correct)")
    if not attacks.any():
        return
    print(f"  attacking choices         : {int(attacks.sum())}")
    print(f"  type-dominated            : {dominated[attacks].mean():.1%} "
          f"({int(dominated[attacks].sum())} of {int(attacks.sum())})")

    ratios = [r["ratio"] for r in rows
              if r["dominated"] and r["ratio"] and r["chosen_power"] > 0]
    if ratios:
        print(f"  better move stronger by   : {np.mean(ratios):.1f}x avg, "
              f"{max(ratios):.1f}x worst")
    big = [r for r in rows
           if r["chosen_power"] > 0 and r["best_power"] >= 2 * r["chosen_power"]]
    print(f"  a >=2x stronger move existed: {len(big)} "
          f"({len(big) / attacks.sum():.1%} of attacks)")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/ppo_v1_selfplay_seed0")
    ap.add_argument("--battles", type=int, default=100)
    ap.add_argument("--opponent", default="heuristic", choices=sorted(OPPONENTS))
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg = ExperimentConfig.from_yaml(run_dir / "config.yaml")
    model = ALGO_CLASSES[cfg.algo].load(run_dir / "model.zip", device="cpu")

    player = TypeAuditPlayer(
        policy=model.policy,
        encoder=cfg.encoder,
        battle_format=cfg.battle_format,
        max_concurrent_battles=10,
        account_configuration=_unique_account("ty"),
    )
    opponent = OPPONENTS[args.opponent](
        battle_format=cfg.battle_format,
        max_concurrent_battles=10,
        account_configuration=_unique_account("to"),
    )

    print(f"{run_dir.name}: {args.battles} battles vs {args.opponent}")
    await player.battle_against(opponent, n_battles=args.battles)
    print()
    summarise(player.rows, run_dir.name)

    out = run_dir / "type_choice.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(player.rows[0].keys()))
        writer.writeheader()
        writer.writerows(player.rows)
    print(f"\nper-decision log -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
