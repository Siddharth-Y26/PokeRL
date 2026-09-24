"""Measure what a trained policy actually does about switching, on real battle states.

Exploration for the coach feature probed the policy with hand-built and randomly-sampled
observations. Both are out of distribution -- a synthetic two-Pokemon team produced
``P(switch) = 0`` while uniformly-random feature vectors produced 35-41%, and neither number
describes real play. This script settles it by logging the policy's full masked action
distribution at every decision of N genuine battles.

It answers the question the coach is built on: does the agent have an opinion about switching
out a low-HP Pokemon, and does that opinion move with HP?

Usage:
    python scripts/diagnose_policy.py --run results/ppo_masked_v1_seed0 --battles 200
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

from poke_env.battle import AbstractBattle  # noqa: E402
from poke_env.environment import SinglesEnv  # noqa: E402
from poke_env.player import BattleOrder  # noqa: E402

from pokerl.agents.policy_player import PolicyPlayer  # noqa: E402
from pokerl.config import ExperimentConfig  # noqa: E402
from pokerl.evaluate import ALGO_CLASSES, _unique_account  # noqa: E402
from pokerl.utils.opponents import OPPONENTS  # noqa: E402

# Index of "our active Pokemon's HP fraction" within each encoder's flat vector. Used for the
# counterfactual probe. v0 lays out base_power(0-3), multiplier(4-7), fainted(8-9), hp(10-11);
# v1 leads with our active HP. Keep in step with pokerl.env.encoders.
OWN_HP_INDEX = {"v0": 10, "v1": 0}


class DiagnosticPlayer(PolicyPlayer):
    """PolicyPlayer that records the masked distribution and critic value at every turn."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows: list[dict] = []

    def _forward(self, obs_np: np.ndarray, mask: np.ndarray):
        """Masked action probabilities and V(s) for one state."""
        tensor, _ = self.policy.obs_to_tensor(
            {"observation": obs_np[None, :], "action_mask": mask[None, :]}
        )
        with torch.no_grad():
            # get_distribution is mask-aware as of the coach work; without that override this
            # would silently reuse the previous call's mask.
            probs = self.policy.get_distribution(tensor).distribution.probs[0].numpy()
            value = float(self.policy.predict_values(tensor)[0].item())
        return probs, value

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        mask = np.array(SinglesEnv.get_action_mask(battle), dtype=np.int8)
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            return super().choose_move(battle)

        obs = self.encoder(battle)
        probs, value = self._forward(obs, mask)

        switch_actions = [a for a in legal if a < 6]
        move_actions = [a for a in legal if a >= 6]
        top = int(legal[np.argmax(probs[legal])])

        # Counterfactual: would it still choose this at full HP? A flip means the low HP -- not
        # the type matchup -- is what is driving the recommendation. This is the explanation the
        # coach is meant to surface, so measure how often it is even available.
        cf_top = top
        idx = OWN_HP_INDEX.get(self.encoder.name)
        if idx is not None and switch_actions:
            cf_obs = obs.copy()
            cf_obs[idx] = 1.0
            cf_probs, _ = self._forward(cf_obs, mask)
            cf_top = int(legal[np.argmax(cf_probs[legal])])

        ordered = np.sort(probs[legal])[::-1]
        self.rows.append(
            {
                "turn": battle.turn,
                "own_hp": round(
                    float(battle.active_pokemon.current_hp_fraction or 0.0)
                    if battle.active_pokemon else 0.0, 3),
                "opp_hp": round(
                    float(battle.opponent_active_pokemon.current_hp_fraction or 0.0)
                    if battle.opponent_active_pokemon else 0.0, 3),
                "n_legal_switch": len(switch_actions),
                "n_legal_move": len(move_actions),
                "p_switch": round(float(probs[switch_actions].sum()), 4) if switch_actions else 0.0,
                "p_attack": round(float(probs[move_actions].sum()), 4) if move_actions else 0.0,
                "top_action": top,
                "top_is_switch": int(top < 6),
                "top_prob": round(float(probs[top]), 4),
                "margin": round(float(ordered[0] - ordered[1]), 4) if len(ordered) > 1 else 1.0,
                "value": round(value, 3),
                "cf_fullhp_top": cf_top,
                "cf_flips": int(cf_top != top),
            }
        )
        return super().choose_move(battle)


def summarise(rows: list[dict]) -> None:
    """Report voluntary switching only.

    After a faint the only legal actions are switches, so counting those as "chose to switch"
    inflates the rate and, because they all occur at 0% HP, manufactures a spurious
    "switches when low" trend. Measured on a 10-battle sample the raw figure was 25.4% with a
    70.7% spike below 20% HP; excluding forced switches it was 7.9% and flat. Only decisions
    where both a move and a switch were legal are a real choice.
    """
    if not rows:
        print("no decisions recorded")
        return

    hp = np.array([r["own_hp"] for r in rows])
    is_switch = np.array([r["top_is_switch"] for r in rows])
    p_switch = np.array([r["p_switch"] for r in rows])
    value = np.array([r["value"] for r in rows])
    flips = np.array([r["cf_flips"] for r in rows])
    n_switch = np.array([r["n_legal_switch"] for r in rows])
    n_move = np.array([r["n_legal_move"] for r in rows])

    genuine = (n_switch > 0) & (n_move > 0)
    forced = n_move == 0

    print(f"\ndecisions: {len(rows)}   forced switches (excluded): {int(forced.sum())}   "
          f"genuine choices: {int(genuine.sum())}")
    if not genuine.any():
        return
    print(f"chose to switch : {is_switch[genuine].mean():.1%} of genuine choices")
    print(f"mean P(switch)  : {p_switch[genuine].mean():.1%}")
    print(f"V(s)            : mean {value.mean():+.2f}  "
          f"range [{value.min():+.2f}, {value.max():+.2f}]")
    print(f"full-HP counterfactual flips the choice: {flips[genuine].mean():.1%}")

    print("\n own HP band       n   chose switch   mean P(switch)")
    edges = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.01]
    for lo, hi in zip(edges, edges[1:]):
        m = genuine & (hp >= lo) & (hp < hi)
        if not m.any():
            continue
        print(f"  {lo:>4.0%}-{min(hi, 1.0):<4.0%} {int(m.sum()):7d}   "
              f"{is_switch[m].mean():11.1%}   {p_switch[m].mean():14.1%}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--battles", type=int, default=200)
    ap.add_argument("--opponent", default="heuristic", choices=sorted(OPPONENTS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg = ExperimentConfig.from_yaml(run_dir / "config.yaml")
    model = ALGO_CLASSES[cfg.algo].load(run_dir / "model.zip", device="cpu")

    # Unique account names, so several arms can be diagnosed at once. Without them poke-env
    # names players after their class and concurrent processes collide on the server, which
    # hangs with no error and no battles rather than failing loudly.
    player = DiagnosticPlayer(
        policy=model.policy,
        encoder=cfg.encoder,
        battle_format=cfg.battle_format,
        max_concurrent_battles=10,
        account_configuration=_unique_account("dg"),
    )
    # Built from OPPONENTS directly rather than via make_opponent, matching evaluate.py.
    # make_opponent defaults to start_listening=False because training drives the opponent
    # through SingleAgentWrapper; battle_against instead needs it connected to the server, and
    # a non-listening opponent simply hangs forever waiting for a battle that never starts.
    opponent = OPPONENTS[args.opponent](
        battle_format=cfg.battle_format,
        max_concurrent_battles=10,
        account_configuration=_unique_account("do"),
    )

    print(f"{run_dir.name}: {args.battles} battles vs {args.opponent} (encoder {cfg.encoder})")
    await player.battle_against(opponent, n_battles=args.battles)
    print(f"win rate: {opponent.n_lost_battles / args.battles:.1%}")

    summarise(player.rows)

    out = Path(args.out) if args.out else run_dir / "diagnostic.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(player.rows[0].keys()))
        writer.writeheader()
        writer.writerows(player.rows)
    print(f"\nper-decision log -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
