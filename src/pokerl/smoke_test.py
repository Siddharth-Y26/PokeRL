"""Pipeline smoke test.

Verifies, in order, that:
  1. the local Showdown server accepts connections and runs battles to completion,
  2. both encoders produce vectors of the declared size with no NaNs and in-bounds values,
  3. the Gymnasium env resets and steps with a legal masked action.

If this fails, nothing downstream is worth debugging.

Usage:  python -m pokerl.smoke_test
"""

from __future__ import annotations

import asyncio

import numpy as np
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

from pokerl.env.battle_env import DEFAULT_FORMAT, make_env
from pokerl.env.encoders import ENCODERS, get_encoder


def check_battles(n: int = 10) -> None:
    print(f"[1/3] Running {n} RandomPlayer vs RandomPlayer battles...")
    p1 = RandomPlayer(battle_format=DEFAULT_FORMAT, max_concurrent_battles=5)
    p2 = RandomPlayer(battle_format=DEFAULT_FORMAT, max_concurrent_battles=5)
    asyncio.run(p1.battle_against(p2, n_battles=n))

    assert p1.n_finished_battles == n, f"only {p1.n_finished_battles}/{n} battles finished"
    rate = 100 * p1.n_won_battles / n
    print(f"      OK: {n} battles finished, p1 won {rate:.0f}% (expect ~50%)")


def check_encoders() -> None:
    print("[2/3] Checking encoders against a live battle...")
    from poke_env.player import Player

    captured: dict[str, object] = {}

    class Capture(RandomPlayer):
        def choose_move(self, battle):
            captured.setdefault("battle", battle)
            return super().choose_move(battle)

    agent = Capture(battle_format=DEFAULT_FORMAT)
    opp = RandomPlayer(battle_format=DEFAULT_FORMAT)
    asyncio.run(agent.battle_against(opp, n_battles=1))

    battle = captured.get("battle")
    assert battle is not None, "no battle state captured"

    for name in ENCODERS:
        enc = get_encoder(name)
        obs = enc(battle)
        assert obs.shape == (enc.size,), f"{name}: got {obs.shape}, declared {enc.size}"
        assert obs.dtype == np.float32, f"{name}: dtype is {obs.dtype}, expected float32"
        assert not np.isnan(obs).any(), f"{name}: produced NaNs"
        assert (obs >= enc.low).all() and (obs <= enc.high).all(), (
            f"{name}: values outside [{enc.low}, {enc.high}] "
            f"(min={obs.min():.2f}, max={obs.max():.2f})"
        )
        print(f"      OK: {name} -> {enc.size} features, range "
              f"[{obs.min():.2f}, {obs.max():.2f}]")


def check_env() -> None:
    print("[3/3] Checking Gymnasium env reset/step...")
    for encoder in ENCODERS:
        env = make_env(
            opponent=RandomPlayer(battle_format=DEFAULT_FORMAT, start_listening=False),
            encoder=encoder,
        )
        obs, _ = env.reset()
        assert "observation" in obs and "action_mask" in obs, f"bad obs keys: {obs.keys()}"

        mask = np.asarray(obs["action_mask"])
        assert mask.sum() > 0, "reset produced no legal actions"

        # Take a legal action only; an illegal one would raise under strict mode.
        action = int(np.flatnonzero(mask)[0])
        obs, reward, terminated, truncated, _ = env.step(action)
        assert np.isfinite(reward), f"non-finite reward {reward}"
        env.close()
        print(f"      OK: {encoder} env stepped, reward={reward:+.3f}, "
              f"{mask.sum()} legal actions")


def main() -> None:
    check_battles()
    check_encoders()
    check_env()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
