"""Training entrypoint.

One config in, one trained agent plus TensorBoard logs out. Every algorithm in the study is
reached through this file so that the parts that must be identical across arms — env, reward,
opponent, seed, timestep budget — cannot drift apart.

Usage:
    python -m pokerl.train --config configs/exp/ppo_masked_v0.yaml --seed 0

IMPORTANT (Windows): SubprocVecEnv spawns processes, so all work must sit behind the
``if __name__ == "__main__"`` guard at the bottom. Without it, each child re-imports this
module and spawns its own children, recursively.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from pokerl.agents.tabular_q import TabularQAgent, discretise
from pokerl.config import ExperimentConfig
from pokerl.env.battle_env import make_env
from pokerl.policies.masked_ac import MaskedActorCriticPolicy, UnmaskedActorCriticPolicy
from pokerl.policies.masked_dqn import MaskedDQN, MaskedDQNPolicy
from pokerl.utils.opponents import DEFAULT_SNAPSHOT, make_opponent
from pokerl.utils.seeding import set_global_seed, set_torch_threads

# algo name -> (SB3 class, policy class, whether the env should reject illegal actions)
#
# strict is False everywhere. It was originally True for the masked algorithms, to catch bugs
# in our own action encoding — but the failure it actually produced came from the *opponent*:
# SingleAgentWrapper.step converts the opponent's chosen order back to an action via
# ``[m.id for m in mvs].index(order.order.id)``, and SimpleHeuristicsPlayer can select a move
# that is momentarily absent from ``available_moves`` (observed: 'hydropump' is not in list).
# That is a poke-env inconsistency we neither cause nor control, and under strict=True it
# killed a 13-minute run outright. With strict=False poke-env substitutes a default order and
# the battle continues. Our own actions are mask-verified, so strict bought little and cost
# whole runs.
SB3_ALGOS = {
    "ppo_masked": (PPO, MaskedActorCriticPolicy, False),
    "ppo_unmasked": (PPO, UnmaskedActorCriticPolicy, False),
    "a2c_masked": (A2C, MaskedActorCriticPolicy, False),
    "dqn_masked": (MaskedDQN, MaskedDQNPolicy, False),
}


def _env_factory(cfg: ExperimentConfig, strict: bool, rank: int):
    """Build a picklable thunk for one training environment."""

    def _init():
        opponent = make_opponent(cfg.train_opponent, cfg.battle_format)
        return make_env(
            opponent=opponent,
            encoder=cfg.encoder,
            reward=cfg.reward,
            battle_format=cfg.battle_format,
            monitor_path=str(cfg.run_dir / "monitor" / f"env{rank}"),
            strict=strict,
        )

    return _init


class SnapshotCallback(BaseCallback):
    """Periodically publish the learner's current policy for self-play opponents to pick up.

    ``SnapshotOpponent`` (pokerl.utils.opponents) loads the newest snapshot in this directory,
    so writing on a schedule is what turns ``train_opponent: snapshot`` into self-play against
    a lagging copy of the agent rather than against a frozen one.

    Each save gets its own zero-padded filename rather than overwriting a fixed path. On
    Windows, replacing a file that a SubprocVecEnv worker currently has open raises
    PermissionError, and workers all reload as soon as a new snapshot appears — so an
    in-place write is a race that kills the run. Old snapshots are pruned best-effort; a
    failed unlink means a reader still holds it, and it will go on the next pass.
    """

    KEEP = 3

    def __init__(self, path: Path, save_freq: int):
        super().__init__()
        self.dir = Path(path)
        self.save_freq = max(save_freq, 1)

    def _on_training_start(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True
        self.model.save(self.dir / f"opponent_{self.num_timesteps:09d}.zip")
        stale = sorted(self.dir.glob("opponent_*.zip"))[: -self.KEEP]
        for old_snapshot in stale:
            try:
                old_snapshot.unlink()
            except OSError:
                pass
        return True


def train_sb3(cfg: ExperimentConfig) -> Path:
    algo_cls, policy_cls, strict = SB3_ALGOS[cfg.algo]
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run_dir / "monitor").mkdir(exist_ok=True)

    vec_cls = SubprocVecEnv if cfg.num_envs > 1 else DummyVecEnv
    env = vec_cls([_env_factory(cfg, strict, i) for i in range(cfg.num_envs)])

    common = dict(
        policy=policy_cls,
        env=env,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        seed=cfg.seed,
        device=cfg.device,
        verbose=1,
        tensorboard_log=str(cfg.run_dir / "tensorboard"),
        policy_kwargs={"net_arch": cfg.net_arch},
    )

    if cfg.algo.startswith("ppo"):
        model = algo_cls(
            **common,
            n_steps=max(cfg.n_steps // cfg.num_envs, 1),
            batch_size=cfg.batch_size,
            ent_coef=cfg.ent_coef,
        )
    elif cfg.algo.startswith("a2c"):
        model = algo_cls(
            **common,
            n_steps=max(cfg.n_steps // cfg.num_envs, 1),
            ent_coef=cfg.ent_coef,
        )
    else:  # DQN
        model = algo_cls(
            **common,
            buffer_size=cfg.buffer_size,
            batch_size=cfg.batch_size,
            learning_starts=cfg.learning_starts,
            target_update_interval=cfg.target_update_interval,
            exploration_fraction=cfg.exploration_fraction,
            exploration_final_eps=cfg.exploration_final_eps,
        )

    checkpoint = CheckpointCallback(
        save_freq=max(cfg.total_timesteps // (10 * cfg.num_envs), 1),
        save_path=str(cfg.run_dir / "checkpoints"),
        name_prefix=cfg.algo,
    )
    callbacks: list[BaseCallback] = [checkpoint]

    if cfg.train_opponent == "snapshot" or cfg.train_opponent.startswith("snapshot:"):
        _, _, snap = cfg.train_opponent.partition(":")
        # Refreshed 20x over the run. Too often and the opponent churns faster than the
        # learner can adapt to it; too rarely and self-play degenerates towards a frozen
        # opponent, which for this agent would be weaker than the scripted heuristic.
        callbacks.append(
            SnapshotCallback(
                path=Path(snap) if snap else DEFAULT_SNAPSHOT,
                save_freq=max(cfg.total_timesteps // (20 * cfg.num_envs), 1),
            )
        )

    start = time.time()
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=CallbackList(callbacks) if len(callbacks) > 1 else checkpoint,
        progress_bar=False,
    )
    elapsed = time.time() - start
    env.close()

    model_path = cfg.run_dir / "model.zip"
    model.save(model_path)
    _write_metadata(cfg, elapsed)
    print(f"\nSaved model to {model_path} ({elapsed / 60:.1f} min)")
    return model_path


def _reset_when_settled(env, timeout: float = 5.0, poll: float = 0.02):
    """Wait for the previous battle to be fully closed out, then reset once.

    ``step`` reports ``terminated`` as soon as the env observes the battle end, but each
    player only marks ``battle.finished`` once the final protocol message has been handled on
    the websocket thread. Resetting inside that window raises "Can not reset player's battles
    while they are still running".

    The wait must happen *before* ``reset``, not as a retry around it: ``PokeEnv.reset``
    forfeits and drains both agents' battle queues before the call that raises, so a second
    attempt re-enters with those queues already consumed and can hang. Retrying is actively
    worse than failing.
    """
    inner = env.unwrapped
    poke_env = getattr(inner, "env", None)
    if poke_env is not None and poke_env.battle1 is not None:
        deadline = time.time() + timeout
        while not poke_env.battle1.finished and time.time() < deadline:
            time.sleep(poll)
    return env.reset()


def train_tabular(cfg: ExperimentConfig) -> Path:
    """Tabular Q-learning against the same env, on the same timestep budget."""
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    opponent = make_opponent(cfg.train_opponent, cfg.battle_format)
    env = make_env(
        opponent=opponent,
        encoder=cfg.encoder,
        reward=cfg.reward,
        battle_format=cfg.battle_format,
        monitor_path=str(cfg.run_dir / "monitor" / "env0"),
    )
    n_actions = env.action_space.n
    agent = TabularQAgent(
        n_actions=n_actions,
        learning_rate=cfg.learning_rate if cfg.learning_rate > 1e-3 else 0.1,
        gamma=cfg.gamma,
        exploration_fraction=cfg.exploration_fraction,
        epsilon_end=cfg.exploration_final_eps,
        seed=cfg.seed,
    )

    # The tabular agent discretises the raw battle rather than the encoded vector, so it
    # reaches through the wrappers to the underlying poke-env battle object.
    def current_battle():
        inner = env.unwrapped
        return inner.env.battle1 if hasattr(inner, "env") else None

    start = time.time()
    steps, episodes, wins = 0, 0, 0
    obs, _ = _reset_when_settled(env)
    battle = current_battle()
    state = discretise(battle) if battle else (0,) * 6

    while steps < cfg.total_timesteps:
        eps = agent.epsilon_at(steps / cfg.total_timesteps)
        mask = np.asarray(obs["action_mask"])
        action = agent.act(state, mask, eps)

        # poke-env's action_to_order calls .item() on the action, so it must be a numpy
        # scalar rather than a Python int. SB3 supplies numpy actions; this loop must too.
        obs, reward, terminated, truncated, _ = env.step(np.int64(action))
        steps += 1

        battle = current_battle()
        next_state = discretise(battle) if battle else None
        next_mask = np.asarray(obs["action_mask"])

        if terminated or truncated:
            agent.update(state, action, reward, None, None)
            episodes += 1
            if battle is not None and battle.won:
                wins += 1
            if episodes % 50 == 0:
                print(
                    f"  step {steps}/{cfg.total_timesteps} | episodes {episodes} | "
                    f"win rate {100 * wins / episodes:.0f}% | eps {eps:.2f} | "
                    f"states {len(agent.q)}"
                )
            obs, _ = _reset_when_settled(env)
            battle = current_battle()
            state = discretise(battle) if battle else (0,) * 6
        else:
            agent.update(state, action, reward, next_state, next_mask)
            state = next_state if next_state is not None else state

    elapsed = time.time() - start
    env.close()

    model_path = cfg.run_dir / "qtable.pkl"
    agent.save(model_path)
    _write_metadata(cfg, elapsed, extra={"states_visited": len(agent.q), "episodes": episodes})
    print(f"\nSaved Q-table to {model_path} ({len(agent.q)} states, {elapsed / 60:.1f} min)")
    return model_path


def _write_metadata(cfg: ExperimentConfig, elapsed: float, extra: dict | None = None) -> None:
    cfg.save(cfg.run_dir / "config.yaml")
    meta = {"run_id": cfg.run_id, "train_seconds": round(elapsed, 1), **(extra or {})}
    (cfg.run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PokeRL agent.")
    parser.add_argument("--config", required=True, help="path to an experiment YAML")
    parser.add_argument("--seed", type=int, default=None, help="override the config seed")
    parser.add_argument("--timesteps", type=int, default=None, help="override total_timesteps")
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="override torch intra-op threads (1 is the tuned value; 0 = torch default)",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(
        args.config, seed=args.seed, total_timesteps=args.timesteps
    )
    # Before any tensor work, and before workers are spawned so children inherit the cap.
    set_torch_threads(args.torch_threads if args.torch_threads is not None else cfg.torch_threads)
    set_global_seed(cfg.seed)

    print(f"=== {cfg.run_id} ===")
    print(f"algo={cfg.algo} encoder={cfg.encoder} reward={cfg.reward} "
          f"opponent={cfg.train_opponent} steps={cfg.total_timesteps}")

    if cfg.algo == "tabular_q":
        train_tabular(cfg)
    elif cfg.algo in SB3_ALGOS:
        train_sb3(cfg)
    else:
        raise KeyError(
            f"Unknown algo {cfg.algo!r}. Available: {sorted(SB3_ALGOS) + ['tabular_q']}"
        )


if __name__ == "__main__":
    main()
