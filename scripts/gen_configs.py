"""Generate the experiment-matrix config files.

The matrix is defined here rather than hand-written as ~14 near-identical YAML files, so that
the "vary exactly one axis from the reference cell" property is enforced by construction
instead of by care. Run this once; the generated files are committed alongside results.

Usage:  python scripts/gen_configs.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokerl.config import CONFIG_DIR, ExperimentConfig  # noqa: E402

# The reference cell. Every other cell differs from this in exactly one field.
REFERENCE = ExperimentConfig(
    name="ppo_masked_v0",
    algo="ppo_masked",
    encoder="v0",
    reward="shaped",
    train_opponent="heuristic",
    total_timesteps=300_000,
    num_envs=6,
    n_steps=3072,
    batch_size=128,
    eval_battles=500,
)

# axis -> {cell name: field overrides}
MATRIX: dict[str, dict[str, dict]] = {
    # Axis A: algorithm (the main results table)
    "algo": {
        "ppo_unmasked_v0": {"algo": "ppo_unmasked"},
        "a2c_masked_v0": {"algo": "a2c_masked"},
        "dqn_masked_v0": {"algo": "dqn_masked", "learning_rate": 1e-4, "num_envs": 4},
        "tabular_q_v0": {"algo": "tabular_q", "num_envs": 1, "learning_rate": 0.1},
    },
    # Axis B: state encoding. v1 is ~24x wider, so it gets a wider network and the GPU.
    "encoder": {
        "ppo_masked_v1": {"encoder": "v1", "net_arch": [256, 256], "device": "cuda"},
    },
    # Axis C: reward shaping
    "reward": {
        "ppo_masked_v0_sparse": {"reward": "sparse"},
        "ppo_masked_v0_aggressive": {"reward": "aggressive"},
    },
    # Axis D: training opponent
    "train_opponent": {
        "ppo_masked_v0_vs_random": {"train_opponent": "random"},
        "ppo_masked_v0_vs_maxpower": {"train_opponent": "max_power"},
        "ppo_masked_v0_vs_mixed": {"train_opponent": "mixed"},
    },
}


# --------------------------------------------------------------------------------------
# Coach arms — deliberately NOT part of the study matrix
# --------------------------------------------------------------------------------------
# These exist to produce an agent strong enough to give advice, not to answer a research
# question, and they vary several fields at once. Folding them into MATRIX would break the
# "exactly one field differs from the reference cell" property that makes the study's
# comparisons attributable, and would change the reported cell count. They live in
# configs/coach/ and the published 300k results stay exactly as they are.
#
# All three use the v1 encoder. v1 scores lower than v0 (26.5% vs 31.1% against
# SimpleHeuristics) but it is the only encoder that sees the bench at all — v0's 12 features
# carry no teammate HP, types or matchups — so it is the only one whose switch advice can be
# grounded in its own input.
#
# Baseline to beat, measured over 3,372 genuine decisions of the 300k v1 agent
# (scripts/diagnose_policy.py): it switches voluntarily 9.0% of the time, and its switch rate
# is flat in its own HP (5.8-11.8% across every HP band). The judgement the coach is meant to
# teach is therefore not currently represented in the agent's behaviour.
#
# device is cpu because this machine has the CPU-only torch wheel; the study's v1 config asks
# for cuda and SB3 silently falls back, so recording cpu is simply honest.
COACH_BASE = replace(
    REFERENCE,
    name="ppo_v1_long",
    encoder="v1",
    net_arch=[256, 256],
    total_timesteps=5_000_000,
    # 6, not 8. Three concurrent runs at 8 envs exhausted this machine's 15.4 GB: 24 env
    # subprocesses each carry their own torch and poke-env, and one arm died on a 3.39 MiB
    # allocation. All arms must share this value -- train.py passes n_steps // num_envs, so a
    # different count would change the rollout length and make the arms incomparable.
    num_envs=6,
    device="cpu",
)

COACH_ARMS: dict[str, dict] = {
    # Budget alone. Both encoders were still improving at 300k, so the first question is
    # whether v1 closes the gap given 16x the steps.
    "ppo_v1_long": {},
    # Self-play against a lagging copy of the learner. Of the scripted opponents only
    # SimpleHeuristicsPlayer switches at all, so there is little switching to model.
    "ppo_v1_selfplay": {"train_opponent": "snapshot"},
    # Higher entropy bonus, guarding against early collapse onto a narrow action set.
    "ppo_v1_entropy": {"ent_coef": 0.03},
}


def write_coach_arms() -> None:
    coach_dir = CONFIG_DIR / "coach"
    coach_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, overrides in COACH_ARMS.items():
        cfg = replace(COACH_BASE, name=name, **overrides)
        cfg.save(coach_dir / f"{name}.yaml")
        written.append(cfg)

    print()
    print(f"Wrote {len(written)} coach arms to {coach_dir}:")
    for cfg in written:
        print(f"  {cfg.name:<20} enc={cfg.encoder} steps={cfg.total_timesteps:,} "
              f"envs={cfg.num_envs} opp={cfg.train_opponent} ent_coef={cfg.ent_coef}")


def main() -> None:
    out_dir = CONFIG_DIR / "exp"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = [REFERENCE]
    REFERENCE.save(out_dir / f"{REFERENCE.name}.yaml")

    for axis, cells in MATRIX.items():
        for name, overrides in cells.items():
            cfg = replace(REFERENCE, name=name, **overrides)
            cfg.save(out_dir / f"{name}.yaml")
            written.append(cfg)

    print(f"Wrote {len(written)} configs to {out_dir}:")
    for cfg in written:
        print(f"  {cfg.name:<28} algo={cfg.algo:<13} enc={cfg.encoder:<3} "
              f"reward={cfg.reward:<10} opp={cfg.train_opponent}")
    print(f"\n{len(written)} cells x 3 seeds = {len(written) * 3} runs")


if __name__ == "__main__":
    main()
    write_coach_arms()
