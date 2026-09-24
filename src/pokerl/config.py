"""Experiment configuration.

One dataclass describes a single cell of the experiment matrix. Configs are loaded from YAML
so that a run is fully described by a file that can be committed alongside its results — which
is what makes the study reproducible rather than merely repeatable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIG_DIR = PROJECT_ROOT / "configs"


@dataclass
class ExperimentConfig:
    name: str
    algo: str = "ppo_masked"          # ppo_masked | ppo_unmasked | dqn_masked | a2c_masked | tabular_q
    encoder: str = "v0"               # axis B
    reward: str = "shaped"            # axis C
    train_opponent: str = "heuristic"  # axis D
    battle_format: str = "gen9randombattle"

    total_timesteps: int = 100_000
    num_envs: int = 6                 # 8 physical cores, leaving headroom for the Node server
    seed: int = 0
    device: str = "cpu"               # the simulator is the bottleneck; see README

    # Intra-op threads for torch. 0 leaves torch at its default (every core).
    #
    # Measured during the first matrix run: the learner process burned ~709% CPU (seven
    # cores) doing gradient updates on a 64x64 MLP, while the six env workers sat at ~15% of
    # a core each, blocked on websocket round-trips. For tensors this small, thread
    # synchronisation costs more than the matmul, so torch's default parallelism is pure
    # waste — it starves the workers that are actually on the critical path.
    #
    # Measured on the corrective re-runs, idle machine, same 300k steps:
    #   ppo_masked_v0  10.5 min  vs ~13 min at torch's default  (19% faster)
    #   ppo_masked_v1  12.0 min  vs ~13 min
    #   tabular_q_v0   14.0 min  vs ~13 min  (single-env, barely touches torch)
    # So 1 is the default. Runs completed before this was measured record torch_threads: 0
    # in their own saved config.yaml, so their wall-clock figures remain interpretable.
    torch_threads: int = 1

    # Shared SB3 hyperparameters. Only those an algorithm accepts are forwarded.
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 128
    n_steps: int = 512                # per env; PPO/A2C rollout length
    ent_coef: float = 0.01
    net_arch: list[int] = field(default_factory=lambda: [64, 64])

    # DQN-specific
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    target_update_interval: int = 500
    exploration_fraction: float = 0.3
    exploration_final_eps: float = 0.05

    eval_battles: int = 500

    @property
    def run_id(self) -> str:
        return f"{self.name}_seed{self.seed}"

    @property
    def run_dir(self) -> Path:
        return RESULTS_DIR / self.run_id

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(asdict(self), sort_keys=False), encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> ExperimentConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        data.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise KeyError(f"Unknown config keys in {path}: {sorted(unknown)}")
        return cls(**data)
