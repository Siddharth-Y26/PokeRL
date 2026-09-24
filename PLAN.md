# PokeRL — A Benchmark Study of RL Algorithms for Pokémon Battles

## Status — September 2026

All eight build-order steps below are done. The study is complete: 11 cells × 3 seeds at 300k
steps, plus the sanity run, evaluated and cross-evaluated. Results are in
[report/RESULTS.md](report/RESULTS.md) and summarised in the [readme](readme.md#results).

Where the build departed from the plan:

| Planned | What happened |
|---|---|
| `v1` with ~250 features | 289 features |
| A2C "optional, if time permits" | Included, 3 seeds |
| `configs/base.yaml`, `utils/logging.py` | Not needed. Defaults live in `ExperimentConfig` ([config.py](src/pokerl/config.py)), and SB3's logger plus `Monitor` CSVs cover logging |
| Masked DQN as a ~30-line `_predict` override | Masking was needed in three places: action selection, the bootstrapped target, and the epsilon/warmup sampling that bypasses `predict()` |
| `strict=False` as a DQN fallback | Every run uses `strict=False`, because `SimpleHeuristicsPlayer` occasionally picks a move that is not in `available_moves` |
| Smoke test via `pokerl.evaluate --agent random` | A dedicated `python -m pokerl.smoke_test` |
| Same seed twice → identical learning curve | Not achievable. Showdown's damage, crit, accuracy and team RNG is not seeded from Python, so only the agent side is seeded |
| GPU for `v1` | PyPI's Windows torch wheel is CPU-only, so `v1` ran on CPU (SB3 falls back automatically) |

Verification outcomes: the sanity gate needed >80% vs Random and scored **96.5%**.
`SimpleHeuristicsPlayer` beat Random **98%** in the cross-evaluation. No learned agent beat
`SimpleHeuristicsPlayer` head to head.

### After the study: live coach (not in the original plan)

A teaching overlay that shows a human what a trained agent would play, and why, during a
live battle. It is described in the [readme](readme.md#live-coaching-mode),
[client/README.md](client/README.md) and [report/COACH_FINDINGS.md](report/COACH_FINDINGS.md).
Work done:

- `pokerl.coach` package: protocol → `Battle` reconstruction (verified 90/90 decision points
  on a captured battle), explanations (attack/switch split, counterfactuals, board facts),
  and a websocket server.
- Browser panel as a userscript and a console build, plus `install_coach_client.py` to serve
  the client from the local server with the panel attached.
- `diagnose_policy.py`: showed the 300k `v1` agent has no switch-vs-HP judgement.
- Three 5M-step `v1` training arms (longer, higher entropy, self-play), kept outside the
  study matrix. Self-play reached 31.2% vs SimpleHeuristics and is the only arm that learned
  switch-or-stay (2.4× HP gradient).
- `measure_type_choice.py`: showed self-play picks a type-dominated attack 23.7% of the time,
  against 12.5% for the 300k agent.

Open: flag type-dominated recommendations in the panel; league training (self-play +
scripted heuristic); calibrate the critic's value into a win probability; 5+ seeds;
recurrent policy.

---

*The rest of this file is the original plan (5 August 2026), kept as written.*

## Context

`d:\PokeRL` started empty (one blank `readme.md`). The goal is an academic project applying
reinforcement learning to Pokémon battles.

**Findings on the two reference repos:**

| Repo | What it actually is | Verdict |
|---|---|---|
| [reddheeraj/PokemonRL](https://github.com/reddheeraj/PokemonRL) | PPO on **Pokémon Red via PyBoy emulator** — pixel observations, curriculum learning to walk out of a house. 4★. | **Not a battle project.** Wrong problem domain. Ignore. |
| [leolellisr/poke_RL](https://github.com/leolellisr/poke_RL) | Battles on Pokémon Showdown via poke-env. Tabular (MC, Q-learning, SARSA(λ)) + deep (DQN, DDQN, PPO, REINFORCE). Backed by *"I Choose You, Reinforcement Learning!"*, SBGames 2024. 17★. | **Right domain, stale code.** Built on Keras-rl / TF1-era APIs and poke-env's removed `EnvPlayer` class. |

poke-env is now at **0.15.0 (April 2026)** and has replaced `EnvPlayer` with a PettingZoo-style
`SinglesEnv` + `SingleAgentWrapper`, plus built-in `get_action_mask()`. Cloning poke_RL means
fighting dependency hell for no benefit.

**Approach:** build fresh on current poke-env, using poke_RL's *experimental design* (algorithm
comparison across environments) as the template for what an academic deliverable looks like.

**Deliverable:** a benchmark study comparing 4+ algorithms across controlled ablations
(state encoding, reward shaping, action masking, training opponent) on `gen9randombattle`,
with win-rate tables, learning curves, and a cross-evaluation matrix.

**Confirmed decisions:** benchmark study (not novel research); local GPU available;
`gen9randombattle` format.

---

## Verified environment

| Component | Status |
|---|---|
| Python | 3.13.2 ✓ (poke-env needs ≥3.10) |
| Node.js | v22.14.0 ✓ (Showdown server) |
| CPU | Ryzen 7 5800H, 8C/16T → `num_envs=6` |
| GPU | RTX 3050 Ti Laptop, 4 GB |
| RAM | 15.4 GB |

> **Honest note on the GPU:** for this workload the bottleneck is the **Node.js Showdown
> simulator**, not the neural network. The official poke-env example explicitly passes
> `device="cpu"` because tiny MLPs are faster on CPU than paying GPU transfer overhead.
> The GPU only earns its keep for the large-encoding (`v1`) runs. **Parallel environments
> matter far more than the GPU** — this is planned around `SubprocVecEnv`, not CUDA.

---

## Canonical reference

The single most important reference is poke-env's own working example — verified current:
`https://github.com/hsahovic/poke-env/blob/master/examples/reinforcement_learning.py`

It already provides, working and correct:
- `ExampleEnv(SinglesEnv)` with `calc_reward` / `embed_battle`
- `MaskedActorCriticPolicy` — action masking by adding `-inf` to illegal logits
- `FeaturesExtractor` — pulls `obs["observation"]` out of the dict obs
- `PolicyPlayer(Player)` — wraps a trained policy back into a poke-env `Player` for evaluation
- `SubprocVecEnv` + `PPO` training loop and `battle_against` evaluation

Notably its 12-feature encoding is *the same one poke_RL used*. **Start from this file** —
it becomes our `v0` baseline, and the study's value is everything built on top of it.

**Reuse rather than reimplement:**
- `poke_env.player.RandomPlayer`, `MaxBasePowerPlayer`, `SimpleHeuristicsPlayer` — the three
  evaluation opponents, no need to write baselines
- `self.reward_computing_helper(battle, fainted_value=, hp_value=, status_value=, victory_value=)`
  — built into `PokeEnv`, computes potential-difference rewards
- `SinglesEnv.get_action_mask(battle)` — legal-action mask, already implemented
- `SinglesEnv.action_to_order()` / `order_to_action()` — action encoding
- `poke_env.player.cross_evaluate` — pairwise win-rate matrix for the results table
- `poke_env.data.GenData.from_gen(gen).type_chart` — type effectiveness

Action space (singles): `0–5` switch, `6–9` move, `10–13` mega, `14–17` z-move,
`18–21` dynamax, `22–25` terastallize. Gen 9 → moves + tera are the live ranges.

---

## Repository layout

```
d:\PokeRL\
├── README.md                     # setup + how to reproduce every result
├── requirements.txt
├── configs/
│   ├── base.yaml                 # format, seeds, eval battle count
│   └── exp/*.yaml                # one file per experiment cell
├── src/pokerl/
│   ├── env/
│   │   ├── battle_env.py         # PokeRLEnv(SinglesEnv) — encoder/reward injected
│   │   ├── encoders.py           # v0 (12-feat) and v1 (rich) encoders
│   │   └── rewards.py            # sparse / shaped / aggressive reward variants
│   ├── policies/
│   │   ├── masked_ac.py          # MaskedActorCriticPolicy (PPO/A2C)
│   │   └── masked_dqn.py         # masked Q-network (see note below)
│   ├── agents/
│   │   ├── policy_player.py      # PolicyPlayer — trained policy → poke-env Player
│   │   └── tabular_q.py          # classical Q-learning on discretised state
│   ├── train.py                  # config-driven entrypoint
│   ├── evaluate.py               # win rates + cross_evaluate matrix → CSV
│   └── utils/{seeding,logging}.py
├── scripts/
│   ├── setup_showdown.ps1        # clone + npm install Showdown
│   ├── start_showdown.ps1        # node pokemon-showdown start --no-security
│   └── run_all.ps1               # every experiment cell × 3 seeds
├── results/                      # CSVs, TensorBoard logs, checkpoints
└── report/figures/               # generated plots
```

`battle_env.py` takes the encoder and reward function as **constructor arguments** so every
ablation is a config change, not a new class. This is the one design decision that keeps the
experiment matrix from exploding into copy-pasted files.

---

## The two encoders (the core contribution)

**`v0` — baseline, 12 features.** Copied verbatim from the poke-env example / poke_RL:
4 move base powers, 4 damage multipliers, fainted counts (both sides), both active HP fractions.
This exists to reproduce prior work and prove the pipeline is sound.

**`v1` — rich, ~250 features.** Where the academic value is:
- *Our active:* HP frac, status one-hot (7), types one-hot (18×2), stat boosts (7, normalised)
- *Opponent active:* same, with explicit "unknown" handling for hidden information
- *Our 4 moves:* base power, accuracy, category one-hot (3), type effectiveness, PP remaining,
  priority, STAB flag
- *Team (6×):* HP frac, fainted flag, types, status
- *Field:* weather one-hot, terrain one-hot, entry hazards per side, trick room, turn count

Every feature normalised to roughly `[-1, 4]` to match the `Box` observation space.

---

## Algorithms benchmarked

| # | Agent | Notes |
|---|---|---|
| 1 | Random / MaxBasePower / SimpleHeuristics | Non-learning references, free from poke-env |
| 2 | Tabular Q-learning | Discretised state, poke_RL-style. Represents classical RL. |
| 3 | DQN | Off-policy value-based |
| 4 | PPO (no masking) | On-policy, illegal actions allowed |
| 5 | **PPO + action masking** | Expected best; the example's `MaskedActorCriticPolicy` |
| 6 | A2C + masking | Optional, if time permits |

> **DQN masking caveat:** SB3's `DQN` has no native action masking, and `sb3-contrib` provides
> `MaskablePPO` but **no maskable DQN**. Plan: subclass `DQNPolicy` and add `-inf` to illegal
> action Q-values in `_predict`, mirroring `MaskedActorCriticPolicy` (~30 lines). If that proves
> fiddly, fall back to `strict=False` (poke-env silently converts illegal actions to a default
> move) and report it as a documented limitation — this is itself a legitimate finding.

---

## Experiment matrix

Four axes, each varied against a fixed default so the study stays tractable:

- **A. Algorithm** — the 6 rows above (main results table)
- **B. State encoding** — `v0` (12) vs `v1` (~250), on masked PPO
- **C. Reward shaping** — sparse (win/loss only) vs shaped (`fainted=2, hp=1, status=0.5, victory=30`)
- **D. Training opponent** — `RandomPlayer` / `MaxBasePowerPlayer` / `SimpleHeuristicsPlayer` / mixed

Every cell: **3 seeds**, report mean ± std. This is what separates a project from a demo.

**Evaluation protocol** (identical for all agents, no exceptions):
- 500 battles vs each of the 3 built-in baselines via `Player.battle_against`
- `cross_evaluate` matrix among all trained agents — agents fight *each other*
- Learning curves from SB3 `Monitor` → TensorBoard
- Fixed evaluation seeds, separate from training seeds

---

## Build order

1. **Setup & smoke test** — `requirements.txt`, Showdown scripts, confirm `RandomPlayer` vs
   `RandomPlayer` completes 10 battles locally. *Nothing else matters until this works.*
2. **Port the reference example** — get `v0` + masked PPO training end-to-end, ~100k steps,
   confirm it beats `RandomPlayer`. This is the known-good foundation.
3. **Refactor for configurability** — split encoder/reward out of the env, add YAML configs,
   seeding, checkpointing, TensorBoard.
4. **Add algorithms** — tabular Q, DQN (+ masked variant), unmasked PPO.
5. **Build `v1` encoder** — the rich feature set.
6. **Run the matrix** — `run_all.ps1`, 3 seeds per cell.
7. **Analysis** — `evaluate.py` → CSVs, plotting script → `report/figures/`.
8. **Write-up** — README with full reproduction steps.

---

## Known pitfalls (design around these from day one)

- **Windows + `SubprocVecEnv` + asyncio.** poke-env is asyncio-based; `SubprocVecEnv` spawns
  processes. On Windows every entrypoint **must** be guarded by `if __name__ == "__main__":`
  or you get infinite process spawning. Non-negotiable.
- **Showdown must run with `--no-security`**, otherwise rate limiting throttles parallel envs
  into timeouts. Start the server *before* any training run.
- **Unique usernames per env.** poke-env auto-generates them, but colliding accounts across
  parallel envs cause silent hangs. Verify at `num_envs > 1`.
- **Python 3.13 wheels.** poke-env supports 3.10–3.14, but if PyTorch/SB3 wheels are missing
  for 3.13, fall back to a 3.12 venv. Check at step 1, not step 6.
- **`gen9randombattle` variance is high.** Random teams every battle means noisy curves —
  this is *why* 500 eval battles and 3 seeds are required, not optional.
- **Don't chase the GPU.** Set `device="cpu"` for `v0`; only benchmark GPU on `v1`.

---

## Verification

1. **Server** — `scripts/start_showdown.ps1`, then open `http://localhost:8000`.
2. **Smoke test** — `python -m pokerl.evaluate --agent random --opponent random --n 10`
   completes and reports ~50% win rate. Proves the whole loop works.
3. **Training sanity** — masked PPO, `v0`, 100k steps vs `RandomPlayer`; win rate must exceed
   **80%**. If it doesn't, the reward or masking is broken — stop and fix before proceeding.
4. **Reproducibility** — same seed twice → identical learning curve.
5. **Full run** — `scripts/run_all.ps1` produces `results/*.csv` and every figure regenerates
   from committed CSVs alone.
6. **Sanity ceiling** — `SimpleHeuristicsPlayer` should beat `RandomPlayer` ~90%. If our trained
   agent can't approach that, something is wrong regardless of what the curves look like.
