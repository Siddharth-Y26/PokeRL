# Coach findings — observed failures in live play

Things the coach got wrong while a human played against it. Each entry is a concrete position,
because aggregate win rate hides exactly the errors that make a teaching tool untrustworthy.

Model under test unless stated: `results/ppo_v1_selfplay_seed0` (v1 encoder, 5M steps,
31.2% vs `SimpleHeuristicsPlayer`).

---

## 1. Recommends a resisted move over a super-effective one

**2026-09-17, live battle, turn 6.** Yanmega (Bug/Flying, 100% HP) vs Rampardos (Rock, 38% HP).
Human is faster (base Speed 95 vs 58).

The panel recommended **Air Slash at 40%** and ranked **Giga Drain 7th at 5%**.

| move | type | multiplier vs Rock | STAB | effective power | model |
|---|---|---|---|---|---|
| Air Slash | Flying | **0.5x** | yes | 56 | **40%** |
| Giga Drain | Grass | **2x** | no | **150** | 5% |
| Bug Buzz | Bug | 1x | yes | 135 | 9% |
| U-turn | Bug | 1x | yes | 105 | 3% |

Rock resists Flying. The model picked the **weakest** of four moves, at its highest confidence
of the turn, against a target already at 38% that Giga Drain would very likely finish — and
Giga Drain heals on top. Even granting Yanmega's Tinted Lens (which doubles not-very-effective
moves, taking Air Slash to ~112) Giga Drain is still the better play.

**Why this one matters more than a normal misplay.** The panel's own *On the board* section
said "Your best type matchup is gigadrain -- super effective (2x) against Rampardos", directly
above a recommendation contradicting it. The calculator was right and the model was wrong, in
the same panel, at the same moment. Under the current "the model is authoritative" design the
learner is told to follow the weaker move.

**Suspected cause.** `EncoderV1` gives each of the four move slots a damage multiplier
(`_damage_multiplier`, `encoders.py`), so the information is in the input — this is not a blind
spot like the bench is for `v0`. What the encoder does *not* carry is any notion that base power
and multiplier should be combined, and nothing in the reward teaches type reasoning directly;
it is only ever reinforced through eventual HP swings. A 31.2% agent has evidently not
generalised it.

Also worth noting: the model rated `gigadrain terastallize` (14%) nearly 3x higher than plain
`gigadrain` (5%), which is hard to read as anything but noise over the tera action block.

**Status: open, and measured.** See below -- this is not an isolated slip.

---

## 2. Measured: ~1 in 4 attacking recommendations is type-dominated

`scripts/measure_type_choice.py`, 60 battles per model vs `SimpleHeuristicsPlayer`. A choice
counts as *dominated* when another legal move is at least 1.5x stronger on
`base power x type multiplier x STAB`. Status moves score 0 and are excluded: counting them
would brand every Swords Dance an error and inflated the first sample from 18% to 40%.

| | baseline `ppo_masked_v1_seed0` (300k) | `ppo_v1_selfplay_seed0` (5M) |
|---|---|---|
| win rate vs heuristic | 24.8% (cell mean 26.5%) | **31.2%** |
| attacking choices sampled | 950 | 797 |
| **type-dominated** | **12.5%** | **23.7%** |
| better move stronger by | 2.5x avg, 9.6x worst | 2.8x avg, **12.0x worst** |
| >=2x stronger move available | 7.8% | **15.2%** |
| mean confidence when wrong | 63% | 48% |

**Self-play is roughly twice as bad at type matchups as the model it replaced, while winning
more.** Both things are true at once, and win rate hid it completely.

The likely mechanism is visible in the last row. Self-play's policy is flatter -- lower
confidence when wrong, and it picks status moves far more often (30% of choices vs 12%). It
was trained against copies of itself, which do not punish a weak move the way
`SimpleHeuristicsPlayer` does, so there was less pressure to sharpen move selection. What it
gained instead was the switch-vs-stay judgement that the baseline entirely lacks: it switches
voluntarily in 14.1% of decisions below 20% HP against 5.9% at or above 60%. That is a 2.4x
gradient (n = 427 and 2,064). A naive two-proportion test gives z ≈ 6. That overstates it
somewhat, because decisions within one battle are not independent, but the gradient is
still well outside noise. The baseline's rates are 10.6% vs 8.7% (`scripts/diagnose_policy.py`,
~3,500 genuine decisions each, forced switches excluded). The errors are diffuse across many moves rather than concentrated on pivots
like U-turn (8% of them), so this is noise in move choice, not a coherent strategy being
misread by the metric.

### What this means for the product

The "the RL model is authoritative" decision is defensible for **switch-or-stay**, which
self-play demonstrably learned and which no calculator answers well. It is **not** defensible
for **move choice**, where a learner can check the answer in two seconds against a type chart
and will be wrong about one time in four.

Recommended split, and the reason it is also better teaching:

1. **Model leads on switch-or-stay.** Its counterfactuals there are grounded in something real.
2. **Panel flags type-dominated recommendations** instead of hiding the disagreement -- the
   calculator already computes the better move and prints it under *On the board*, directly
   contradicting the advice above it (finding 1). Surfacing "this is type-dominated; gigadrain
   hits 2.7x harder" is honest, and teaches the type chart better than silent agreement would.
3. **Do not simply retrain for longer.** 16x the budget moved win rate +0.9 on the `long` arm
   and did not touch this. A league of mixed opponents -- self-play *plus* the scripted
   heuristic -- is the change that would plausibly keep the switch gains without losing move
   sharpness, since the heuristic is what punishes weak moves.

### Caveats

`base power x type multiplier x STAB` ignores stats, items, abilities (Yanmega's Tinted Lens
doubles not-very-effective moves), and secondary effects such as pivoting, priority and
Knock Off's item removal. Some fraction of the 23.7% is therefore defensible play. The 12x
worst case and the 15.2% with a >=2x alternative are not.

Both models are a single seed, and each rate comes from 60 battles. The 12.5% vs 23.7% gap
is large against that sample (about 800–950 attacking choices each). The exact values are
not precise.

---

## Status

| Recommendation | State |
|---|---|
| Point the coach at self-play for switch-or-stay | Done in the docs and commands. The server's `--run` default is still the 300k baseline |
| Flag type-dominated recommendations in the panel | **Not built.** The calculator's type facts still appear under *On the board*, but the panel does not mark the conflict |
| League training (self-play + `SimpleHeuristicsPlayer`) | **Not run** |

Per-decision logs behind these numbers: `results/<run>/type_choice.csv` (move audit) and
`results/<run>/diagnostic.csv` (switch behaviour).
