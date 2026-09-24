# PokeRL: Project Explanation

---

## 1. What this project is

We built a system that teaches a computer to play competitive Pokémon battles by trial and
error, using **reinforcement learning (RL)**. We did not stop at building one agent and
showing that it works. We set up a **controlled comparison** instead: 33 trained agents,
each differing from a reference setup in exactly one respect, so that we could measure which
choices actually matter. The *learning algorithm* turned out to matter a great deal. Two
things that are usually assumed to be important, how the reward is designed and which
opponent the agent trains against, made no measurable difference to how well the agent
played on average.

---

## 2. Why Pokémon battles are an interesting problem for RL

A Pokémon battle makes a good test problem because it has all the properties that make
decision-making hard:

| Property | What it means here |
|---|---|
| **Partially observable** | You cannot see the opponent's held item, exact stats, or their remaining moves. You have to infer them as the battle unfolds. |
| **Stochastic** | Attacks can miss, critical hits happen randomly, and damage varies within a range. The same action can have different outcomes. |
| **Long horizon** | A battle lasts roughly 30 turns. A mistake on turn 3 may only cost you the game on turn 25. |
| **Large discrete action space** | Each turn you choose one of up to 26 possible actions (4 moves, 5 switches, plus special variants), and most of them are illegal at any given moment. |
| **Strategic depth** | Type matchups, switching, status effects and stat boosts interact in ways that are not obvious. |

Chess and Go are fully observable and deterministic. Pokémon is neither, which puts it closer
to real-world decision problems and makes it much harder for a simple algorithm.

---

## 3. How the system works

We did not write our own Pokémon simulator. We use **Pokémon Showdown**, the open-source
battle simulator that the competitive community plays on, running as a server on our own
machine. Our Python agent connects to it the same way a human player's browser does.

```
   ┌────────────────────┐         battle state          ┌─────────────────────┐
   │  Our RL agent      │  <───────────────────────     │  Pokémon Showdown   │
   │  (Python, PyTorch) │                               │  server (Node.js,   │
   │                    │  ───────────────────────>     │  running locally)   │
   └────────────────────┘         chosen action         └─────────────────────┘
            │                                                     │
            │  learns from                            plays against
            ▼                                                     ▼
     reward signal                                     a scripted opponent bot
```

The bridge between the two is **poke-env**, an open-source Python library that exposes
Showdown battles in the standard format RL libraries expect. Our learning algorithms come
from **Stable-Baselines3**, a widely used, peer-reviewed RL library.

Training and evaluation run entirely on our own machine. Once everything is installed, they
need no internet connection and no outside service.

---

## 4. What the agent sees and does

Every RL problem has three parts: what the agent sees (the state), what it can do (the
action), and how it is scored (the reward). This is how each one works in our project.

### State: what the agent sees

The raw battle is far too complex to feed into a neural network directly, so we convert it
into a list of numbers. We built **two versions**, because how much information to give the
agent was one of the questions we wanted to answer:

- **`v0`, 12 numbers.** The power of each of our 4 moves, how effective each move is against
  the opponent's type, how many Pokémon each side has lost, and the health of both active
  Pokémon. We kept this identical to the encoding used in the published prior work, so our
  results can be compared with it directly.

- **`v1`, 289 numbers.** Everything in `v0`, plus status conditions (poison, burn, sleep),
  stat boosts, both Pokémon's types, the accuracy, category and remaining uses of each move,
  the health and status of all six team members, weather, terrain and entry hazards.

### Action: what the agent does

Each turn the agent picks one number from 0 to 25: use one of 4 moves, switch to one of 5
teammates, or a special variant such as Terastallizing. **On any given turn, most of these
are illegal.** You cannot switch to a fainted Pokémon or use a move that has run out.

This turned out to matter, and it became one of our findings. We give the agent a
**legal-action mask**: before it chooses, every illegal option is removed so it cannot pick
one. Without the mask, the agent is free to pick something impossible and waste the turn.

### Reward: how the agent knows it did well

After each turn the agent gets a score based on how the battle changed. Damage dealt counts
as positive and damage taken as negative, knocking out an opposing Pokémon is a large
positive, and winning the battle is a large positive. We tested three different reward
designs (see section 6).

---

## 5. The algorithms we compared

We implemented five learning methods, covering the main families of RL:

| Algorithm | Family | Idea in one sentence |
|---|---|---|
| **Tabular Q-learning** | Classical | Keep a big lookup table of "how good is action A in situation S", with no neural network at all. |
| **DQN** | Deep, value-based | Use a neural network to estimate how good each action is, and always pick the best. |
| **A2C** | Deep, policy-based | Learn the strategy directly, updating it after short bursts of experience. |
| **PPO** | Deep, policy-based | Like A2C, but with a safeguard that stops the strategy from changing too much in one update. |
| **PPO + masking** | Deep, policy-based | PPO, but illegal actions are removed before choosing. |

We included tabular Q-learning on purpose, as a classical baseline. It shows how much we gain
from using neural networks at all, and it links our work to the older RL literature.

---

## 6. How we designed the experiment

The main idea behind our design is to **change one thing at a time.**

We picked a reference configuration (PPO + masking, `v0` state, shaped reward, trained
against the strongest scripted bot). Every other configuration differs from it in exactly
one respect, so any change in performance can be traced back to that one difference.

We tested four questions:

| Axis | Question | Variants |
|---|---|---|
| **A. Algorithm** | Which learning method works best? | Tabular Q, DQN, A2C, PPO, PPO+masking |
| **B. State** | Does giving the agent more information help? | `v0` (12 numbers) vs `v1` (289 numbers) |
| **C. Reward** | Does a detailed reward beat a simple win/lose reward? | sparse, shaped, aggressive |
| **D. Opponent** | Does training against a stronger bot produce a stronger agent? | random, max-power, heuristic, mixed |

That gives **11 configurations**. We trained each one **three times with different random
starting conditions ("seeds")**, because a single training run can succeed or fail by luck.
That makes **33 training runs**, each 300,000 decisions long, and about 8 hours of training
in total.

---

## 7. How we measured performance

We tested every trained agent the same way, against three scripted opponents of increasing
strength that come built into poke-env:

1. **Random** picks legal actions at random. This is the floor: any competent agent should
   beat it easily.
2. **Max Base Power** always uses its strongest attack. Simple, but not trivial.
3. **Simple Heuristics** is a hand-written strategist. It understands type matchups, switches
   out of bad matchups and sets up hazards. It is a strong opponent.

Each agent played **500 battles against each opponent**, which is 1,500 battles per agent and
about 50,000 battles overall. We also ran a **cross-evaluation**, where the trained agents
play against *each other* as well as against the scripted bots.

We report results as **mean ± standard deviation across the three seeds**.

---

## 8. Results

Win rate against **Simple Heuristics**, the hardest opponent and the one that actually
separates the methods:

| Configuration | Win rate |
|---|---|
| **PPO + masking (`v0`)** | **31.1 ± 2.9** |
| PPO + masking (`v1`, 289 numbers) | 26.5 ± 1.6 |
| PPO without masking | 24.7 ± 1.5 |
| DQN + masking | 11.9 ± 0.5 |
| A2C + masking | 5.7 ± 2.5 |
| Tabular Q-learning | 1.8 ± 0.5 |

Against the two weaker opponents, our best agent wins **96.7%** of battles against Random and
**72.5%** against Max Base Power.

### Finding 1: The algorithm matters a great deal

PPO beats DQN by 19 points, DQN beats A2C by 6, and every deep method is far ahead of
classical tabular Q-learning. This is the clearest result in the study, and the gaps are far
larger than the error bars.

Tabular Q-learning fails because a battle has far too many possible situations to store in a
table. Even after we compressed the state into 6 coarse features, it only ever saw about
6,800 distinct situations and filled in 19% of its table. It barely beats a random player
(54.9%) and is helpless against a real strategy (1.8%).

### Finding 2: Blocking illegal moves helps, but only against strong opponents

| Opponent | With masking | Without | Gain |
|---|---|---|---|
| Random | 96.7 | 96.1 | +0.6 |
| Max Base Power | 72.5 | 70.5 | +2.0 |
| **Simple Heuristics** | **31.1** | **24.7** | **+6.4** |

When the agent picks an illegal move, it wastes the turn. A weak opponent does not punish a
wasted turn, but a strong one does, so masking pays off most against Simple Heuristics.

### Finding 3: More information did not help (at this training budget)

The richer 289-number state scored **lower** than the simple 12-number one (26.5 vs 31.1).

This does not mean the extra information is useless. The larger state needs a much larger
neural network, and that network needs far more practice to train. Both agents were **still
improving when we stopped them**, so the fair conclusion is that `v1` *learns more slowly*,
not that it is worse. With only three runs each, the difference is also not yet
statistically significant, so we treat it as suggestive.

### Finding 4: Reward shaping buys reliability, not skill (our most interesting result)

We expected the detailed reward to beat a bare win/lose reward. On **average** it did not:
all three reward designs reached about 31%.

However, the simple win/lose reward **completely failed in one of its three runs**. That
agent learned nothing across 300,000 decisions and 9,072 battles, and its score never rose
above its starting value. The other two runs were fine. Its average is therefore 22.3 with a
standard deviation of ±18.6, which is a huge spread.

When we looked into why, we found that the failed agent's strategy stayed *random* instead of
sharpening. When the only reward comes at the very end of a battle, the agent has to trace
credit back across about 30 turns. If that never gets started, there is no signal to improve
from and the agent never recovers.

**The detailed reward did not make the agent play better. It made training fail less often.**

This finding also supports our experimental design. If we had run two seeds instead of three
to save time, we would probably have missed this failure and reported the opposite
conclusion.

### Finding 5: The training opponent does not matter

| Trained against | Win rate vs Simple Heuristics |
|---|---|
| Random bot | 28.9 ± 4.7 |
| Max Base Power bot | 32.1 ± 1.9 |
| Simple Heuristics bot | 31.1 ± 2.9 |
| A mix of all three | 30.1 ± 2.7 |

Training against a *purely random* opponent produces an agent as strong as training against
the best scripted one. Our explanation is that the `v0` state contains no information about
the opponent's *behaviour*, only about types, damage and health. So the agent learns type and
damage arithmetic, which is the same no matter who it practises against.

### Finding 6: No learned agent beats the hand-written strategist

In head-to-head cross-evaluation, the scripted Simple Heuristics bot **wins every matchup**
against every one of our trained agents (56 to 79% win rate). At this training budget, a
well-designed hand-written strategy is still stronger than everything we trained. It is a
negative result, and we think it is an important one to report.

---

## 9. Engineering work (and what went wrong)

A large part of our time went into checking that the system produced correct results. We
found and fixed seven real bugs. Several of them would have quietly corrupted the results
without ever crashing:

1. **Duplicate player names.** We seed the random number generator so that experiments are
   repeatable, but the library used that same generator to create the network usernames.
   Every run therefore tried to log in with an *identical* name and collided with the
   previous run's session, and battles never ended. We fixed it by generating names from a
   source that seeding does not affect.

2. **DQN could pick illegal moves while exploring.** During its warm-up phase the library
   samples completely at random, which bypasses our mask. Since only a handful of the 26
   actions are ever legal, this filled its memory almost entirely with impossible moves.

3. **No maskable DQN exists**, so we implemented one, applying the mask both when choosing an
   action and when estimating future value.

4. **The tabular agent had a hidden bias.** When it had no preference between actions, the
   standard "pick the best" function silently picked the *lowest-numbered* action, and
   actions 0 to 5 are switches. So it switched on 73% of turns, took free damage and lost to
   a random bot. Breaking ties randomly instead improved it by 22 points.

5. **Some moves have no priority value.** The forced recharge turn after Hyper Beam is
   represented as a pseudo-move with incomplete data, and it crashed a run 300,000 steps in.

6. **The opponent bot sometimes chooses an unavailable move.** This is an inconsistency in
   the library, and it was ending whole runs.

7. **Unreadable result labels** in the cross-evaluation made the comparison matrix
   meaningless until we fixed them.

We also profiled performance and found the learner using seven CPU cores to do arithmetic on
a very small neural network. Coordinating the threads cost more than the work itself.
Limiting it to one core made training **19% faster**.

---

## 10. Limitations

- **Training was too short.** Every agent was still improving when we stopped at 300,000
  decisions. All our conclusions hold "at this training budget", not in general.
- **Limited statistical power.** With three runs per configuration, the 95% confidence
  interval on any result is roughly ±7 points. Only the large differences (algorithm choice,
  masking) are statistically significant. The `v0` vs `v1` gap is suggestive but not proven.
- **No hyperparameter tuning.** Every configuration used one set of settings. The richer `v1`
  state used a much larger network but the *same* learning rate, so its weaker result may
  come from the settings rather than from the state itself.
- **No memory.** Our agent only sees the current turn. The opponent's set is hidden and
  revealed gradually, so an agent with memory should do better.
- **Random teams.** We used `gen9randombattle`, where teams are randomised every battle. This
  forces general skill rather than a memorised team strategy, but it adds noise.

---

## 11. What we would do next

Our original list, in priority order, and where each item stands now:

1. **Train longer.** *Partly done, for `v1` only (section 12).* 16× the budget (5 million
   steps) moved `v1` from 26.5% (study average) to 27.4% against Simple Heuristics. The
   learning curves told us we had stopped too early, but for `v1` more steps alone bought
   little. `v0` at a longer budget is still untested.
2. **Run 5 to 10 seeds instead of 3**, so we can properly resolve the differences we can
   currently only call suggestive. *Not done.*
3. **Self-play.** *Done, for `v1` (section 12).* It gave our best `v1` agent so far (31.2%)
   and the only one with real switching judgement. It still does not beat the heuristic bot,
   and it got worse at choosing moves.
4. **Add memory** (a recurrent network) to handle the hidden information properly.
   *Not done.*
5. **Tune hyperparameters**, especially for the larger `v1` state. *One variant tried:* a
   higher exploration bonus (28.2%) did no better than plain longer training.

The follow-up added one item to this list: train against a **league** made of self-play
*and* the scripted heuristic bot. Self-play taught switching, and the heuristic bot is what
punishes weak move choices. Mixing the two looks like the most likely way to keep both
skills.

---

## 12. Follow-up: a live coach

After the study we used one of our trained agents as a **teacher**. A panel sits inside the
Showdown game screen while a human plays. On every turn it shows what the agent would do, how
confident it is, and what in the position drives that choice. For example: "switch,
*because* you are at 15% HP; at full health it would attack." The panel only gives advice.
It cannot make a move for you.

Two choices shaped it:

- **It uses the 289-number `v1` state, not the better-scoring `v0`.** `v0` knows nothing
  about your other five Pokémon, so it could never explain *why* switching to one of them is
  a good idea.
- **It ignores forced switches.** When your Pokémon faints, switching is the only option.
  Counting those as "decisions" made the agent look like it switches nearly three times as
  often as it really does (25% instead of 9%).

The key question was whether the agent had actually learned *when to switch out*. The
study's agent had not: how often it switched did not depend on its own health at all. So we
trained three new versions at 5 million steps each, about five hours per run, all on the
same laptop:

| Version | What changed | Win rate vs Simple Heuristics | Switches more when low on HP? |
|---|---|---|---|
| Study agent | Nothing, 300,000 steps | 24.8% | No |
| Longer | 16× more training | 27.4% | No |
| More exploration | 16× more training, higher randomness bonus | 28.2% | No |
| **Self-play** | 16× more training, plays against a copy of itself | **31.2%** | **Yes, 2.4× as often** |

Self-play came out ahead on both counts. Then, while we were playing with the coach, we
found something the win rate had hidden. The coach recommended a move that the opponent
resists over one that was super effective, even though its own panel pointed out the better
matchup. We measured how often this happens. In **about 1 in 4** of its attacking choices,
self-play picks a move where another available move is at least 1.5× stronger on type
matchup alone. For the study agent it was 1 in 8.

The explanation makes sense: a copy of itself does not punish a weak move the way the
scripted strategist does. In practice, the coach can be trusted on *whether to switch*,
which no simple calculator answers well. On *which move to use*, a learner can and should
check the type chart. More detail is in [COACH_FINDINGS.md](COACH_FINDINGS.md) and
[client/README.md](../client/README.md).

---

## 13. Reproducing the results

Everything can be reproduced from the repository:

```powershell
.\scripts\setup_showdown.ps1        # one-time: install the battle simulator
.\scripts\start_showdown.ps1        # start the local server
python -m pokerl.smoke_test          # verify the whole pipeline works
.\scripts\run_all.ps1                # run all 33 experiments + analysis
```

Each experiment is described by a small configuration file, saved next to its own results.
Every figure regenerates from the saved data files without re-running a single battle. The
trained networks themselves are not stored in the repository, so anything that needs to
*play*, such as re-evaluating an agent or running the coach, needs that run to be retrained
first.

The coach follow-up adds `.\scripts\train_coach_arms.ps1`, which trains the three
5-million-step versions together.

**Software used:** Python 3.13, PyTorch, Stable-Baselines3 2.9 (RL algorithms), poke-env 0.15
(Showdown interface), Pokémon Showdown (battle simulator).

**Outputs:**
- `report/RESULTS.md`: full result tables for the study
- `report/COACH_FINDINGS.md`: what the coach got wrong in live play, and how often
- `report/figures/`: win-rate comparison, learning curves, cross-evaluation matrix
- `results/summary.csv`: all numbers in machine-readable form
