# PokeRL Coach — live teaching mode

A panel that watches your Pokémon Showdown battle and tells you what the trained RL agent
would play, how strongly it prefers that, and what in the position is driving the preference.

The recommendation is **the model's**. The facts listed under *On the board* — type
effectiveness, base Speed, revealed moves, hazard chip on a switch-in — come from battle data,
not from the network, because they are things you can check yourself. They describe the
position; they never argue with the model.

The panel never plays for you. It cannot send a move to Showdown at all.

---

## Running it

The coach loads a trained policy, so the run you point it at must have a `model.zip`. Weights
are not committed; train the run first if you cloned the repo fresh.

```powershell
# 1. Showdown server (leave running)
.\scripts\start_showdown.ps1

# 2. Coach backend (leave running)
.\.venv\Scripts\python.exe -m pokerl.coach.server --run results\ppo_v1_selfplay_seed0

# 3. Once: serve the Showdown client from the local server with the coach built in
.\.venv\Scripts\python.exe scripts\install_coach_client.py
```

Then open `http://localhost:8000`, pick any name, and start a **gen9randombattle**. The panel
appears top-right. It says `connected` once it reaches the backend, then updates on every
turn where Showdown asks you for a decision.

`install_coach_client.py --revert` restores the server's original page. The script downloads
the client markup from `play.pokemonshowdown.com` once, and the page keeps loading the
client's scripts and sprites from there, so this step needs an internet connection. Training
and evaluation do not.

### Why the client is served from localhost

The local server ships **no UI**. `pokemon-showdown/server/static/index.html` is a redirect
that sends `localhost:8000` to the hosted client at `psim.us`. That client cannot reach a
local server in current browsers. It is a public HTTPS site, and its SockJS requests to
loopback are blocked by Private Network Access, so it reports *"Couldn't connect to server!"*.

`install_coach_client.py` replaces the redirect with the real client markup, served from
localhost, and injects three things:

- a server override with `host: 'localhost'`. With that exact host the client skips SockJS
  and opens a direct WebSocket;
- a local stand-in for the login server's `action.php`. The server runs `--no-security`, so
  any name is accepted and registered accounts are not supported;
- the coach panel ([`coach-console.js`](coach-console.js)), loaded from the same origin.

Each injection is commented in the script with the client behaviour it works around.

### Other ways to attach the panel

- **Console:** paste [`coach-console.js`](coach-console.js) into the DevTools console of an
  open Showdown battle page. It hooks `app.receive`, so it works on a page that has already
  connected. Re-pasting is safe.
- **Userscript:** [`pokerl-coach.user.js`](pokerl-coach.user.js) for Tampermonkey or
  Violentmonkey. It wraps `window.WebSocket` at `document-start`. It came before the
  self-hosted page and still works wherever the page can reach its server. Because of the
  block above, the hosted `psim.us` client can no longer reach the local one.

Both are single files with no build step. That is the right trade while the *content* of the
advice is still changing. Forking `smogon/pokemon-showdown-client` is the durable answer once
the panel's content has settled.

### Other pages

[`spectate.html`](spectate.html) is a read-only viewer for the battles running on the local
server, such as training or evaluation. Open it straight from disk. A `file://` page may open
`ws://localhost:8000` directly, which the HTTPS client cannot do. It never sends a move.

To play *against* an agent rather than be coached by one:

```powershell
.\.venv\Scripts\python.exe scripts\play_vs_agent.py --run results\ppo_v1_selfplay_seed0
```

It logs in under a fixed name, which it prints, and waits for you to challenge it to a
gen9randombattle.

## Why it hooks the websocket

The coach needs the private `|request|` payload: your team, your moves, your PP. Showdown only
ever sends that to the player themselves — which is also why a spectator bot could not do this
job, and why the advice has to be generated from your own client's traffic.

The script wraps `window.WebSocket`, watches frames for `gen9randombattle` rooms, and forwards
the protocol lines plus each `|request|` to `ws://localhost:8765`. The backend replays them
into a poke-env `Battle` — the same object the training environment builds, and therefore the
only thing the encoders can read.

The console build hooks `app.receive` instead. That is the client's own handler for every
raw frame, so it can attach after the socket already exists.

Reconstruction is verified against real traffic: `scripts/capture_protocol.py` records a live
battle and replays it, checking active Pokémon, legal moves and legal switches at every
decision point (**90/90 exact** on the last run). `scripts/test_coach_server.py` replays that
capture (`scripts/_capture.json`) through a running coach server exactly as the browser
would. It checks every turn for a usable recommendation and reports latency.

---

## Reading the panel

| Element | Meaning |
|---|---|
| **Move name** | what the model would play, with its confidence and the position eval |
| **attack / switch bar** | probability mass over attacking actions vs switching actions — the direct answer to "should I switch here" |
| **Why** | counterfactuals. Only probes that *change* the answer are shown |
| **On the board** | checkable facts from battle data, not from the model |
| **Weighing most** | which parts of the position the decision is sensitive to |
| **All options** | every legal action with its probability |

A **Forced switch** banner means your Pokémon fainted and switching is the only legal action.
That is not a decision, and the panel says so rather than taking credit for it — counting
those as "chose to switch" is what inflates a measured 9% switch rate to 25%.

### The eval number

It is the critic's value estimate, **not a win probability**. Under the `shaped` reward it is a
discounted mix of HP swing, faints, status and the win bonus, so it is useful for comparing
positions against each other and meaningless as a percentage. Calibrating it into a real win
probability is outstanding work.

---

## Current limitations

- **`gen9randombattle` only.** Both encoders and the trained policy assume it; the userscript
  refuses other formats rather than giving advice from a model that has never seen them.
- **The advice is only as good as the agent, and which agent matters.**

  | `--run` | vs SimpleHeuristics | Switch-or-stay | Move choice |
  |---|---|---|---|
  | `ppo_masked_v1_seed0` (server default, 300k) | 24.8% | flat in HP: no opinion | 12.5% type-dominated |
  | `ppo_v1_selfplay_seed0` (5M, self-play) | **31.2%** | switches 2.4× more below 20% HP than above 60% | **23.7%** type-dominated |

  Use self-play. It is the only agent that has learned the switch judgement this tool most
  wants to teach. Its move picks are wrong about one time in four, though: another legal move
  was ≥1.5× stronger on base power × type × STAB. Check them against the *On the board* type
  facts, which come from the type chart and not the model. See
  [report/COACH_FINDINGS.md](../report/COACH_FINDINGS.md). The panel does not yet flag a
  type-dominated recommendation itself.
- **Every attach method is coupled to the client's internals.** A client update can break the
  socket hooks. `install_coach_client.py` also depends on the client's markup, and it stops
  with an explicit error if the script tags it injects around have moved.
- **The eval number is uncalibrated**; see above.

Measure any candidate model before trusting it:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_policy.py --run results\<run> --battles 200
```
