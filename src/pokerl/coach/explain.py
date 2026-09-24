"""Turn a battle position into a recommendation and the reasons behind it.

The recommendation is the model's. Everything else in an ``Advice`` exists to make that
recommendation checkable by a human rather than taken on trust:

* the **attack-vs-switch split** answers the question directly, because probability mass over
  actions 0-5 and 6-25 is exactly "how much does it want to switch";
* **counterfactuals** isolate *what* drives the choice, by re-encoding the position with one
  thing changed and reporting only the probes that flip the answer. This is what turns
  "switch" into "switch *because you are at 15%* -- at full HP it would attack";
* **attribution** names which parts of the position the choice is sensitive to;
* **facts** come from poke-env's battle data, never from the network. The model cannot supply
  them: ``v1`` encodes none of the opponent's moves and only ``[hp, fainted, revealed]`` for
  their bench, so "what could they hit my switch-in with" is not in its input at all.

Facts describe; the model decides. They are never presented as a competing recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from poke_env.battle import AbstractBattle, Move, PokemonType, SideCondition
from poke_env.data import GenData
from poke_env.environment import SinglesEnv

from pokerl.coach import features as F
from pokerl.env.encoders import Encoder, get_encoder

SWITCH_ACTIONS = range(0, 6)
IG_STEPS = 32

EFFECTIVENESS = {
    0.0: "no effect",
    0.25: "double resisted",
    0.5: "resisted",
    1.0: "neutral",
    2.0: "super effective",
    4.0: "double super effective",
}


@dataclass
class Option:
    action: int
    kind: str  # "move" | "switch"
    label: str
    probability: float


@dataclass
class Counterfactual:
    probe: str
    original: str
    flipped_to: str
    sentence: str


@dataclass
class Advice:
    battle_tag: str
    turn: int
    recommendation: Option
    confidence: str
    margin: float
    p_attack: float
    p_switch: float
    value: float
    best_move: Option | None
    best_switch: Option | None
    options: list[Option] = field(default_factory=list)
    counterfactuals: list[Counterfactual] = field(default_factory=list)
    attributions: list[tuple[str, float]] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    forced: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "battle_tag": self.battle_tag,
            "turn": self.turn,
            "forced": self.forced,
            "recommendation": vars(self.recommendation),
            "confidence": self.confidence,
            "margin": round(self.margin, 4),
            "p_attack": round(self.p_attack, 4),
            "p_switch": round(self.p_switch, 4),
            "value": round(self.value, 3),
            "best_move": vars(self.best_move) if self.best_move else None,
            "best_switch": vars(self.best_switch) if self.best_switch else None,
            "options": [vars(o) for o in self.options],
            "counterfactuals": [vars(c) for c in self.counterfactuals],
            "attributions": [
                {"bucket": b, "weight": round(w, 4)} for b, w in self.attributions
            ],
            "facts": self.facts,
        }


def _label(action: int, battle: AbstractBattle) -> tuple[str, str]:
    """Human-readable name for an action index, and whether it is a move or a switch."""
    if action in SWITCH_ACTIONS:
        team = list(battle.team.values())
        if action < len(team):
            return "switch", f"switch to {team[action].species.capitalize()}"
        return "switch", f"switch (slot {action})"
    order = SinglesEnv.action_to_order(np.int64(action), battle, strict=False)
    text = str(getattr(order, "message", order)).replace("/choose move ", "")
    if action >= 22:
        return "move", f"{text} (terastallise)"
    return "move", text


def _confidence(margin: float) -> str:
    if margin >= 0.5:
        return "clear"
    if margin >= 0.2:
        return "moderate"
    return "close call"


class Coach:
    """Explains one trained policy's choices."""

    def __init__(self, policy: Any, encoder: Encoder | str = "v1"):
        self.policy = policy
        self.encoder = get_encoder(encoder) if isinstance(encoder, str) else encoder
        if self.encoder.name != "v1":
            # v0 carries no bench features, so a switch recommendation from it cannot be
            # explained and the counterfactual probes would have nothing to perturb.
            raise ValueError(
                f"the coach requires the v1 encoder, got {self.encoder.name!r}: v0 encodes "
                "nothing about the bench, so its switch advice has no grounds to report"
            )

    # -- model queries -------------------------------------------------------------------

    def _tensors(self, obs: np.ndarray, mask: np.ndarray):
        return self.policy.obs_to_tensor(
            {"observation": obs[None, :], "action_mask": mask[None, :]}
        )[0]

    def _probs_and_value(
        self, obs: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, float]:
        tensors = self._tensors(obs, mask)
        with torch.no_grad():
            probs = self.policy.get_distribution(tensors).distribution.probs[0]
            value = float(self.policy.predict_values(tensors)[0].item())
        return probs.cpu().numpy(), value

    def _attributions(
        self, obs: np.ndarray, mask: np.ndarray, action: int
    ) -> list[tuple[str, float]]:
        """Integrated gradients of log p(action) w.r.t. the observation, grouped by segment.

        A plain gradient answers "how would the score move if this feature nudged", which for
        a saturated softmax is often near zero even for the feature that decided the outcome.
        Integrating along the path from an all-zero baseline attributes the whole decision
        instead, so the reported weights mean something when summed.
        """
        mask_t = torch.as_tensor(mask[None, :], dtype=torch.float32)
        action_t = torch.tensor([action])
        baseline = np.zeros_like(obs)
        gradients = np.zeros_like(obs)

        for step in range(1, IG_STEPS + 1):
            point = baseline + (obs - baseline) * (step / IG_STEPS)
            obs_t = torch.as_tensor(point[None, :], dtype=torch.float32)
            obs_t.requires_grad_(True)
            dist = self.policy.get_distribution(
                {"observation": obs_t, "action_mask": mask_t}
            )
            grad = torch.autograd.grad(dist.log_prob(action_t).sum(), obs_t)[0]
            gradients += grad[0].detach().cpu().numpy()

        attribution = (obs - baseline) * gradients / IG_STEPS

        buckets: dict[str, float] = {}
        for index, weight in enumerate(attribution):
            name = F.bucket_of(index)
            buckets[name] = buckets.get(name, 0.0) + float(weight)

        magnitude = sum(abs(v) for v in buckets.values()) or 1.0
        ranked = sorted(buckets.items(), key=lambda kv: -abs(kv[1]))
        return [(name, value / magnitude) for name, value in ranked]

    # -- counterfactuals -----------------------------------------------------------------

    def _counterfactuals(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        battle: AbstractBattle,
        top: int,
        legal: np.ndarray,
    ) -> list[Counterfactual]:
        _, original_label = _label(top, battle)
        own_hp_pct = f"{obs[F.OWN_HP]:.0%}"
        probes: list[tuple[str, np.ndarray, str]] = []

        if obs[F.OWN_HP] < 0.95:
            perturbed = obs.copy()
            perturbed[F.OWN_HP] = 1.0
            probes.append(
                ("your Pokemon at full HP", perturbed, f"your {own_hp_pct} HP, not the matchup")
            )

        if obs[F.OPP_HP] < 0.95:
            perturbed = obs.copy()
            perturbed[F.OPP_HP] = 1.0
            probes.append(("their Pokemon at full HP", perturbed, "how low they are"))

        if obs[F.OWN_HAZARDS.start : F.OWN_HAZARDS.stop].any():
            perturbed = obs.copy()
            perturbed[F.OWN_HAZARDS.start : F.OWN_HAZARDS.stop] = 0.0
            probes.append(
                ("no hazards on your side", perturbed, "the hazards on your side")
            )

        healthy = [slot for slot in range(6) if obs[F.team_slot_hp(slot)] > 0.5]
        if healthy:
            perturbed = obs.copy()
            for slot in healthy:
                perturbed[F.team_slot_hp(slot)] = 0.15
            probes.append(
                (
                    "your bench all at low HP",
                    perturbed,
                    "having a healthy Pokemon to bring in",
                )
            )

        results: list[Counterfactual] = []
        for probe, perturbed, reason in probes:
            probs, _ = self._probs_and_value(perturbed, mask)
            new_top = int(legal[int(np.argmax(probs[legal]))])
            if new_top == top:
                continue
            _, new_label = _label(new_top, battle)
            results.append(
                Counterfactual(
                    probe=probe,
                    original=original_label,
                    flipped_to=new_label,
                    sentence=(
                        f"With {probe} it would play {new_label} instead, so the reason to "
                        f"{original_label} here is {reason}."
                    ),
                )
            )
        return results

    # -- grounded facts ------------------------------------------------------------------

    def _facts(self, battle: AbstractBattle) -> list[str]:
        facts: list[str] = []
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        if me is None or opp is None:
            return facts
        chart = GenData.from_gen(battle.gen).type_chart

        best: tuple[float, Move] | None = None
        for move in battle.available_moves:
            if move.type is None:
                continue
            multiplier = move.type.damage_multiplier(
                opp.type_1, opp.type_2, type_chart=chart
            )
            if best is None or multiplier > best[0]:
                best = (multiplier, move)
        if best is not None:
            multiplier, move = best
            wording = EFFECTIVENESS.get(multiplier, f"{multiplier:g}x")
            facts.append(
                f"Your best type matchup is {move.id} -- {wording} ({multiplier:g}x) "
                f"against {opp.species.capitalize()}."
            )

        # Only base Speed is knowable: a random-battle opponent's EVs, IVs and nature are
        # hidden, so this is stated as a likelihood, not a fact about the actual stat.
        if me.base_stats.get("spe") and opp.base_stats.get("spe"):
            mine = me.base_stats["spe"]
            theirs = opp.base_stats["spe"]
            verdict = "faster than" if mine > theirs else "slower than" if mine < theirs else "tied with"
            facts.append(
                f"Base Speed {mine} vs {theirs}, so you are likely {verdict} "
                f"{opp.species.capitalize()} (their exact spread is hidden)."
            )

        revealed = [m.id for m in opp.moves.values()]
        facts.append(
            f"They have revealed {len(revealed)} move(s): {', '.join(revealed)}."
            if revealed
            else "They have revealed no moves yet, so any switch-in read is a guess."
        )

        rocks = battle.side_conditions.get(SideCondition.STEALTH_ROCK, 0)
        spikes = int(battle.side_conditions.get(SideCondition.SPIKES, 0))
        if rocks or spikes:
            costs = []
            for mon in battle.available_switches:
                chip = 0.0
                if rocks:
                    chip += 0.125 * PokemonType.ROCK.damage_multiplier(
                        mon.type_1, mon.type_2, type_chart=chart
                    )
                if spikes and PokemonType.FLYING not in mon.types:
                    chip += {1: 1 / 8, 2: 1 / 6, 3: 1 / 4}.get(spikes, 0.0)
                if chip:
                    costs.append(f"{mon.species.capitalize()} {chip:.0%}")
            if costs:
                facts.append(
                    "Switching in costs entry-hazard damage: " + ", ".join(costs) + "."
                )
        return facts

    # -- entry point ---------------------------------------------------------------------

    def advise(self, battle: AbstractBattle) -> Advice:
        mask = np.array(SinglesEnv.get_action_mask(battle), dtype=np.int8)
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            raise ValueError(f"no legal action in {battle.battle_tag} turn {battle.turn}")

        obs = self.encoder(battle)
        probs, value = self._probs_and_value(obs, mask)

        options = []
        for action in legal:
            kind, text = _label(int(action), battle)
            options.append(Option(int(action), kind, text, float(probs[action])))
        options.sort(key=lambda o: -o.probability)

        top = options[0]
        margin = top.probability - (options[1].probability if len(options) > 1 else 0.0)
        switch_options = [o for o in options if o.kind == "switch"]
        move_options = [o for o in options if o.kind == "move"]

        # A position with no legal move is a forced switch after a faint, not a decision. The
        # measured switch rate collapses from 25% to 9% once these are excluded, so they must
        # never be reported as the model choosing to switch.
        forced = not move_options

        advice = Advice(
            battle_tag=battle.battle_tag,
            turn=battle.turn,
            recommendation=top,
            confidence=_confidence(margin),
            margin=margin,
            p_attack=float(sum(o.probability for o in move_options)),
            p_switch=float(sum(o.probability for o in switch_options)),
            value=value,
            best_move=move_options[0] if move_options else None,
            best_switch=switch_options[0] if switch_options else None,
            options=options,
            forced=forced,
            facts=self._facts(battle),
        )
        if not forced and switch_options:
            advice.counterfactuals = self._counterfactuals(
                obs, mask, battle, top.action, legal
            )
            advice.attributions = self._attributions(obs, mask, top.action)[:4]
        return advice
