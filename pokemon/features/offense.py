"""Active-species and observed-move features using official battle data.

The feature builder never guesses hidden move sets. Only damaging moves
actually used by the active species during the observed turns are considered.
"""

import pandas as pd

from pokemon.features.base import type_mult


CATEGORIES = ["active_p1", "active_p2", "active_type_matchup"]


def _observed_offense(battle, dex, chart, side, attacker, defender):
    """Estimate the best observed attack and count distinct damaging moves.

    Damage is a deterministic, heuristic proxy combining base power, accuracy,
    observed types, STAB and capped base-stat ratios. It is not a full
    Pokémon battle simulator.
    """
    options = {}

    if attacker not in dex or defender not in dex:
        return 0.0, 0

    for turn in battle["battle_timeline"]:
        active = turn[f"{side}_pokemon_state"]
        move = turn[f"{side}_move_details"]

        if (
            active["name"] != attacker
            or not move
            or move.get("base_power", 0) <= 0
        ):
            continue

        attack_stat = "atk" if move["category"] == "physical" else "spa"
        defense_stat = "def" if move["category"] == "physical" else "spd"

        power = float(move["base_power"])
        multiplier = type_mult(
            chart, move["type"], dex[defender]["types"]
        )
        stab = 1.5 if move["type"] in dex[attacker]["types"] else 1.0
        ratio = min(
            2.5,
            max(
                0.5,
                float(dex[attacker][attack_stat])
                / max(1, dex[defender][defense_stat]),
            ),
        )

        damage = (
            power
            * float(move.get("accuracy", 1) or 1)
            * multiplier
            * stab
            * ratio
        )
        options[move["name"]] = max(
            damage, options.get(move["name"], 0)
        )

    return float(max(options.values(), default=0)), len(options)


def _type_matchup(dex, chart, attacker, defender):
    """Find the best STAB effectiveness for two known species."""
    return max(
        type_mult(chart, attack_type, dex[defender]["types"])
        for attack_type in dex[attacker]["types"]
        if attack_type != "notype"
    )


def make_frame(battles, dex, chart):
    """Build three categorical and four numeric active-move features."""
    rows = []

    for battle in battles:
        last_turn = battle["battle_timeline"][-1]
        p1 = last_turn["p1_pokemon_state"]["name"]
        p2 = last_turn["p2_pokemon_state"]["name"]

        if p1 in dex and p2 in dex:
            p1_multiplier = _type_matchup(dex, chart, p1, p2)
            p2_multiplier = _type_matchup(dex, chart, p2, p1)
            matchup = f"{p1_multiplier:g}|{p2_multiplier:g}"
        else:
            matchup = "unknown"

        p1_best, p1_moves = _observed_offense(
            battle, dex, chart, "p1", p1, p2
        )
        p2_best, p2_moves = _observed_offense(
            battle, dex, chart, "p2", p2, p1
        )

        rows.append({
            "active_p1": p1,
            "active_p2": p2,
            "active_type_matchup": matchup,
            "seen_active_attack_p1": p1_best,
            "seen_active_attack_p2": p2_best,
            "n_seen_active_attacks_p1": p1_moves,
            "n_seen_active_attacks_p2": p2_moves,
        })

    return pd.DataFrame(rows)
