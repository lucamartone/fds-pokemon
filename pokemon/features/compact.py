"""Compact resource features built only from the first 30 recorded turns.

The strength values are heuristic coefficients, not externally learned Pokemon
metadata. Species base statistics and types come from competition data.
"""

import numpy as np

from pokemon.features.base import type_mult


STATUS_COST = {
    "nostatus": 1.0,
    "par": 0.78,
    "brn": 0.8,
    "psn": 0.78,
    "tox": 0.72,
    "slp": 0.65,
    "frz": 0.5,
    "fnt": 0.0,
}

HEAL = {
    "rest", "recover", "softboiled", "milkdrink",
    "moonlight", "morningsun", "synthesis",
}


def side(battle, side_name, dex):
    """Summarize the resources and observed statuses of one player's team.

    Unobserved members of P1's disclosed roster and unknown P2 team slots are
    assigned full HP as in the frozen candidate. No hidden species is inferred.
    """
    timeline = battle["battle_timeline"]
    state_key = f"{side_name}_pokemon_state"
    move_key = f"{side_name}_move_details"

    last_state = {}
    observed = []

    for turn in timeline:
        state = turn[state_key]
        species = state["name"]
        last_state[species] = state
        if species not in observed:
            observed.append(species)

    if side_name == "p1":
        all_names = [member["name"] for member in battle["p1_team_details"]]
    else:
        all_names = observed

    unshown = 0 if side_name == "p1" else max(0, 6 - len(all_names))

    hps = np.array(
        [
            float(last_state[name]["hp_pct"]) if name in last_state else 1.0
            for name in all_names
        ] + [1.0] * unshown
    )
    factors = np.array(
        [
            STATUS_COST.get(last_state[name]["status"], 1.0)
            if name in last_state else 1.0
            for name in all_names
        ] + [1.0] * unshown
    )

    effective_hp = hps * factors
    alive_hp = np.array([hp for hp in hps if hp > 0])
    known_names = [name for name in all_names if name in dex]
    alive_known = [
        name for name in known_names
        if name not in last_state or last_state[name]["hp_pct"] > 0
    ]

    observed_moves = [
        turn[move_key]["name"] for turn in timeline if turn[move_key]
    ]
    active_state = timeline[-1][state_key]
    last_ten_active_hp = [
        float(turn[state_key]["hp_pct"]) for turn in timeline[-10:]
    ]
    fainted = sum(
        last_state[name]["status"] == "fnt"
        or last_state[name]["hp_pct"] == 0
        for name in observed
    )

    features = {
        "hp": float(np.sum(hps)),
        "effective_hp": float(np.sum(effective_hp)),
        "alive": int(np.sum(hps > 0)),
        "fainted": fainted,
        "critical": int(np.sum((hps > 0) & (hps < 0.3))),
        "below_half": int(np.sum((hps > 0) & (hps < 0.5))),
        "healthy": int(np.sum(hps >= 0.8)),
        "hp_low3": float(np.sort(hps)[:3].sum()),
        "hp_high3": float(np.sort(hps)[-3:].sum()),
        "hp_std": float(np.std(hps)),
        "hp_min_alive": float(np.min(alive_hp)) if len(alive_hp) else 0.0,
        "status_weight_loss": float(np.sum(hps - effective_hp)),
        "statused": int(np.sum((hps > 0) & (factors < 0.95))),
        "active_hp": float(active_state["hp_pct"]),
        "active_effective_hp": float(active_state["hp_pct"])
        * STATUS_COST.get(active_state["status"], 1.0),
        "active_boost": float(sum(active_state["boosts"].values())),
        "active_speed_boost": float(active_state["boosts"]["spe"]),
        "last10_active_hp_mean": float(np.mean(last_ten_active_hp)),
        "heal_used": sum(move in HEAL for move in observed_moves),
        "heal_seen": int(any(move in HEAL for move in observed_moves)),
        "last5_heal_used": sum(
            (turn[move_key] or {}).get("name") in HEAL
            for turn in timeline[-5:]
        ),
        "unshown": unshown,
        "shown": len(observed),
        "last_switch_age": 29 - next(
            (
                turn_idx
                for turn_idx in range(29, 0, -1)
                if timeline[turn_idx][state_key]["name"]
                != timeline[turn_idx - 1][state_key]["name"]
            ),
            0,
        ),
    }

    for stat in ("atk", "spa", "def", "spd", "spe"):
        features["surviving_" + stat] = float(
            sum(
                dex[name][stat]
                * (last_state[name]["hp_pct"] if name in last_state else 1.0)
                * STATUS_COST.get(last_state[name]["status"], 1.0)
                if name in last_state else dex[name][stat]
                for name in alive_known
            )
        )

    return features, alive_known, last_state


def row(battle, dex, chart):
    """Generate the original 50 compact features in stable column order."""
    p1, alive_p1, last_p1 = side(battle, "p1", dex)
    p2, alive_p2, last_p2 = side(battle, "p2", dex)

    features = {"p2_unshown": p2["unshown"], "p2_shown": p2["shown"]}

    for key in p1:
        if key not in ("unshown", "shown"):
            features["diff_" + key] = p1[key] - p2[key]

    for key in (
        "hp", "effective_hp", "status_weight_loss", "alive",
        "fainted", "heal_seen", "active_hp", "active_boost",
    ):
        features["p1_" + key] = p1[key]
        features["p2_" + key] = p2[key]

    if alive_p1 and alive_p2:

        def best_stab(attacker, defender):
            """Return the best STAB type multiplier from the supplied chart."""
            return max(
                type_mult(chart, attack_type, dex[defender]["types"])
                for attack_type in dex[attacker]["types"]
                if attack_type != "notype"
            )

        offense = np.array([
            [best_stab(a, b) for b in alive_p2] for a in alive_p1
        ])
        defense = np.array([
            [best_stab(b, a) for b in alive_p2] for a in alive_p1
        ])

        p1_hp = np.array([
            (
                last_p1[name]["hp_pct"]
                * STATUS_COST.get(last_p1[name]["status"], 1.0)
            ) if name in last_p1 else 1.0
            for name in alive_p1
        ])
        p2_hp = np.array([
            last_p2[name]["hp_pct"]
            * STATUS_COST.get(last_p2[name]["status"], 1.0)
            for name in alive_p2
        ])

        p1_offense = (offense.max(axis=0) * p2_hp).sum()
        p2_offense = (defense.max(axis=1) * p1_hp).sum()

        features["matchup_offense_minus_defense"] = float(
            p1_offense - p2_offense
        )
        features["matchup_offense"] = float(p1_offense)
        features["matchup_defense"] = float(p2_offense)
        features["coverage_p1"] = float(np.mean(offense.max(axis=0) >= 2))
        features["coverage_p2"] = float(np.mean(defense.max(axis=1) >= 2))
    else:
        features.update({
            key: 0.0
            for key in (
                "matchup_offense_minus_defense",
                "matchup_offense",
                "matchup_defense",
                "coverage_p1",
                "coverage_p2",
            )
        })

    return features
