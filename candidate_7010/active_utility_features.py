"""Observed utility moves of the two active Pokemon.

Six binary features indicate whether each current active species was actually
seen using recovery, status-inflicting or setup moves. No hidden move is
inferred. Move-name sets are fixed in source code, not externally sourced data.
"""

import pandas as pd


RECOVERY = {
    "rest", "recover", "softboiled", "milkdrink",
    "moonlight", "morningsun", "synthesis",
}

STATUS = {
    "toxic", "thunderwave", "stunspore", "spore",
    "sleeppowder", "hypnosis", "lovelykiss", "sing",
    "glare", "poisonpowder", "willowisp",
}

SETUP = {
    "curse", "swordsdance", "growth", "agility", "amnesia",
    "bellydrum", "dragondance", "calmmind", "irondefense",
    "nastyplot", "bulkup", "barrier", "acidarmor",
}

COLUMNS = [
    f"{side}_active_{kind}"
    for side in ("p1", "p2")
    for kind in ("recovery", "disrupt", "setup")
]


def row(battle):
    """Build utility flags from recorded moves of the turn-30 active species."""
    timeline = battle["battle_timeline"]
    features = {}

    for side in ("p1", "p2"):
        active_species = timeline[-1][f"{side}_pokemon_state"]["name"]
        observed_moves = {
            move["name"]
            for turn in timeline
            if turn[f"{side}_pokemon_state"]["name"] == active_species
            for move in [turn[f"{side}_move_details"]]
            if move is not None
        }

        features[f"{side}_active_recovery"] = float(
            bool(observed_moves & RECOVERY)
        )
        features[f"{side}_active_disrupt"] = float(
            bool(observed_moves & STATUS)
        )
        features[f"{side}_active_setup"] = float(
            bool(observed_moves & SETUP)
        )

    return features


def make_frame(battles):
    """Encode a batch in a stable, documented six-column order."""
    return pd.DataFrame(
        [row(battle) for battle in battles],
        columns=COLUMNS,
    )
