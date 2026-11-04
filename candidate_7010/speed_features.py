"""Turn-30 effective speed and observed move-priority features.

Only organizer-provided base speed, the active states, and the observed move
history are used. Missing species metadata retains the frozen default of 75.
"""

import numpy as np
import pandas as pd


COLUMNS = [
    "speed_effective_difference",
    "speed_effective_ratio",
    "speed_p1_first",
    "priority_observed_difference",
    "speed_threat_difference",
]


def _speed_and_priority(timeline, dex, side, state):
    """Estimate speed after boosts/paralysis and highest observed priority."""
    name = state["name"]
    base_speed = dex.get(name, {}).get("spe", 75)
    speed_stage = state["boosts"].get("spe", 0)

    if speed_stage >= 0:
        stage_factor = (2 + speed_stage) / 2
    else:
        stage_factor = 2 / (2 - speed_stage)

    effective_speed = (
        base_speed
        * stage_factor
        * (0.25 if state["status"] == "par" else 1.0)
    )

    priorities = [
        move.get("priority", 0)
        for turn in timeline
        if turn[f"{side}_pokemon_state"]["name"] == name
        for move in [turn[f"{side}_move_details"]]
        if move is not None
    ]

    return effective_speed, max(priorities, default=0)


def row(battle, dex):
    """Create five speed and observed priority features for one battle."""
    timeline = battle["battle_timeline"]
    p1_state = timeline[-1]["p1_pokemon_state"]
    p2_state = timeline[-1]["p2_pokemon_state"]

    p1_speed, p1_priority = _speed_and_priority(
        timeline, dex, "p1", p1_state
    )
    p2_speed, p2_priority = _speed_and_priority(
        timeline, dex, "p2", p2_state
    )

    if p1_priority > p2_priority or (
        p1_priority == p2_priority and p1_speed > p2_speed
    ):
        moves_first = 1.0
    elif p1_priority == p2_priority and p1_speed == p2_speed:
        moves_first = 0.5
    else:
        moves_first = 0.0

    return {
        "speed_effective_difference": p1_speed - p2_speed,
        "speed_effective_ratio": np.log((p1_speed + 1) / (p2_speed + 1)),
        "speed_p1_first": moves_first,
        "priority_observed_difference": p1_priority - p2_priority,
        "speed_threat_difference": float(p1_speed > p2_speed),
    }


def make_frame(battles, dex):
    """Encode a batch of battles in the same fixed column order."""
    return pd.DataFrame(
        [row(battle, dex) for battle in battles],
        columns=COLUMNS,
    )
