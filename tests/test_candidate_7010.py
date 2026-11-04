"""Unit tests for official-data-only candidate 7010 feature generation."""

from __future__ import annotations

import copy
import unittest

import numpy as np
import pandas as pd

from pokemon.features.utility import (
    COLUMNS as UTILITY_COLUMNS,
    make_frame as utility_frame,
)
from pokemon.features.compact import row as compact_row
from pokemon.features.offense import (
    CATEGORIES,
    make_frame as observed_frame,
)
from pokemon.features.speed import (
    COLUMNS as SPEED_COLUMNS,
    make_frame as speed_frame,
)


def sample_battle():
    """Construct 30 observed turns without any external Pokemon information."""
    team = [{"name": f"ally{i}"} for i in range(6)]
    state_p1 = {
        "name": "ally0",
        "hp_pct": 0.8,
        "status": "nostatus",
        "boosts": {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
    }
    state_p2 = {
        "name": "enemy0",
        "hp_pct": 0.5,
        "status": "par",
        "boosts": {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
    }
    attack = {
        "name": "tackle",
        "base_power": 40,
        "accuracy": 1,
        "category": "physical",
        "type": "normal",
        "priority": 0,
    }
    setup = {
        "name": "curse",
        "base_power": 0,
        "accuracy": 1,
        "category": "status",
        "type": "ghost",
        "priority": 0,
    }

    return {
        "battle_id": 123,
        "player_won": 1,
        "p1_team_details": team,
        "battle_timeline": [
            {
                "p1_pokemon_state": copy.deepcopy(state_p1),
                "p2_pokemon_state": copy.deepcopy(state_p2),
                "p1_move_details": copy.deepcopy(attack),
                "p2_move_details": copy.deepcopy(setup),
            }
            for _ in range(30)
        ],
    }


def sample_species():
    """Return a minimal base-stat dictionary derived from test fixtures."""
    species = {
        name: {
            "hp": 90,
            "atk": 90,
            "def": 90,
            "spa": 90,
            "spd": 90,
            "spe": 90,
            "types": ["normal", "notype"],
        }
        for name in [*(f"ally{i}" for i in range(6)), "enemy0"]
    }
    return species


class CandidateFeatureTests(unittest.TestCase):
    """Validate dimensions, fixed column order and label independence."""

    def setUp(self):
        self.battle = sample_battle()
        self.species = sample_species()
        self.chart = {("normal", "normal"): 1.0}

    def _features(self, battle):
        """Evaluate each feature family without reading the battle target."""
        return (
            compact_row(battle, self.species, self.chart),
            observed_frame([battle], self.species, self.chart),
            speed_frame([battle], self.species),
            utility_frame([battle]),
        )

    def test_dimensions_and_order(self):
        compact, offense, speed, utility = self._features(self.battle)

        self.assertEqual(len(compact), 50)
        self.assertEqual(offense.shape, (1, 7))
        self.assertEqual(speed.shape, (1, 5))
        self.assertEqual(utility.shape, (1, 6))
        self.assertEqual(list(offense.columns[:3]), CATEGORIES)
        self.assertEqual(list(speed.columns), SPEED_COLUMNS)
        self.assertEqual(list(utility.columns), UTILITY_COLUMNS)
        self.assertEqual(compact["p2_unshown"], 5)

    def test_features_ignore_winner_label(self):
        changed = copy.deepcopy(self.battle)
        changed["player_won"] = 0

        before = self._features(self.battle)
        after = self._features(changed)

        self.assertEqual(before[0], after[0])
        for original_frame, changed_frame in zip(before[1:], after[1:]):
            pd.testing.assert_frame_equal(original_frame, changed_frame)

    def test_observed_attack_and_utility_flags(self):
        _, offense, speed, utility = self._features(self.battle)

        self.assertGreater(offense.loc[0, "seen_active_attack_p1"], 0)
        self.assertEqual(offense.loc[0, "n_seen_active_attacks_p1"], 1)
        self.assertEqual(offense.loc[0, "n_seen_active_attacks_p2"], 0)
        self.assertEqual(utility.loc[0, "p2_active_setup"], 1.0)
        self.assertEqual(utility.loc[0, "p1_active_setup"], 0.0)
        self.assertTrue(np.isfinite(speed.to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
