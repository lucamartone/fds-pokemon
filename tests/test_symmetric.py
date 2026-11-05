"""Regression tests for the two-sided features and roster identity extractor."""

from __future__ import annotations

import copy
import unittest

import numpy as np
import pandas as pd

from pokemon.consensus import roster_features
from pokemon.features.symmetric import orientation


class SymmetricFeatureTests(unittest.TestCase):
    """Check exchangeability and independence from the battle winner."""

    def test_orientation_swap_reverses_differences(self):
        """Mirroring a battle must reverse advantages and preserve totals."""
        first = {
            "values": {"hp": 4.4, "alive": 5.0},
            "active": "ally",
            "active_state": {"status": "nostatus"},
            "lead": "ally-lead",
        }
        second = {
            "values": {"hp": 3.2, "alive": 4.0},
            "active": "enemy",
            "active_state": {"status": "par"},
            "lead": "enemy-lead",
        }
        forward = orientation(first, second, (0.62, 10), (0.42, 5))
        mirrored = orientation(second, first, (0.42, 5), (0.62, 10))

        for name, value in forward.items():
            if name.startswith("d_"):
                self.assertAlmostEqual(value, -mirrored[name])
            elif name.startswith("s_"):
                self.assertAlmostEqual(value, mirrored[name])

        self.assertEqual(forward["active_a"], mirrored["active_b"])
        self.assertEqual(forward["lead_a"], mirrored["lead_b"])

    def test_roster_features_do_not_read_labels(self):
        """Changing the known training outcome must not affect inference data."""
        battle = {
            "player_won": 0,
            "p1_team_details": [
                {"name": f"ally-{i}"} for i in range(6)
            ],
            "p2_lead_details": {"name": "enemy"},
            "battle_timeline": [
                {
                    "p1_pokemon_state": {
                        "name": "ally-0", "status": "nostatus"
                    },
                    "p2_pokemon_state": {
                        "name": "enemy", "status": "par"
                    },
                }
                for _ in range(30)
            ],
        }
        altered = copy.deepcopy(battle)
        altered["player_won"] = 1

        pd.testing.assert_frame_equal(
            roster_features([battle]), roster_features([altered])
        )
        self.assertEqual(roster_features([battle]).shape, (1, 18))

    def test_sum_difference_columns_are_finite(self):
        """Numerical sides must remain finite after the orientation mapping."""
        first = {
            "values": {"hp": 4.0, "attack": 90.0},
            "active": "a",
            "active_state": {"status": "nostatus"},
            "lead": "a",
        }
        second = {
            "values": {"hp": 1.0, "attack": 50.0},
            "active": "b",
            "active_state": {"status": "brn"},
            "lead": "b",
        }
        frame = orientation(first, second)
        self.assertTrue(np.isfinite([
            value for key, value in frame.items()
            if key.startswith(("d_", "s_"))
        ]).all())


if __name__ == "__main__":
    unittest.main()
