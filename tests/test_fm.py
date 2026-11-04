"""Smoke tests for the classical Factorization Machine experiment."""

import unittest

import numpy as np

from pokemon.fm import LogisticFM, battle_fm_features
from fm_experiment import fm_oof, make_folds


class FactorizationMachineTests(unittest.TestCase):
    @staticmethod
    def sample_battle(p2_species=0):
        timeline = []
        for _ in range(30):
            timeline.append({
                "p1_pokemon_state": {
                    "name": "ally0", "hp_pct": 0.8, "status": "nostatus"},
                "p2_pokemon_state": {
                    "name": f"enemy{p2_species}", "hp_pct": 0.3, "status": "par"},
            })
        return {
            "p1_team_details": [{"name": f"ally{i}"} for i in range(6)],
            "battle_timeline": timeline,
        }

    def test_no_hidden_p2_members_invented(self):
        features = battle_fm_features(self.sample_battle())
        self.assertEqual(features["ctx:p2_unknown_count"], 5 / 6)
        self.assertEqual(features["p1_health:ally0"], 0.8)
        self.assertEqual(features["p1_health:ally5"], 1.0)
        self.assertEqual(features["p2_health:enemy0"], 0.3)
        self.assertNotIn("p2_seen:enemy1", features)
        self.assertNotIn("player_won", features)

    def test_sparse_logistic_factorization(self):
        rng = np.random.default_rng(123)
        X = rng.integers(0, 2, size=(80, 6)).astype(float)
        y = ((X[:, 0] * X[:, 1] + X[:, 2]) > 0).astype(int)
        fm = LogisticFM(rank=3, maxiter=80).fit(X, y)
        p = fm.predict_proba(X)
        self.assertEqual(p.shape, (80, 2))
        self.assertTrue(np.isfinite(p).all())
        self.assertGreater(np.mean((p[:, 1] > 0.5) == y), 0.8)

    def test_oof_folds_and_alignment(self):
        y = np.asarray([0, 1] * 20)
        rows = [battle_fm_features(self.sample_battle(i % 3))
                for i in range(40)]
        oof, info = fm_oof(rows, y, make_folds(y), rank=2, maxiter=8)
        self.assertEqual(len(info), 5)
        self.assertEqual(oof.shape, (40,))
        self.assertTrue(np.isfinite(oof).all())
        self.assertTrue(((oof >= 0) & (oof <= 1)).all())

    def test_wrong_shape_rejected(self):
        model = LogisticFM(rank=2, maxiter=2).fit(
            np.ones((8, 4)), np.array([0, 1] * 4))
        with self.assertRaises(ValueError):
            model.predict_proba(np.ones((2, 3)))


if __name__ == "__main__":
    unittest.main()
