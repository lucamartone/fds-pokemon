"""Validate the unanimous consensus decision without test labels."""

import unittest

import numpy as np

from pokemon.consensus import apply_unanimity_gate


class ConsensusGateTests(unittest.TestCase):
    """Check uncertain, confident, unanimous and split-vote scenarios."""

    def test_confident_baseline_is_unchanged(self):
        """High-confidence decisions must remain unchanged even if opposed."""
        baseline = np.array([0.90, 0.10])
        opposing = np.array([0.20, 0.80])
        labels = apply_unanimity_gate(baseline, opposing, opposing, opposing)
        self.assertEqual(labels.tolist(), [1, 0])

    def test_uncertain_unanimity_can_override(self):
        """Require all three alternate experts to agree on the other class."""
        baseline = np.array([0.54, 0.46, 0.54])
        roster = np.array([0.30, 0.75, 0.20])
        hybrid = np.array([0.40, 0.80, 0.70])
        symmetric = np.array([0.20, 0.90, 0.20])
        labels = apply_unanimity_gate(baseline, roster, hybrid, symmetric)
        self.assertEqual(labels.tolist(), [0, 1, 1])

    def test_decisions_do_not_use_labels(self):
        """The decision function accepts only feature-derived probabilities."""
        baseline = np.array([0.49, 0.51, 0.95])
        expert = np.array([0.8, 0.2, 0.1])
        output = apply_unanimity_gate(baseline, expert, expert, expert)
        self.assertEqual(output.tolist(), [1, 0, 1])


if __name__ == "__main__":
    unittest.main()
