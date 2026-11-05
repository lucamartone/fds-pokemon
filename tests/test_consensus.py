"""Validate the unanimous consensus decision without test labels."""

import unittest

import numpy as np
import pandas as pd
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from pokemon.consensus import (
    SUBMISSION_FILES, apply_unanimity_gate, assemble_submissions,
    train_selected_submissions, write_submissions,
)


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


class SubmissionGenerationTests(unittest.TestCase):
    """Check that all three artifacts share training without label leakage."""

    def setUp(self):
        self.ids = np.array([101, 102, 103, 104])
        self.baseline = pd.DataFrame({
            "battle_id": self.ids,
            "candidate_7001": [0.8, 0.3, 0.49, 0.55],
            "candidate_7010": [0.7, 0.57, 0.45, 0.9],
        })
        self.roster = np.array([0.9, 0.2, 0.9, 0.1])
        self.hybrid = np.array([0.8, 0.3, 0.8, 0.2])
        self.symmetric = np.array([0.75, 0.4, 0.7, 0.3])

    def build(self):
        return assemble_submissions(
            self.baseline, self.ids, self.roster,
            self.hybrid, self.symmetric,
        )

    def test_three_selected_labels_and_ids(self):
        """Offense, full and consensus predictions must remain distinct."""
        frames = self.build()
        self.assertEqual(tuple(frames), SUBMISSION_FILES)
        self.assertEqual(frames[SUBMISSION_FILES[0]].player_won.tolist(),
                         [1, 0, 0, 1])
        self.assertEqual(frames[SUBMISSION_FILES[1]].player_won.tolist(),
                         [1, 1, 0, 1])
        self.assertEqual(frames[SUBMISSION_FILES[2]].player_won.tolist(),
                         [1, 0, 1, 1])
        for frame in frames.values():
            self.assertEqual(frame.battle_id.tolist(), self.ids.tolist())
            self.assertEqual(list(frame.columns), ["battle_id", "player_won"])

    def test_rejects_misaligned_ids_and_invalid_predictions(self):
        with self.assertRaisesRegex(ValueError, "not aligned"):
            assemble_submissions(
                self.baseline, self.ids[::-1], self.roster,
                self.hybrid, self.symmetric,
            )
        with self.assertRaisesRegex(ValueError, "Invalid probabilities"):
            assemble_submissions(
                self.baseline, self.ids, self.roster,
                self.hybrid, np.array([0.8, np.nan, 0.7, 0.3]),
            )

    def test_writes_three_files_and_preserves_schema(self):
        frames = self.build()
        with TemporaryDirectory() as directory:
            output = write_submissions(frames, Path(directory))
            self.assertEqual(tuple(output), SUBMISSION_FILES)
            for name, path in output.items():
                self.assertTrue(path.is_file())
                pd.testing.assert_frame_equal(pd.read_csv(path), frames[name])

    def test_only_one_fit_of_each_component(self):
        """The shared baseline and experts should not be retrained per CSV."""
        with TemporaryDirectory() as directory:
            base = Path(directory)
            for name in (
                "train.jsonl", "test.jsonl", "type_chart_gen2.csv",
                "sample_submission.csv",
            ):
                (base / name).touch()
            with (patch("pokemon.consensus.train_and_predict") as fit_baseline,
                  patch("pokemon.consensus.fit_independent_experts") as fit_experts,
                  patch("pokemon.consensus.validate_submission") as validate):
                fit_baseline.return_value = (None, self.baseline)
                fit_experts.return_value = (
                    self.ids, self.roster, self.hybrid, self.symmetric
                )
                result = train_selected_submissions(base)
            self.assertEqual(tuple(result), SUBMISSION_FILES)
            fit_baseline.assert_called_once_with(base)
            fit_experts.assert_called_once_with(base)
            self.assertEqual(validate.call_count, 3)


if __name__ == "__main__":
    unittest.main()