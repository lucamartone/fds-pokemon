"""Verify a rebuilt Kaggle submission against a frozen reference file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def compare_submissions(reference: Path, candidate: Path) -> None:
    """Check binary predictions, schema and battle-ID alignment exactly."""
    expected = pd.read_csv(reference)
    actual = pd.read_csv(candidate)

    required_columns = ["battle_id", "player_won"]

    if list(expected.columns) != required_columns:
        raise ValueError("Reference is not a valid binary submission CSV.")

    if list(actual.columns) != required_columns:
        raise ValueError("Candidate is not a valid binary submission CSV.")

    if not actual["battle_id"].equals(expected["battle_id"]):
        raise ValueError("Battle IDs or row order differ.")

    if not actual["player_won"].isin((0, 1)).all():
        raise ValueError("Candidate labels must be binary.")

    changed = int(
        (actual["player_won"] != expected["player_won"]).sum()
    )

    if changed:
        raise AssertionError(
            f"Regression check failed: {changed} predictions differ."
        )

    print(f"Reproduction check passed: {len(actual)} identical predictions.")


def main() -> None:
    """Read CLI arguments and perform an exact label comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()

    compare_submissions(args.reference, args.candidate)


if __name__ == "__main__":
    main()
