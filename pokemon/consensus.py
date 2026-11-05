"""Train and reproduce the maintained unanimous-expert ensemble.

All training features derive from the competition's official first 30 turns.
A fixed confidence gate changes the baseline decision only when the roster,
symmetric-resource and hybrid experts all agree on the opposite class.
No cached predictions, external databases or test labels are required.

Usage:
    python run.py --data /path/to/official-data \
        --output submissions/submission.csv
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from pokemon.encoding import oof_te2
from pokemon.features.base import read_jsonl
from pokemon.model import build_all
from pokemon.pipeline import train_and_predict, validate_submission
from pokemon.hybrid import make_hybrid_model
from pokemon.features.symmetric import (
    create_pair_matrix,
    model as symmetric_estimator,
    orientation,
    prepare,
)

MARGIN = 0.10


def roster_features(battles: list[dict]) -> pd.DataFrame:
    """Describe only officially disclosed team members and active statuses."""
    rows = []

    for battle in battles:
        timeline = battle["battle_timeline"]
        player_one = sorted(
            pokemon["name"] for pokemon in battle["p1_team_details"]
        )
        player_two = sorted({
            turn["p2_pokemon_state"]["name"] for turn in timeline
        })

        row = {
            f"p1_member_{index}": player_one[index]
            for index in range(6)
        }
        row.update({
            f"p2_seen_{index}": (
                player_two[index] if index < len(player_two) else "__unseen__"
            )
            for index in range(6)
        })
        row.update({
            "p2_lead": battle["p2_lead_details"]["name"],
            "p1_lead": timeline[0]["p1_pokemon_state"]["name"],
            "p1_active": timeline[-1]["p1_pokemon_state"]["name"],
            "p2_active": timeline[-1]["p2_pokemon_state"]["name"],
            "p1_status": timeline[-1]["p1_pokemon_state"]["status"],
            "p2_status": timeline[-1]["p2_pokemon_state"]["status"],
        })
        rows.append(row)

    return pd.DataFrame(rows)


def build_hybrid_features(
    base: pd.DataFrame,
    team_encodings: pd.DataFrame,
    symmetric: pd.DataFrame,
) -> pd.DataFrame:
    """Combine known numerical, cross-fitted, and symmetric battle features."""
    return pd.concat([
        base.reset_index(drop=True),
        team_encodings.reset_index(drop=True),
        symmetric.reset_index(drop=True),
    ], axis=1)


def fit_independent_experts(data_dir: Path):
    """Fit roster, hybrid and side-symmetric predictors on official labels."""
    # The original feature engine temporarily reads all JSONL data itself.
    # Build its matrices first, before loading battle dictionaries for experts,
    # so peak memory stays usable in the Kaggle notebook environment.
    matrices = build_all(data_dir)
    labels = matrices["y"]
    gc.collect()
    train_battles = read_jsonl(data_dir / "train.jsonl")
    test_battles = read_jsonl(data_dir / "test.jsonl")

    if not np.array_equal(
        labels, [battle["player_won"] for battle in train_battles]
    ):
        raise ValueError("Training rows are not aligned")

    n_train = len(labels)
    n_test = len(test_battles)
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, n_train + n_test)

    # Training-row team statistics are encoded out-of-fold. Test statistics
    # may depend on all official training labels, never on test outcomes.
    enc_train, enc_test = oof_te2(
        matrices["k1"], matrices["k2"], labels,
        matrices["keys"], matrices["C"], train_idx, test_idx,
    )

    roster_train = roster_features(train_battles)
    roster_test = roster_features(test_battles)
    if not roster_train.columns.equals(roster_test.columns):
        raise ValueError("Roster train/test columns differ")

    # Only aggregated roster features are needed after this point.
    del train_battles, test_battles
    gc.collect()

    x_train = build_hybrid_features(
        matrices["A"], enc_train, roster_train
    )
    x_test = build_hybrid_features(
        matrices["B"], enc_test, roster_test
    )
    roster = CatBoostClassifier(
        iterations=500,
        depth=5,
        learning_rate=0.04,
        l2_leaf_reg=10,
        rsm=0.5,
        random_seed=42,
        verbose=False,
        thread_count=4,
        allow_writing_files=False,
        cat_features=list(roster_train.columns),
    )
    print("Expert 1/3: roster model", flush=True)
    roster.fit(x_train, labels)
    p_roster = roster.predict_proba(x_test)[:, 1]
    del roster, x_train, x_test, roster_train, roster_test
    gc.collect()

    # The hybrid model adds symmetric resources to the original 637 features.
    pairs, confirmed_labels, ids = prepare(data_dir)
    if not np.array_equal(labels, confirmed_labels):
        raise ValueError("Symmetric feature labels are misaligned")

    sym_train = pd.DataFrame([
        orientation(a, b) for a, b in pairs[:n_train]
    ]).add_prefix("sym_")
    sym_test = pd.DataFrame([
        orientation(a, b) for a, b in pairs[n_train:]
    ]).add_prefix("sym_")

    hybrid_train = build_hybrid_features(
        matrices["A"], enc_train, sym_train
    )
    hybrid_test = build_hybrid_features(
        matrices["B"], enc_test, sym_test
    )
    if not hybrid_train.columns.equals(hybrid_test.columns):
        raise ValueError("Hybrid train/test columns differ")

    print("Expert 2/3: hybrid resource model", flush=True)
    hybrid = make_hybrid_model()
    hybrid.fit(hybrid_train, labels)
    p_hybrid = hybrid.predict_proba(hybrid_test)[:, 1]
    del hybrid, hybrid_train, hybrid_test, sym_train, sym_test
    gc.collect()

    # Mirrored battles are used only within training; the two directions
    # never cross the boundary of an original-battle validation fold.
    symmetric_train = pd.concat([
        create_pair_matrix(pairs, train_idx, enc_train, swap=False),
        create_pair_matrix(pairs, train_idx, enc_train, swap=True),
    ], ignore_index=True)
    symmetric_labels = np.concatenate([labels, 1 - labels])

    print("Expert 3/3: symmetric mirrored model", flush=True)
    symmetric = symmetric_estimator()
    symmetric.fit(symmetric_train, symmetric_labels)

    sym_a = create_pair_matrix(pairs, test_idx, enc_test, swap=False)
    sym_b = create_pair_matrix(pairs, test_idx, enc_test, swap=True)
    p_symmetric = (
        symmetric.predict_proba(sym_a)[:, 1]
        + 1.0 - symmetric.predict_proba(sym_b)[:, 1]
    ) / 2.0

    del symmetric, symmetric_train, symmetric_labels, sym_a, sym_b, pairs
    gc.collect()
    return ids, p_roster, p_hybrid, p_symmetric


def apply_unanimity_gate(
    baseline: np.ndarray,
    roster: np.ndarray,
    hybrid: np.ndarray,
    symmetric: np.ndarray,
) -> np.ndarray:
    """Change only uncertain decisions contradicted by all three experts."""
    predicted = (baseline > 0.5).astype(int)
    eligible = abs(baseline - 0.5) < MARGIN
    all_positive = (
        (roster > 0.5) & (hybrid > 0.5) & (symmetric > 0.5)
    )
    all_negative = (
        (roster < 0.5) & (hybrid < 0.5) & (symmetric < 0.5)
    )

    predicted[eligible & all_positive] = 1
    predicted[eligible & all_negative] = 0
    return predicted


SUBMISSION_FILES = (
    "submission_1_offense.csv",
    "submission_2_full.csv",
    "submission_3_consensus.csv",
)


def assemble_submissions(
    baseline_probabilities: pd.DataFrame,
    expert_ids: np.ndarray,
    roster: np.ndarray,
    hybrid: np.ndarray,
    symmetric: np.ndarray,
) -> dict[str, pd.DataFrame]:
    """Create all selected submissions from one set of trained predictions.

    The baseline produces both intermediate blends during the same fit.
    The independent experts are shared with the final consensus output.
    No predictions are used as training targets or lookup-table replacements.
    """
    required_columns = {"battle_id", "candidate_7001", "candidate_7010"}
    missing = required_columns.difference(baseline_probabilities.columns)
    if missing:
        raise ValueError(f"Missing baseline probability columns: {sorted(missing)}")

    ids = np.asarray(expert_ids)
    original_ids = baseline_probabilities["battle_id"].to_numpy()
    if ids.ndim != 1 or not np.array_equal(original_ids, ids):
        raise ValueError("Baseline and expert test IDs are not aligned")

    offense = baseline_probabilities["candidate_7001"].to_numpy(dtype=float)
    full = baseline_probabilities["candidate_7010"].to_numpy(dtype=float)
    expert_predictions = (
        np.asarray(roster, dtype=float),
        np.asarray(hybrid, dtype=float),
        np.asarray(symmetric, dtype=float),
    )
    for name, values in zip(
        ("offense", "full", "roster", "hybrid", "symmetric"),
        (offense, full, *expert_predictions),
    ):
        if values.ndim != 1 or len(values) != len(ids):
            raise ValueError(f"Invalid prediction dimensions for {name}")
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"Invalid probabilities for {name}")

    labels = (
        (offense > 0.5).astype(int),
        (full > 0.5).astype(int),
        apply_unanimity_gate(full, *expert_predictions),
    )
    return {
        name: pd.DataFrame({"battle_id": ids, "player_won": target})
        for name, target in zip(SUBMISSION_FILES, labels)
    }


def train_selected_submissions(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Fit the baseline once and the three additional experts once each."""
    required_files = (
        "train.jsonl", "test.jsonl", "type_chart_gen2.csv",
        "sample_submission.csv",
    )
    for name in required_files:
        if not (data_dir / name).is_file():
            raise FileNotFoundError(data_dir / name)

    # A single baseline fit already computes the offense and full blends.
    _, baseline_probabilities = train_and_predict(data_dir)
    ids, roster, hybrid, symmetric = fit_independent_experts(data_dir)
    submissions = assemble_submissions(
        baseline_probabilities, ids, roster, hybrid, symmetric
    )

    sample_path = data_dir / "sample_submission.csv"
    for frame in submissions.values():
        validate_submission(frame, sample_path)
    return submissions


def write_submissions(
    submissions: dict[str, pd.DataFrame], output_dir: Path
) -> dict[str, Path]:
    """Write three independently uploadable Kaggle-ready CSV files."""
    missing = set(SUBMISSION_FILES).difference(submissions)
    if missing:
        raise ValueError(f"Missing selected submissions: {sorted(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    for name in SUBMISSION_FILES:
        path = output_dir / name
        submissions[name].to_csv(path, index=False)
        output_paths[name] = path
    return output_paths


def main() -> None:
    """Rebuild all three final submissions from the organizer's files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("submissions"),
        help="Directory for the three selected Kaggle submissions.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional backwards-compatible extra copy of the consensus CSV.",
    )
    args = parser.parse_args()

    submissions = train_selected_submissions(args.data.resolve())
    output_paths = write_submissions(submissions, args.output_dir)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        submissions[SUBMISSION_FILES[-1]].to_csv(args.output, index=False)

    final = submissions[SUBMISSION_FILES[-1]]["player_won"].to_numpy()
    full = submissions[SUBMISSION_FILES[1]]["player_won"].to_numpy()
    print(json.dumps({
        "submissions": [str(output_paths[name]) for name in SUBMISSION_FILES],
        "rows_per_submission": len(final),
        "baseline_decisions_overridden": int(np.sum(final != full)),
    }, indent=2))


if __name__ == "__main__":
    main()