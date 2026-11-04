"""Rebuild the candidate-7010 Kaggle submission from official competition data.

The 70.10% result is an exploratory five-fold out-of-fold accuracy, not a
measured public leaderboard score. No cached predictions are needed.

Usage:
    python candidate_7010/run.py --data /path/to/competition/files
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Keep the underlying BLAS and tree learners within reasonable CPU limits.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pokemon.features.base import load_type_chart, read_jsonl
from pokemon.features.resources import build_dex
from pokemon.model import build_all, make_model
from pokemon.encoding import oof_te2

from pokemon.features.utility import make_frame as utility_frame
from pokemon.features.compact import row as compact_row
from pokemon.features.offense import CATEGORIES, make_frame as observed_frame
from pokemon.features.speed import make_frame as speed_frame


def make_catboost(cat_features: list[str]) -> CatBoostClassifier:
    """Return the exact fixed CatBoost configuration used in validation."""
    return CatBoostClassifier(
        iterations=500,
        depth=5,
        learning_rate=0.04,
        l2_leaf_reg=8,
        rsm=0.3,
        verbose=False,
        random_seed=42,
        thread_count=2,
        allow_writing_files=False,
        cat_features=cat_features,
    )


def active_species_frame(battles: list[dict]) -> pd.DataFrame:
    """Identify only the two active species at the final observed turn."""
    return pd.DataFrame(
        {
            f"active_{side}": [
                battle["battle_timeline"][-1][f"{side}_pokemon_state"]["name"]
                for battle in battles
            ]
            for side in ("p1", "p2")
        }
    )


def append_features(base: pd.DataFrame, extras: pd.DataFrame) -> pd.DataFrame:
    """Concatenate feature frames in the same stable order as training."""
    if len(base) != len(extras):
        raise ValueError("Feature row counts do not match.")
    return pd.concat(
        [base.reset_index(drop=True), extras.reset_index(drop=True)],
        axis=1,
    )


def validate_submission(
    submission: pd.DataFrame, sample_path: Path
) -> None:
    """Check shape, column names and battle IDs against the official sample."""
    sample = pd.read_csv(sample_path)

    if (
        submission.shape != sample.shape
        or not submission.columns.equals(sample.columns)
        or not submission["battle_id"].equals(sample["battle_id"])
    ):
        raise ValueError("Submission differs from the official sample schema.")

    if not submission["player_won"].isin((0, 1)).all():
        raise ValueError("The submission must contain only binary predictions.")


def train_and_predict(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every ensemble component from scratch and predict the official test.

    The source and feature vocabulary come exclusively from organizer-provided
    JSONL and type-chart files. Supervised target encodings for training rows
    are cross-fitted; test rows are encoded using training labels only.
    No event after the 30 observed turns or external usage statistics is read.
    """
    required_files = (
        "train.jsonl",
        "test.jsonl",
        "type_chart_gen2.csv",
        "sample_submission.csv",
    )

    for filename in required_files:
        if not (data_dir / filename).is_file():
            raise FileNotFoundError(data_dir / filename)

    data = build_all(data_dir)
    train = read_jsonl(data_dir / "train.jsonl")
    test = read_jsonl(data_dir / "test.jsonl")

    base_train, base_test = data["A"], data["B"]
    labels = data["y"]

    if base_train.shape != (10000, 637) or base_test.shape != (5000, 637):
        raise ValueError("The original 637-feature matrices have changed.")

    test_ids = np.asarray([battle["battle_id"] for battle in test])
    train_labels = np.asarray([battle["player_won"] for battle in train])

    if not np.array_equal(test_ids, np.asarray(data["ids"])):
        raise ValueError("Test battle identifiers are not aligned.")

    if not np.array_equal(labels, train_labels):
        raise ValueError("Training labels are not aligned.")

    # Cross-fit team statistics for all labeled rows; fit test encodings only
    # from training rows to prevent outcome leakage.
    train_idx = np.arange(len(train))
    test_idx = np.arange(len(train), len(train) + len(test))
    team_train, team_test = oof_te2(
        data["k1"],
        data["k2"],
        labels,
        data["keys"],
        data["C"],
        train_idx,
        test_idx,
    )

    X_train = append_features(base_train, team_train)
    X_test = append_features(base_test, team_test)

    if not X_train.columns.equals(X_test.columns):
        raise ValueError("Baseline train/test columns do not match.")

    # 1. Original 45/45/10 gradient-boosted ensemble.
    print("Stage 1/6: original ensemble", flush=True)
    original_probabilities = np.zeros(len(test), dtype=float)

    for model_name, weight in (
        ("lgb", 0.45),
        ("xgb", 0.45),
        ("cat", 0.10),
    ):
        estimator = make_model(model_name)
        estimator.fit(X_train, labels)
        original_probabilities += (
            weight * estimator.predict_proba(X_test)[:, 1]
        )

    chart = load_type_chart(data_dir / "type_chart_gen2.csv")

    # The compact linear model uses a dictionary inferred from supplied
    # train/test metadata, as in the previously verified submission.
    print("Stage 2/6: compact logistic regression", flush=True)
    all_species = build_dex(train, test)

    compact_train = pd.DataFrame(
        [compact_row(battle, all_species, chart) for battle in train]
    )
    compact_test = pd.DataFrame(
        [compact_row(battle, all_species, chart) for battle in test]
    ).reindex(columns=compact_train.columns)

    if compact_train.shape != (10000, 50) or compact_test.shape != (5000, 50):
        raise ValueError("Unexpected compact feature dimensions.")

    if compact_train.isna().any().any() or compact_test.isna().any().any():
        raise ValueError("Compact resource features contain missing values.")

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=1000, random_state=42),
    )
    logistic.fit(compact_train, labels)
    compact_probabilities = logistic.predict_proba(compact_test)[:, 1]

    # 2. CatBoost with categorical identities of the current active species.
    print("Stage 3/6: active-species CatBoost", flush=True)
    active_train = active_species_frame(train)
    active_test = active_species_frame(test)
    active_train_features = append_features(X_train, active_train)
    active_test_features = append_features(X_test, active_test)

    active_model = make_catboost(list(active_train.columns))
    active_model.fit(active_train_features, labels)
    active_probabilities = active_model.predict_proba(active_test_features)[:, 1]

    candidate_6995 = (
        0.64 * original_probabilities
        + 0.16 * compact_probabilities
        + 0.20 * active_probabilities
    )

    # 3. CatBoost with actually observed active moves and their type matchups.
    print("Stage 4/6: observed-offense CatBoost", flush=True)
    known_species = build_dex(train)
    observed_train = observed_frame(train, known_species, chart)
    observed_test = observed_frame(test, known_species, chart)

    offense_train = append_features(X_train, observed_train)
    offense_test = append_features(X_test, observed_test)
    offense_model = make_catboost(CATEGORIES)
    offense_model.fit(offense_train, labels)
    offense_probabilities = offense_model.predict_proba(offense_test)[:, 1]

    candidate_7001 = 0.90 * candidate_6995 + 0.10 * offense_probabilities

    # 4. CatBoost using observed effective speed and move priority.
    print("Stage 5/6: speed-aware CatBoost", flush=True)
    speed_train = speed_frame(train, known_species)
    speed_test = speed_frame(test, known_species)

    speed_train_features = append_features(offense_train, speed_train)
    speed_test_features = append_features(offense_test, speed_test)
    speed_model = make_catboost(CATEGORIES)
    speed_model.fit(speed_train_features, labels)
    speed_probabilities = speed_model.predict_proba(speed_test_features)[:, 1]

    candidate_7005 = 0.85 * candidate_7001 + 0.15 * speed_probabilities

    # 5. CatBoost using only recovery, status and setup moves actually seen.
    print("Stage 6/6: observed-utility CatBoost", flush=True)
    utility_train = utility_frame(train)
    utility_test = utility_frame(test)

    utility_train_features = append_features(
        speed_train_features, utility_train
    )
    utility_test_features = append_features(
        speed_test_features, utility_test
    )

    if (
        utility_train_features.shape[1] != 667
        or not utility_train_features.columns.equals(
            utility_test_features.columns
        )
    ):
        raise ValueError("Unexpected 667-column candidate feature matrices.")

    utility_model = make_catboost(CATEGORIES)
    utility_model.fit(utility_train_features, labels)
    utility_probabilities = utility_model.predict_proba(
        utility_test_features
    )[:, 1]

    # Final frozen blend from exploratory OOF selection.
    candidate_7010 = 0.90 * candidate_7005 + 0.10 * utility_probabilities

    if (
        not np.isfinite(candidate_7010).all()
        or np.any((candidate_7010 < 0) | (candidate_7010 > 1))
    ):
        raise ValueError("Invalid final predicted probabilities.")

    submission = pd.DataFrame(
        {
            "battle_id": test_ids,
            "player_won": (candidate_7010 > 0.5).astype(int),
        }
    )
    validate_submission(submission, data_dir / "sample_submission.csv")

    probabilities = pd.DataFrame(
        {
            "battle_id": test_ids,
            "candidate_6995": candidate_6995,
            "candidate_7001": candidate_7001,
            "candidate_7005": candidate_7005,
            "utility_cat": utility_probabilities,
            "candidate_7010": candidate_7010,
        }
    )
    return submission, probabilities


def main() -> None:
    """Parse CLI arguments and write the submission and optional probabilities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Directory containing the four official competition files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission_candidate_7010.csv"),
        help="Destination for the Kaggle-ready binary submission.",
    )
    parser.add_argument(
        "--probabilities",
        type=Path,
        help="Optional output CSV for intermediate ensemble probabilities.",
    )
    args = parser.parse_args()

    submission, probabilities = train_and_predict(args.data.resolve())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)

    if args.probabilities is not None:
        args.probabilities.parent.mkdir(parents=True, exist_ok=True)
        probabilities.to_csv(args.probabilities, index=False)

    print(f"Created {args.output} ({len(submission)} rows).", flush=True)


if __name__ == "__main__":
    main()
