"""Core feature matrices and base estimators for candidate 7010.

Only organizer-provided data and observed first-30-turn states are used.
The canonical submission command is `python run.py`.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from pokemon.features.base import build_frame, load_type_chart, read_jsonl
from pokemon.features.resources import build_dex, build_extras
from pokemon.features.battle import build3
from pokemon.features.matchups import cand_lists, feats4, pool_matrix
from pokemon.encoding import team_keys

SEED = 42
MODEL_NAMES = ("lgb", "xgb", "cat")


def build_all(data_dir):
    """Build numerical training/test matrices from competition-provided inputs.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Directory containing ``train.jsonl``, ``test.jsonl`` and
        ``type_chart_gen2.csv``.

    Returns
    -------
    dict
        ``A`` and ``B`` are matching unlabeled feature matrices; ``y`` contains
        training outcomes, ``k1``/``k2`` are team identifiers, ``keys`` and ``C``
        describe candidate teams, and ``ids`` preserves the original test IDs.

    Notes
    -----
    The vocabulary of move and species features is defined from train only.
    The species base-stat dictionary comes exclusively from supplied records.
    Candidate sets only inspect team identities, never prediction targets.
    """
    data_dir = Path(data_dir)

    # --- Read supplied files without any external Pokémon knowledge base.
    chart = load_type_chart(data_dir / "type_chart_gen2.csv")
    train = read_jsonl(data_dir / "train.jsonl")
    test = read_jsonl(data_dir / "test.jsonl")
    y = np.asarray([battle["player_won"] for battle in train], dtype=int)

    dex = build_dex(train, test)

    # --- Build reusable, entirely label-independent numerical feature groups.
    base_train, moves, species = build_frame(train, chart)
    base_test, _, _ = build_frame(test, chart, moves, species)

    extras_train, slot_columns = build_extras(train, dex)
    extras_test, _ = build_extras(test, dex, slot_columns)

    knockouts_train = build3(train)
    knockouts_test = build3(test)

    # --- Infer candidate P2 lineups from unlabeled observed team identities.
    # The pool matches the original reference pipeline. The experimental
    # two-sided pool was tested separately and did not improve held-out accuracy.
    keys, membership, weights, lookup, species_names = pool_matrix(train + test, dex)
    candidates = cand_lists(train + test, membership, lookup)

    pool_features = pd.DataFrame([
        feats4(battle, candidates[i], membership, weights, species_names, dex, chart)
        for i, battle in enumerate(train + test)
    ])

    n_train = len(train)
    pool_train = pool_features.iloc[:n_train].reset_index(drop=True)
    pool_test = pool_features.iloc[n_train:].reset_index(drop=True)

    # --- Maintain exact train/test column correspondence.
    A = pd.concat([
        base_train.drop(columns="battle_id"), extras_train,
        knockouts_train, pool_train,
    ], axis=1)
    B = pd.concat([
        base_test.drop(columns="battle_id"), extras_test,
        knockouts_test, pool_test,
    ], axis=1).reindex(columns=A.columns)

    k1, k2 = team_keys(train + test)

    return {
        "A": A,
        "B": B,
        "y": y,
        "k1": k1,
        "k2": k2,
        "keys": keys,
        "C": candidates,
        "ids": [battle["battle_id"] for battle in test],
    }


def make_model(name, seed=SEED):
    """Instantiate a seeded classical gradient-boosting classifier.

    Parameters
    ----------
    name : {"lgb", "xgb", "cat"}
        Identifier of the modeling algorithm.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    sklearn-compatible classifier
        A fresh unfitted estimator with ``predict_proba`` support.
    """
    if name == "lgb":
        return lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.03,
            num_leaves=8,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.4,
            min_child_samples=40,
            reg_lambda=10,
            verbose=-1,
            random_state=seed,
            n_jobs=2,
        )

    if name == "xgb":
        return XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.025,
            colsample_bytree=0.75,
            subsample=0.85,
            min_child_weight=12,
            reg_lambda=20,
            tree_method="hist",
            random_state=seed,
            n_jobs=2,
        )

    if name == "cat":
        return CatBoostClassifier(
            iterations=500,
            depth=5,
            learning_rate=0.04,
            l2_leaf_reg=8,
            rsm=0.3,
            verbose=False,
            random_seed=seed,
            thread_count=2,
            allow_writing_files=False,
        )

    raise ValueError(f"Unknown model: {name!r}. Expected one of {MODEL_NAMES}.")
