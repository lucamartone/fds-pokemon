"""Estimator for the mixed baseline and symmetric-feature expert."""

from __future__ import annotations

from catboost import CatBoostClassifier

from pokemon.features.symmetric import CATEGORIES


def make_hybrid_model() -> CatBoostClassifier:
    """Return the fixed CatBoost configuration used in OOF evaluation."""
    return CatBoostClassifier(
        iterations=600,
        depth=5,
        learning_rate=0.035,
        l2_leaf_reg=12,
        rsm=0.5,
        random_seed=2026,
        thread_count=4,
        verbose=False,
        allow_writing_files=False,
        cat_features=[f"sym_{name}" for name in CATEGORIES],
    )
