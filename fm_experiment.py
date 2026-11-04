"""Compare a classical Factorization Machine against the existing ensemble.

Examples
--------
python fm_experiment.py --data /path/to/competition/files
python fm_experiment.py --data /path/to/competition/files --skip-baseline

Uses ONLY the competition-provided train data and the 30 recorded turns.
Nested team target encoding is preserved in the independent baseline.
Existing Kaggle submission files and notebook are not modified.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold

from pokemon.fm import LogisticFM, battle_fm_features

SEED = 42
BLEND_WEIGHTS = (0.15, 0.30)


def metrics(y, p):
    return {
        "accuracy": float(accuracy_score(y, p > 0.5)),
        "auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "correct": int(np.sum((p > 0.5) == y)),
    }


def make_folds(y):
    return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
                .split(np.zeros(len(y)), y))


def fm_oof(rows, y, folds, rank, maxiter=125):
    """Never fit a species vocabulary on the outer validation partition."""
    oof = np.full(len(y), np.nan)
    fold_metrics = []
    for k, (train_idx, valid_idx) in enumerate(folds, 1):
        encoder = DictVectorizer(sparse=True)
        X_train = encoder.fit_transform([rows[i] for i in train_idx])
        X_valid = encoder.transform([rows[i] for i in valid_idx])
        model = LogisticFM(rank=rank, maxiter=maxiter, seed=SEED)
        model.fit(X_train, y[train_idx])
        p = model.predict_proba(X_valid)[:, 1]
        oof[valid_idx] = p
        info = {
            "fold": k, "features": int(X_train.shape[1]),
            "accuracy": float(accuracy_score(y[valid_idx], p > 0.5)),
            **model.optimization_,
        }
        fold_metrics.append(info)
        print(f"FM rank={rank} fold {k}/5: {info['accuracy']:.2%}", flush=True)
    if not np.isfinite(oof).all():
        raise RuntimeError("Incomplete FM OOF predictions")
    return oof, fold_metrics


def original_ensemble_oof(data_dir, y, folds):
    """Re-run the existing 45/45/10 reference on the exact same splits."""
    from pokemon.main import build_all, make_model
    from pokemon.te import oof_te2

    data = build_all(data_dir)
    if len(data["y"]) != len(y) or not np.array_equal(data["y"], y):
        raise ValueError("Reference and FM data/target order disagree")
    output = np.full(len(y), np.nan)
    weights = {"lgb": 0.45, "xgb": 0.45, "cat": 0.10}

    for k, (train_idx, valid_idx) in enumerate(folds, 1):
        tr_te, vl_te = oof_te2(
            data["k1"], data["k2"], y, data["keys"], data["C"],
            train_idx, valid_idx)
        X_train = pd.concat([
            data["A"].iloc[train_idx].reset_index(drop=True), tr_te], axis=1)
        X_valid = pd.concat([
            data["A"].iloc[valid_idx].reset_index(drop=True), vl_te], axis=1)
        p = np.zeros(len(valid_idx), dtype=np.float64)
        for name, weight in weights.items():
            model = make_model(name)
            model.fit(X_train, y[train_idx])
            p += weight * model.predict_proba(X_valid)[:, 1]
        output[valid_idx] = p
        print(f"Baseline fold {k}/5: "
              f"{accuracy_score(y[valid_idx], p > 0.5):.2%}", flush=True)
    return output


def run_experiment(directory, ranks=(8, 16), maxiter=125, skip_baseline=False):
    from pokemon.features import read_jsonl

    for filename in ("train.jsonl", "test.jsonl", "type_chart_gen2.csv"):
        if not (directory / filename).is_file():
            raise FileNotFoundError(f"Missing official input: {directory / filename}")
    battles = read_jsonl(directory / "train.jsonl")
    ids = [battle["battle_id"] for battle in battles]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate battle IDs")
    y = np.array([battle["player_won"] for battle in battles], dtype=int)
    if len(set(y)) != 2:
        raise ValueError("Expected both binary classes")

    rows = [battle_fm_features(b) for b in battles]
    folds = make_folds(y)
    oof_data = pd.DataFrame({"battle_id": ids, "player_won": y})
    report = {
        "train_n": len(y), "fold_seed": SEED, "ranks": {},
        "protocol": "5-fold stratified OOF; fold-local sparse vocabulary",
        "caution": "Exploratory validation only; private leaderboard unknown.",
    }
    if not skip_baseline:
        reference = original_ensemble_oof(directory, y, folds)
        oof_data["prob_baseline"] = reference
        report["baseline"] = metrics(y, reference)

    for rank in ranks:
        probabilities, folds_info = fm_oof(rows, y, folds, rank, maxiter)
        oof_data[f"prob_fm_rank{rank}"] = probabilities
        rank_result = {"fm": metrics(y, probabilities), "folds": folds_info}
        report["ranks"][str(rank)] = rank_result
        if not skip_baseline:
            base_correct = (reference > 0.5) == y
            fm_correct = (probabilities > 0.5) == y
            rank_result["paired_vs_baseline"] = {
                "errors_corrected": int(np.sum(~base_correct & fm_correct)),
                "new_errors": int(np.sum(base_correct & ~fm_correct)),
            }
            for weight in BLEND_WEIGHTS:
                combined = (1 - weight) * reference + weight * probabilities
                rank_result[f"blend_{weight:.2f}"] = metrics(y, combined)
                oof_data[f"prob_blend_rank{rank}_{int(weight * 100)}"] = combined
    return report, oof_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--ranks", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--maxiter", type=int, default=125)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("docs/FM_CV_RESULTS.json"))
    args = parser.parse_args()
    if len(set(args.ranks)) != len(args.ranks) or any(r < 1 for r in args.ranks):
        parser.error("Ranks must be distinct positive integers")
    result, oof = run_experiment(args.data, args.ranks, args.maxiter,
                                 args.skip_baseline)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    oof_path = args.out.with_suffix(".oof.csv.gz")
    oof.to_csv(oof_path, index=False)
    result["oof_file"] = oof_path.name
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "baseline": result.get("baseline"),
        "fm_scores": {r: v["fm"] for r, v in result["ranks"].items()},
        "result_file": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
