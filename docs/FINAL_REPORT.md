# Predicting Pokémon Gen 2 OU Battle Outcomes from Partial Observations
**Foundations of Data Science — Kaggle challenge, Sapienza University of Rome (2026/27)**

## 1. Research question and task

This project addresses a binary supervised classification problem: predict whether player 1 eventually wins a Pokémon Generation 2 OU battle given an incomplete observation of its first 30 active turns. A match may last 30–80 turns, so a locally advantageous state at turn 30 does not necessarily imply victory.

The organizers provide 10,000 labeled battles (`train.jsonl`), 5,000 unlabeled battles (`test.jsonl`), the official Generation 2 type-effectiveness table (`type_chart_gen2.csv`), and `sample_submission.csv`. Each test prediction is either 0 or 1, paired with its original `battle_id`. The evaluation metric is classification accuracy.

The work is restricted to classical machine learning: neural networks, external replay collections, external statistical databases, and pretrained models are not used. Only information that appears in the competition files is available for prediction.

## 2. Data preparation and representation

The pipeline combines several complementary views of the observed game state:

- **Static team composition:** the six disclosed player-1 Pokémon, their base statistics, typing, items, and player-2 lead information.
- **Observed battle dynamics:** currently active Pokémon, surviving and fainted members, health, statuses, estimated knockout timing, switches, attack observations, and health trajectories across the first 30 turns.
- **Resource and matchup indicators:** health-weighted resources, damage pressure, observed move type effectiveness, and inferred properties of unseen player-2 team members.
- **Team-pool statistics:** candidate opposing teams are inferred from the collection of team identities found in organizer-provided examples. Supervised team win-rate statistics are calculated out of fold on labeled training examples.
- **Active offense, speed and utility:** observed damaging moves, effective speed and priority, and observed recovery, status-inflicting and setup actions.
- **Alternative representations:** roster categoricals and symmetric player-orientation resource features for auxiliary expert models.

The engine's hand-coded grouping of moves and deterministic matchup heuristics represents domain assumptions; it is not an external statistical dataset. Their compliance with the prohibition on manually collected tables should nevertheless be confirmed with the course organizers if questioned.

### Leakage control

The supplied test rows contain no outcome labels. The target-encoding procedure builds training-row features using other training folds and uses all labeled training observations only when encoding test rows. Inference models are fitted from official training labels, not from past Kaggle predictions. The model does not inspect turns beyond the supplied observation window.

The main exploratory comparisons reused training folds across many hypotheses; therefore these out-of-fold scores are not treated as independent, unbiased final assessments.

## 3. Model experimentation

### 3.1 Original boosted-tree baseline

An initial ensemble combines LightGBM, XGBoost and CatBoost using weights 0.45, 0.45 and 0.10. Additional representations include a compact logistic regression trained on resource features and CatBoost experts with active-species categories. These predictions are blended with fixed weights into subsequent candidates.

### 3.2 Observed-offense submission

The first maintained output, `submission_1_offense.csv`, adds a CatBoost model informed by the actual damaging moves observed for the active species, their coverage, and current type matchups. In source this intermediate blend is labeled `candidate_7001`; that name is an experiment identifier and is not a leaderboard accuracy.

### 3.3 Full boosted baseline submission

The second output, `submission_2_full.csv`, adds observed speed, move priority and move-utility indicators. The final full boosted blend is internally labeled `candidate_7010`, combining the previous candidate with a utility CatBoost probability component.

### 3.4 Unanimous-expert consensus submission

The third output, `submission_3_consensus.csv`, retains the full baseline prediction except under a restricted confidence-gating rule. Three independent CatBoost experts use (i) revealed roster and status categories, (ii) a hybrid of baseline and resource-state features, and (iii) a paired, player-orientation-symmetric representation. A baseline label may be reversed only if its probability is strictly within 0.10 of 0.5 and **all three** experts agree on the opposite class.

This conservative mechanism aims to correct low-confidence baseline errors while preserving higher-confidence baseline predictions. The threshold and unanimity policy are fixed in source code; no test labels or previously scored test CSVs are model inputs.

## 4. Validation and results

The original experiments used stratified five-fold out-of-fold validation, with fixed model seeds and cross-fitted target encodings. Model comparisons were exploratory and included several model families, feature additions and blend weights.

| Maintained variant | Historical exploratory out-of-fold accuracy |
|---|---:|
| Observed-offense ensemble | 70.01% |
| Full boosted baseline | 70.10% (7,010 / 10,000) |
| Unanimous-expert consensus | 70.17% (7,017 / 10,000) |

The consensus candidate corrected 53 baseline mistakes and introduced 46 new mistakes, a net improvement of only seven battles. Its historical exact paired-sign-test p-value was approximately 0.547. This difference is **not statistically compelling**, and the 70.17% exploratory score should not be presented as evidence of a reliably superior generalization performance.

### Final Kaggle entries

The team identified the following three final selected entries in Kaggle. These are **competition scores**, not validation scores:

| Final selected entry (as listed) | Private accuracy | Public accuracy |
|---|---:|---:|
| Entry 1 | 0.68228 | 0.69733 |
| Entry 2 | 0.68142 | 0.69866 |
| Entry 3 | 0.67914 | 0.69733 |

All three entries point to the same user-supplied Kaggle notebook version: [notebook-fds-pokemon, Version 1](https://www.kaggle.com/code/lucamartone/notebook-fds-pokemon?scriptVersionId=365928113).

**Important verification limitation:** The score listing by itself does not establish which named output file was used for each row, or whether regenerating the current three source-code variants reproduces each selected submission exactly. This report records the three entries without asserting that an unverified mapping has been confirmed. A bitwise or exact-row comparison against the original three uploaded CSVs is required to close that gap.

The private scores are consistently below the public scores. Possible explanations include ordinary sampling variability or public/private composition differences; these are hypotheses, not demonstrated findings about the hidden evaluation subsets.

## 5. Reproducibility and submitted artifacts

The maintained source repository is **[lucamartone/fds-pokemon](https://github.com/lucamartone/fds-pokemon)**. The public-notebook candidate is `notebooks/kaggle_notebook.ipynb` and explicitly clones that repository, fixing the original modeling code at commit `a10ab4c7913787ee5f931671fa114506e21cd347` so subsequent documentation edits do not silently change the estimator.

To regenerate the three proposed CSV variants outside Kaggle:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_*.py' -v
python run.py --data /path/to/official-data --output-dir submissions
```

The input directory must contain the four official files. The entrypoint trains the baseline components and three independent consensus experts, then writes `submission_1_offense.csv`, `submission_2_full.csv`, and `submission_3_consensus.csv`.

The notebook validates the output row count, column order, original `battle_id` order, ID uniqueness and binary target values, and prints SHA-256 hashes. The repository also provides:

```bash
python scripts/check_submission.py \
    --reference /path/to/original-uploaded.csv \
    --candidate submissions/submission_1_offense.csv
```

The same comparison must be completed for each of the three confirmed original-to-rebuilt file mappings. As of this report, **a complete Kaggle notebook rerun and exact final-upload comparison have not been independently verified**. The notebook should be executed and saved on Kaggle; it must be made public for submission to the organizers, together with this report.

## 6. Limitations and conclusion

Prediction is inherently constrained because the later turns and undisclosed opposing Pokémon are unavailable. Complex team interactions and strategic shifts can reverse the apparent game advantage at turn 30. The current models also use heuristic approximations for damage, observed move semantics and hidden-team estimates.

The consensus model offers a modest exploratory accuracy increase, but the small paired difference and multiple experimental comparisons prevent a strong conclusion. For the course assessment, reproducibility and a transparent record of experimental choices matter as much as an additional fraction of a percentage point.

**Conclusion:** A modular, classical-ML workflow was developed for all three final model variants. The final Kaggle scores are documented above. The outstanding requirement is to confirm exact one-to-one reproduction of the selected CSVs using the source and to execute the notebook successfully in the Kaggle environment. Until then, the workflow is prepared for reproducibility but its exact match to the selected uploads remains unproven.
