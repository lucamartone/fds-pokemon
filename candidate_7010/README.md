# Candidate 7010 — reproducible Pokémon Gen 2 outcome classifier

This directory contains the **frozen competition candidate** whose exploratory
five-fold out-of-fold (OOF) accuracy was **70.10%** (7,010/10,000). This is
**not** a verified public or private Kaggle score. The last reported public
result for an earlier candidate was 0.69866.

## Competition compliance

- **Classical ML only:** LightGBM, XGBoost, CatBoost and logistic regression;
  no neural networks, deep learning, pretrained models or external APIs.
- **Allowed inputs only:** organizer-supplied `train.jsonl`, `test.jsonl`,
  `type_chart_gen2.csv` and `sample_submission.csv`.
- **Observed state only:** features inspect the provided first 30 active turns,
  disclosed P1 roster, disclosed P2 lead and observed P2 Pokémon. They never
  inspect future turns or final outcomes while encoding an individual test row.
- **No external knowledge:** no Smogon statistics, external Pokédex tables,
  replay collections, manually assembled species databases or outside labels.
- **Label safety:** `pokemon.te.oof_te2` uses inner-fold target statistics to
  encode train rows; test rows are encoded from train labels only.
- **Preserve identifiers:** predictions follow the official sample submission
  schema and `battle_id` order.
- Do **not** use the public leaderboard or test data labels to select features
  or tune parameters.

## Run from a fresh clone

The source relies on the repository's original `pokemon/` modules, which
reproduce the frozen 637-feature baseline.

```bash
python -m pip install -r requirements.txt
python candidate_7010/run.py \
  --data /path/to/official-competition-data \
  --output submission_candidate_7010.csv
```

Optional: save all intermediate candidate test probabilities for reproducibility:

```bash
python candidate_7010/run.py \
  --data /path/to/official-competition-data \
  --output submission_candidate_7010.csv \
  --probabilities candidate_7010_probabilities.csv
```

No cached predictions or trained model files are required. The script trains
the full ensemble, checks input feature dimensions and validates the CSV against
`sample_submission.csv`. It never uploads a submission.

## Frozen model

The architecture is unchanged from the previously verified experiment:

```text
original   = 0.45 * LightGBM + 0.45 * XGBoost + 0.10 * CatBoost
candidate6995 = 0.64 * original + 0.16 * compact logistic
                + 0.20 * active-identity CatBoost
candidate7001 = 0.90 * candidate6995 + 0.10 * observed-offense CatBoost
candidate7005 = 0.85 * candidate7001 + 0.15 * speed-aware CatBoost
candidate7010 = 0.90 * candidate7005 + 0.10 * observed-utility CatBoost
```

All CatBoost variants use 500 iterations, depth 5, learning rate 0.04,
L2 leaf regularization 8, random seed 42 and the original feature order.

The extra features describe the active identities, reciprocal type matchup,
**actually observed** damaging-move pressure, effective speed / priority and
six observed recovery/status/setup flags. They do not infer unobserved moves.

The 70.10% OOF result was selected after multiple same-fold comparisons:
differences of a few correct predictions are exploratory and may not
generalize to the Kaggle leaderboard.

## Tests

```bash
python -m unittest discover -s tests -p 'test_candidate_7010.py' -v
```

For an end-to-end reproducibility check, generate a fresh submission and
compare it to a locally archived reference CSV:

```bash
python candidate_7010/check_submission.py \
  --reference /path/to/archived/submission_candidate_7010.csv \
  --candidate submission_candidate_7010.csv
```

The competition organizers also require a **public Kaggle notebook** that
clones the GitHub repository and generates the final three selected submissions.
That notebook and the **final methodological report are separate deliverables**;
the final report is intentionally not included in this pull request.
