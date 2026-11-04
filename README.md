# Pokémon Gen 2 OU — Battle Outcome Prediction

Predict the eventual winner of a battle using only the organizer-provided
team data and the first 30 recorded turns.

The maintained model is **candidate 7010**: **70.10% exploratory five-fold
out-of-fold accuracy** (7,010/10,000). This is not a Kaggle leaderboard score;
an earlier candidate scored a user-reported 69.866% publicly.

## Reproduce the submission

```bash
python -m pip install -r requirements.txt
python run.py --data /path/to/official-files \
  --output submissions/submission_candidate_7010.csv
```

The data directory must contain the official `train.jsonl`, `test.jsonl`,
`type_chart_gen2.csv` and `sample_submission.csv`. No external data is used.

Run the feature tests:

```bash
python -m unittest discover -s tests -p 'test_candidate_7010.py' -v
```

Verify a regenerated CSV against an archived copy:

```bash
python scripts/check_submission.py \
  --reference /path/to/archived/submission_candidate_7010.csv \
  --candidate submissions/submission_candidate_7010.csv
```

## Project layout

```text
pokemon/
  features/
    base.py          Official static, state and move features
    resources.py     Remaining resources and status
    battle.py        Switch and knockout approximation
    matchups.py      Team-pool inference and type interactions
    compact.py       Compact logistic-regression resources
    offense.py       Actually observed attacks and active matchup
    speed.py         Effective speed and observed move priority
    utility.py       Observed recovery, status and setup moves
  encoding.py        Nested, leakage-safe team target encoding
  model.py           637 base features and gradient-boosted estimators
  pipeline.py        Full frozen candidate 7010 training and submission
notebooks/kaggle_notebook.ipynb
scripts/check_submission.py
tests/test_candidate_7010.py
docs/PIPELINE.md
run.py
requirements.txt
```

Only classical machine learning is used: LightGBM, XGBoost, CatBoost and
logistic regression. No neural networks, external Pokémon databases,
Smogon usage statistics, pretrained models, extra replays, future turns or
test labels are used. All supervised target encodings are cross-fitted.

Read [technical details](docs/PIPELINE.md) for the fixed blend and validation.
The final report will be prepared separately and is not included here.
