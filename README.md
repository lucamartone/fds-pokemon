# Pokémon Gen 2 OU — Battle Outcome Prediction

Predict the final `player_won` outcome from official team information and
the first 30 recorded battle turns.

The repository maintains **one active classical-ML solution**: candidate 7010.
It reached **70.10% exploratory five-fold out-of-fold accuracy** on 10,000
labeled battles. This is not a verified Kaggle leaderboard score; the last
user-reported public accuracy was 69.866% for an earlier candidate.

## Run

Install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the organizer's `train.jsonl`, `test.jsonl`,
`type_chart_gen2.csv`, and `sample_submission.csv` into one directory,
then run:

```bash
python run.py --data /path/to/official-data \
  --output submissions/submission_candidate_7010.csv
```

The root `run.py` is the **canonical entry point** and trains candidate
7010 from scratch. It does not rely on cached probabilities or submit to
Kaggle automatically.

Smoke tests:

```bash
python -m unittest discover -s tests -p 'test_candidate_7010.py' -v
```

## Project structure

- `run.py`: maintained command-line entry point.
- `candidate_7010/run.py`: full original + compact + active-identity +
  observed-offense + speed + utility ensemble.
- `candidate_7010/*_features.py`: candidate-specific feature engineering.
- `pokemon/features.py` through `features4.py`: required base features.
- `pokemon/te.py`: cross-fitted target encoding without label leakage.
- `pokemon/main.py`: shared data matrices and base model parameters.
- `tests/test_candidate_7010.py`: feature and target-independence tests.
- `notebooks/kaggle_notebook.ipynb`: Kaggle notebook that clones this
  repository and runs the maintained pipeline.
- `candidate_7010/README.md`: detailed English technical instructions.

## Competition compliance

Use only the organizer's files and their observed states. No neural networks,
pretrained models, external replays, Smogon statistics, external Pokémon
databases, or future turns. The target encoding uses training-fold labels,
never the row's own label or a held-out/test outcome.

Research experiments and outdated committed submission snapshots have been
removed from the working tree. Git history preserves them if needed.
The final report is **not** included; it will be prepared separately.
