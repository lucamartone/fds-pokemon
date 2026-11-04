# Factorization Machine: 70% research experiment

## Hypothesis

The current three-model ensemble uses extensive tabular summaries, but might
not capture the strength of rare **interactions** between a known P1 lineup and
a partially revealed P2 lineup. A rank-8/rank-16 second-order logistic
Factorization Machine learns low-dimensional pair effects with regularization.
This is an **unverified hypothesis**, not a reported competition improvement.

## Compliance

- Only the provided train JSONL and its first 30 recorded turns supply features.
- P1's six names and P2's actually observed names, HP, and statuses are used.
- Never invent hidden P2 identities, moves, or HP.
- No neural networks, pretrained models, external Pokémon database, extra
  replay data, winner labels from validation, or post-turn-30 data.
- The FM dictionary is fit on the outer-training fold only.
- The baseline uses nested target encoding already implemented in
  `pokemon/te.py`; all predictions are out of fold on identical 5 folds.
- The code makes **no** new Kaggle submission and does not alter the notebook,
  `run.py`, the baseline models, or the three existing CSV files.

## Run

Place the official `train.jsonl`, `test.jsonl`, and
`type_chart_gen2.csv` in a local directory, then:

```bash
python -m pip install -r requirements.txt
python fm_experiment.py --data /path/to/competition_data
```

For a faster first screening without retraining baseline:

```bash
python fm_experiment.py --data /path/to/competition_data \
  --skip-baseline --ranks 8 --maxiter 125
```

Output: `docs/FM_CV_RESULTS.json` with fold accuracies, AUC and log-loss,
optimizer convergence and paired errors; `docs/FM_CV_RESULTS.oof.csv.gz`
with row-aligned probabilities. The full experiment also measures fixed
85/15 and 70/30 probability blends against the original 45/45/10 ensemble.
These blend weights are exploratory and should not be chosen from the public
leaderboard.

Synthetic-data smoke tests:

```bash
python -m unittest discover -s tests -p 'test_fm.py' -v
```

## Evaluation criterion

The 69.72% Kaggle-environment reference is a historical result, not a target
automatically reproduced by a new run. Compare the baseline and FM **within
one identical execution**. To score >70% on 10,000 OOF predictions requires
at least 7,001 correct classifications. Gains of only a few dozen examples
are statistically fragile and vulnerable to iterative model selection.

**Status:** Code implementation and synthetic tests available; real-data
cross-validation not run here because the official JSONL files are absent.
