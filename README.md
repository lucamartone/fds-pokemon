# Pokémon Gen 2 OU — Battle Outcome Prediction

A reproducible, **classical machine learning** pipeline for predicting the
winner of a Generation 2 OU battle from the official organizer data through
the first 30 turns.

## Model

The maintained model is an **unanimous-expert consensus ensemble**.
It combines the original calibrated boosted-tree baseline with three
independently trained CatBoost experts:

- **Roster expert:** disclosed P1 team members, revealed P2 species, active
  identities and observed statuses.
- **Symmetric expert:** each battle is represented in both player orientations;
  the two model probabilities are averaged with reversed labels.
- **Hybrid expert:** original numeric features and folded team statistics are
  combined with a symmetric resource-state representation.

The final prediction follows the baseline except when its probability is
within **0.10** of the decision boundary and all three experts unanimously
predict the opposite class. The gate does not use Kaggle feedback or test
labels. The baseline remains preserved as a component of the implementation.

The exploratory five-fold out-of-fold accuracy was **70.17% (7,017/10,000)**,
compared with **70.10%** for the baseline. The net seven-battle improvement is
not statistically reliable (paired exact *p* ≈ 0.547); the public Kaggle score
of this consensus pipeline has **not yet been verified**. An unrelated
submission scored 70.133% publicly; this repository does not claim to
reproduce that submission.

## Quick start

Install pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

The `--data` directory must contain the organizer's `train.jsonl`, `test.jsonl`,
`type_chart_gen2.csv` and `sample_submission.csv` files.

```bash
python run.py --data /path/to/official-data \
  --output-dir submissions
```

One run writes **three Kaggle-compatible files**, each generated from the same
training session:

```text
submissions/
  submission_1_offense.csv    Observed-offense ensemble (70.01% exploratory CV)
  submission_2_full.csv       Full boosted baseline (70.10% exploratory CV)
  submission_3_consensus.csv  Unanimous-expert ensemble (70.17% exploratory CV)
```

The first two predictions are intermediate blends from the same baseline
training; the three additional experts are trained once and shared with the
consensus output. No archived predictions, external labels or test outcomes
are used. Training is computationally expensive. The historical public Kaggle
score for the offense ensemble was 69.866%, for the full baseline 69.733%,
and the public consensus score has not been verified.

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

To compare two independently generated submissions:

```bash
python scripts/check_submission.py \
  --reference /path/to/reference.csv \
  --candidate submissions/submission_3_consensus.csv
```

## Project layout

```text
pokemon/
  features/
    base.py          Original team, state and move features
    resources.py     HP, remaining resources and status
    battle.py        Switches and knockout estimates
    matchups.py      Team-pool inference and type matchups
    compact.py       Compact resource state for a linear expert
    offense.py       Observed damaging attacks
    speed.py         Observed speed and priority
    utility.py       Observed support moves
    symmetric.py     Exchangeable side features and mirrored-battle model
  encoding.py        Cross-fitted team target encodings
  model.py           Baseline feature matrices and estimators
  pipeline.py        Frozen baseline prediction probabilities
  hybrid.py          Hybrid expert model configuration
  consensus.py       Shared fitting, confidence gate and three submissions
notebooks/kaggle_notebook.ipynb
scripts/check_submission.py
tests/test_features.py
tests/test_consensus.py
tests/test_symmetric.py
docs/PIPELINE.md
run.py
requirements.txt
```

## Competition compliance and limitations

Models use LightGBM, XGBoost, CatBoost and logistic regression, without neural
networks or pretrained models. Features are derived from the organizer's four
files only. Hidden opponent rosters, future observations, external replay
collections, Smogon/Pokédex databases and test labels are never used.
Target statistics on labeled training rows are cross-fitted; the mirrored
orientations of a battle stay together within each validation fold.

[Technical details and evaluation caveats](docs/PIPELINE.md).
The final methodological report is a separate deliverable and is not included.