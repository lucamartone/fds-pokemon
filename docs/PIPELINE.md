# Reproducible battle-outcome ensemble — implementation and validation

## Prediction architecture

`python run.py` calls `pokemon.consensus.main`, which trains and predicts
from the four official competition files. The underlying
`pokemon.pipeline.train_and_predict` retains the frozen baseline blend and
its initial feature engineering. That implementation is intentionally not
retrained from an archived test-probability file.

The baseline is based on 637 numeric features augmented by cross-fitted team
win-rate priors. It combines LightGBM, XGBoost, CatBoost, a compact logistic
regression and CatBoost experts over observed active species, damaging moves,
effective speed and observed utility moves. Its historical exploratory OOF
accuracy is 70.10% (7,010/10,000); its user-reported Kaggle public result was
69.733%.

The maintained ensemble then fits three additional classical-ML experts:

1. **Roster expert:** fixed CatBoost (500 iterations, depth 5, learning rate
   0.04, L2 regularization 10, RSM 0.5, seed 42). It receives the original
   numeric features, folded team encodings and 18 disclosed categorical
   roster, active-species and status features.
2. **Hybrid expert:** CatBoost (600 iterations, depth 5, learning rate 0.035,
   L2 regularization 12, RSM 0.5, seed 2026). It receives the original
   numeric features, folded team encodings and the side-symmetric resource
   representation.
3. **Symmetric expert:** CatBoost (550 iterations, depth 6, learning rate
   0.035, L2 regularization 12, RSM 0.8, seed 42). Training includes both
   orientations of every official training battle, with the mirrored target
   inverted. At inference both orientations are predicted and averaged
   with the complemented mirror probability.

The fixed label rule uses a **strict 0.10 margin** around baseline probability
0.5 and overrides a baseline decision only if all three experts agree on the
opposite label. Equalities are not overrides. Its parameters are frozen before
inference and are not derived from any Kaggle-scored test submission.

## Exploratory validation

Five stratified outer folds with seed 42 were used for the archived OOF
comparison. The baseline had 7,010 correct outcomes (70.10%). The consensus
had 7,017 correct (70.17%), correcting 53 baseline errors while introducing
46 new ones. The exact paired sign-test p-value was approximately 0.547,
which does **not** support a robust performance improvement. Many hypotheses
were explored on the same folds; this is a limitation of the estimate.

The public Kaggle accuracy of this consensus model is **unknown**. In
particular, the unrelated `submission_final.csv` with user-reported 70.133%
public accuracy is not used as an input, target, lookup table or calibration
signal. The consensus code does **not** claim to reconstruct that file.

## Run and verify

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_*.py' -v
python run.py --data /path/to/official-data \
  --output submissions/submission.csv
```

The generated CSV is checked against the official sample for column names,
row order and IDs. Every estimator is refit; no model binary, cached predictions
or external dataset is required. The training process is compute intensive.

The notebook `notebooks/kaggle_notebook.ipynb` clones the repository, installs
pinned requirements, locates the organizer's mounted files and invokes the
same root entrypoint. The competition requires three selected final
submissions; selection and reproducibility of those three files is a
**separate outstanding deliverable** and should not be claimed completed by
this single-model notebook.

## Data safety

- Use only the organizer's `train.jsonl`, `test.jsonl`, `type_chart_gen2.csv`
  and `sample_submission.csv`.
- Never inspect turns beyond the supplied 30 observations, hidden opponent
  roster positions or outcome labels of official test battles.
- Team encoding for each training row is learned out-of-fold; test team
  encoding is computed solely from the training outcomes.
- Do not train on a previous Kaggle submission or tune the override threshold
  using the public leaderboard.
- Keep model and Kaggle output versions documented for reproducibility.

The methodological report is intentionally not part of this repository.
