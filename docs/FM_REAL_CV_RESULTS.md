# Real-data validation — 4 November 2026

## Inputs and compliance

Executed on the organizer-provided archive with 10,000 labeled `train.jsonl` battles and 5,000 unlabeled `test.jsonl` battles; the supplied `type_chart_gen2.csv` is available. No test labels, prohibited external databases or future turns were used in training or validation. Five stratified outer folds, shuffle=True, random_state=42; full OOF accuracy on 10,000 examples.

Input SHA-256:
- `train.jsonl`: `78f9e91c51b92adebe803bb09221347a55423dce080e433465edf527d2c20d0d`
- `test.jsonl`: `25bdbb2a2f43d6ca480228c142299d325d3ea41eebfab8c4e652447b82914d60`
- `type_chart_gen2.csv`: `66c8e6f8eaa54f07fdd586663324f5e0fbbaaa8e7f0b7e41d83729c28d837e10`

## Factorization Machine, five-fold results

| Model | Accuracy | ROC AUC | Log-loss |
| --- | ---: | ---: | ---: |
| Rank 8, initial regularization (0.001 / 0.0005), 125 iterations | 61.26% | 0.65686 | 1.01071 |
| Rank 16, initial regularization (0.001 / 0.0005), 125 iterations | 60.34% | 0.64336 | 1.37401 |
| Rank 4, L2=0.01/0.01, 110 iterations | 65.45% | 0.72324 | 0.61067 |
| **Rank 8, L2=0.01/0.01, 110 iterations** | **65.96%** | **0.72372** | **0.61015** |
| Rank 4, L2=0.003/0.01, 110 iterations | 65.85% | 0.72595 | 0.60815 |

Strong regularization materially helped; the raw FM alone is still well below the established 69.72% Kaggle-run reference. The optimization of the rank-8/0.01 candidate often stopped at the iteration cap; this is an exploratory configuration, not a claim of optimal convergence.

## Separate exploratory end-state experiment

An independent 713-feature match-state LightGBM model was prototyped using only organizer-supplied features. It is **not** part of the original GitHub submission pipeline, and its new feature representation has not been integrated into the existing source.

| Experiment | 5-fold OOF accuracy |
| --- | ---: |
| State LGBM (250 trees, 8 leaves) | 67.56% |
| State LGBM with cross-fitted team win-rate encodings (350 trees, 8 leaves) | 68.85% |
| 80% team-rating LGBM + 20% regularized FM | 69.04% |
| 70% team-rating LGBM + 30% regularized FM | 69.08% |
| 70% team-rating LGBM + 10% state LGBM + 20% regularized FM | **69.11%** |

For the team-rating model, statistics use only the outer training labels; inner 5-fold cross-fitting prevents the training row's label from entering its own encoding. The opponent candidates are derived from unlabeled team identities in the organizer's train/test inputs. The best blend was selected after checking several weights on the same OOF predictions, so 69.11% is **exploratory**, not a fresh or unbiased test score.

The chosen exploratory 69.11% blend corrected 207 errors made by the standalone team-rating model and introduced 181 new errors, for a net 26 extra correct battles. The exact McNemar/binomial discordance test gives p≈0.204, providing no strong evidence of generalization gain. Its five fold accuracies were 69.80%, 68.90%, 69.25%, 69.35%, and 68.25%.

### Interpretation

No newly tested model exceeded 70%. The historical 69.72% Kaggle OOF baseline is **not** a same-process paired comparator; no claim of measurable improvement over it can be made. Do **not** replace the three existing competition submissions on the basis of these experiments.

### Status

The FM repository defaults were revised to 0.01/0.01 after observing severe overfitting at the initial regularization. Original submission scripts, notebook and CSV snapshots remain unchanged. A re-run of the full comparison against the original pipeline in the same environment would be needed to measure paired error correction and determine whether an FM blend is useful.
