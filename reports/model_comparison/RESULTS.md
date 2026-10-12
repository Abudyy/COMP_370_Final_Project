# First model comparison

Dataset: 484 original human annotations.

Development: 387 rows; held-out test: 97 rows.

Candidates are selected using three development folds and macro-F1. The selected model is then evaluated on the held-out test. Near-identical headline groups stay in one partition. TF-IDF is fitted inside each training fold.

| Task | Selected model | Development macro-F1 | Test accuracy | Test macro-F1 | Test majority accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| sentiment | embedding_lr | 0.554 | 48.5% | 0.479 | 39.2% |
| topic | embedding_lr | 0.493 | 48.5% | 0.479 | 19.6% |

## Scope of these results

These results measure held-out articles from the same historical collection, not future-period performance. The test is small, so scores have substantial uncertainty. Near-identical headline matching does not identify all shared events or syndication. There is no temporal holdout because publication dates are absent.

Sentiment means portrayal of Mamdani (confirmed by the user), not general emotional tone. Existing labels are unchanged. Codebook consistency and inter-rater agreement have not been independently established.

Automatic labeling has not been validated by this benchmark. Predictions are marked as requiring review; model scores are uncalibrated. Validate on newly collected articles and inspect precision/recall before setting an acceptance threshold. Model predictions are not new human training labels.

Use review_candidates.csv for development-error inspection. test_predictions.csv is an audit artifact; avoid repeatedly tuning against those test errors. If the test informs future changes, create a fresh test set.

## Files

- summary.json: configuration, metrics, class reports, confusion matrices, and package versions.
- leaderboard.csv: development comparison only.
- splits.csv: reproducible partitions and fold assignments.
- test_predictions.csv: held-out predictions from the selected models.
- review_candidates.csv: up to 40 development mistakes per task, with blank review fields.

## Exploratory check on excluded outlets

| Task | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| sentiment | 47.7% | 0.477 |
| topic | 48.6% | 0.476 |

These five folds exclude entire outlets and overlapping near-identical headlines from training. The selected models are unchanged. This is exploratory historical evidence, not a future-period test.
