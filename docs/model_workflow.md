# Stage 1: supervised article classification

This stage compares models, evaluates the selected candidates, and provides reproducible training and batch prediction commands. It does not yet collect new articles, schedule jobs, or mix machine labels into the original research charts.

The two tasks use the original 484 human annotations: sentiment (Positive / Neutral / Negative portrayal of Mamdani) and the six topic categories. Model input is title + description; outlet identifiers, annotator names, simulated scores, and labels are not input features. Known `No summary available in snippet...` placeholders are removed consistently for training and prediction. The original TF-IDF visualization and source CSV are unchanged.

## Run the comparison

From the repository folder, activate a Python 3.12 virtual environment.

For the fast, fully offline classifiers:

```bash
python -m pip install -r requirements-dev.txt
python -m learning compare --models majority tfidf_word_lr tfidf_char_svm
```

For the full comparison, including pretrained embeddings and SetFit, install CPU PyTorch first to avoid unnecessarily downloading CUDA packages:

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-ml.txt
python -m learning compare
```

The first full run downloads the public `sentence-transformers/all-MiniLM-L6-v2` weights, pinned to revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. No paid API or private account is needed. A network connection and sufficient disk space are required for that first download. If your environment uses a SOCKS proxy, its HTTP client may also need `python -m pip install 'httpx[socks]'`.

| Candidate | What is trained on this dataset |
| --- | --- |
| majority | Most common label in the current training fold |
| tfidf_word_lr | Word/bigram TF-IDF vocabulary and a balanced logistic-regression classifier |
| tfidf_char_svm | Character 3–5-gram TF-IDF vocabulary and a balanced linear SVM |
| embedding_lr | A balanced logistic-regression head on frozen pretrained sentence embeddings |
| setfit | A pretrained sentence encoder adapted to this task, plus its classification head |

SetFit uses a fixed, deliberately bounded CPU budget of 30 embedding training steps per fit. This is a first configuration, not an exhaustive search or a guarantee that this budget is optimal. `--setfit-steps` can change the budget for a future development experiment, but repeated choices based on test results invalidate that test as independent evidence.

Missing optional libraries or inaccessible weights are recorded as unavailable; they are not given fabricated scores. Core-classifier failures stop the comparison. The report records package versions, dataset SHA-256, split seed, and training configuration.

## Evaluation design

1. Conservative near-identical-headline groups are identified without using the labels, using character 3–5-gram cosine similarity ≥ 0.88. This is duplicate protection, not complete event clustering.
2. A seeded group-stratified split reserves roughly 20% of articles as a shared test set. Both tasks use the same development/test partitions.
3. Each candidate is evaluated on three group-stratified folds inside development data. Predictive TF-IDF vocabularies and classifier parameters are fitted only on that fold's training rows. Frozen embeddings are pretrained externally; SetFit reloads a fresh pretrained encoder for every fold.
4. The best non-dummy candidate is selected by pooled development macro-F1. Final test metrics do not participate in selection.
5. The selected candidate is fitted on development data and evaluated on the held-out test once for that comparison. Accuracy, macro-F1, per-class precision/recall/F1, confusion matrices, and individual predictions are saved.

`reports/model_comparison/splits.csv` records the partition and validation-fold assignments. No publication dates are present in the annotation file, so this is not a temporal test. Source styles and related events can still correlate across partitions. Validation on newly collected, human-reviewed articles is needed before interpreting automated trends.

The Model lab tab in Streamlit reads the committed report without downloading models or running training. Its fixed evaluation is independent of the sidebar article filters.

An additional exploratory outlet check uses the models already selected by development scores:

```bash
python -m learning source-check
```

It runs five folds with entire outlets excluded from training and removes near-identical headlines that would cross that boundary. Results go to `source_stress.json` and appear in Model lab. This uses the historical collection as an exploratory generalization check; it does not participate in model selection and does not substitute for a newer-period test.

## Train selected models and label an input CSV

After reading the comparison, fit the selected experimental models on all human-labeled rows:

```bash
python -m learning train
python -m learning predict --input examples/new_articles.csv --output artifacts/example_predictions.csv
```

The three example headlines are fabricated demonstrations of the input format. Predictions on these examples do not measure model performance.

To label a real batch, supply a CSV with `title` and preferably `description`. Other fields, including `published_at`, `url`, and existing human labels, are preserved:

```bash
python -m learning predict --input new_articles.csv --output artifacts/new_article_predictions.csv
```

Training saves local `sentiment.joblib`, `topic.joblib`, and a versioned `manifest.json` in `artifacts/classifiers/`. These reproducible outputs are ignored by git; the large pretrained weights are cached locally. The manifest ties the models to the annotation hash and evaluation report. Load only model artifacts you generated or otherwise trust: joblib uses Python serialization.

Prediction adds `predicted_sentiment`, `predicted_topic`, their model scores and score types, `model_version`, `label_source=model`, and `review_required=True`. It does not overwrite the input file or original annotations. Model scores are uncalibrated and differ in scale across classifier types; do not interpret them as measured chances of correctness.

## Next data collection step

Use [annotation_policy.md](annotation_policy.md) to review development disagreements. Then collect a fresh sample with publication timestamps, review its labels, and assess performance on that newer period. Store corrections separately, preserving original annotations. That newer test is the evidence needed to choose a review/acceptance threshold and connect predictions to automated collection and time-based dashboard views.

Rerunning experiments is allowed, but if you use the published test results to make model changes, treat that partition as development evidence and use a fresh final test for the next version.
