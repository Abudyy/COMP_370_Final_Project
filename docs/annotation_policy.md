# Annotation policy for the first supervised models

Sentiment describes **the portrayal of Mamdani**: favorable, neutral, or unfavorable. The user confirmed this interpretation before model development. The training labels themselves have not been changed.

The original CSV uses Positive, Neutral, and Negative. These labels do not describe an outlet's overall political bias, public approval, or the emotional tone of an unrelated event. For example, reporting that Mamdani condemned a violent attack does not automatically imply unfavorable portrayal of Mamdani.

The original topic labels are Election, Policy, Social Response, Opinion, Political Identity, and Political Interaction. The repository still lacks the group's original detailed coding rubric; this file does not invent one or redefine the labels. Before reviewing new articles, document category boundaries and tie-breaking rules with the original annotators. A model's prediction is not evidence that a newly encountered topic fits the old codebook.

## Review process

1. Inspect `reports/model_comparison/review_candidates.csv`, which contains development-fold mistakes and blank review fields.
2. For each row, assess the headline and description using the original coding rules. A disagreement can indicate a model mistake, an ambiguous case, or a human-label inconsistency; do not automatically replace the human label.
3. Preserve the original annotation and add the reviewed label, reviewer, date, and explanation separately.
4. Evaluate on a newly collected, independently reviewed sample that includes different outlets and periods. Include a random audit sample, not just uncertain predictions.
5. Only human-reviewed corrections become additional training labels. Do not train a new version using the model's own unverified predictions as ground truth.

Every prediction from this initial workflow is marked `review_required=True`. Numeric model scores are not calibrated probabilities of correctness; SVM decision margins and logistic-regression scores are different scales. No acceptance threshold or automatic dashboard publication has been enabled.

The [local LLM experiment](reliability_workflow.md) applies this confirmed target-portrayal definition using Negative, Neutral, Positive, and a separate Uncertain output for insufficient or ambiguous context. The expanded prompt rules are a proposed operational interpretation, not a recovered original group codebook. Its measured agreement with the original annotations is reported explicitly. The batch workflow continues to mark predictions for review while thresholds are only historical development experiments.

The rows in `examples/new_articles.csv` are fabricated input-format examples, not collected news or labeled evaluation cases.
