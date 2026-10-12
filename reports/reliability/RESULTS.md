# Sentiment reliability experiments

The local LLM uses Qwen3 4B Q4_K_M with a pinned revision, explicit target-portrayal rules, and 0, 6 balanced TF-IDF-retrieved human examples. Retrieval is confined to training data.

## SetFit tuning

SetFit settings are selected by pooled macro-F1 across three development folds (387 articles).

| Configuration | Development accuracy | Development macro-F1 |
| --- | ---: | ---: |
| setfit_30 | 54.3% | 0.540 |
| setfit_120 | 56.3% | 0.561 |
| setfit_120_low_lr | 57.1% | 0.569 |

Selected SetFit configuration: `setfit_120_low_lr`.

## Paired development comparison

The following paired comparison uses the same 129 validation articles and training partition for all candidates.

| Candidate | Validation accuracy | Macro-F1 | Accepted coverage at target precision |
| --- | ---: | ---: | ---: |
| setfit_120_low_lr | 60.5% | 0.593 | 15.5% (20 articles) |
| local_llm_0_examples | 55.8% | 0.460 | 19.4% (25 articles) |
| local_llm_6_examples | 53.5% | 0.473 | No qualifying threshold |
| embedding_lr | 60.5% | 0.596 | No qualifying threshold |

Accepting only labels where selected SetFit and the selected LLM agree gives 69.1% observed validation precision at 52.7% coverage (68 articles). This is exploratory; the two models can share errors.

Precision target: 90%; minimum accepted validation articles: 20.
A qualifying threshold is an exploratory validation result, not a claim of guaranteed precision. summary.json includes Wilson intervals and per-label support. The same validation data selected these thresholds, so the intervals do not account for threshold selection.

## Historical diagnostics

The original 97-row test has already been examined. These results are explicitly diagnostic, not fresh independent validation of a new model or an acceptance rule.

| Candidate | Diagnostic accuracy | Diagnostic macro-F1 | Precision at validation threshold | Accepted coverage |
| --- | ---: | ---: | ---: | ---: |
| setfit_120_low_lr | 51.5% | 0.510 | 61.3% | 32.0% (31 articles) |
| local_llm_6_examples | 55.7% | 0.491 | No qualifying validation threshold | — |

## Interpretation

Uncertain predictions are never silently converted to Neutral. Rejected predictions remain stored and must be counted when showing sentiment coverage; selective acceptance can bias trends. Model scores are uncalibrated and differ between SetFit and the restricted-token LLM scorer.

No existing human labels or research charts are replaced. Topic classification is unchanged in this experiment. Ongoing collection and automated publication are not enabled by these historical benchmarks.
