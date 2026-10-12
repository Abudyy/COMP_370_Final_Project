# Stage 2: sentiment reliability

This experiment tests two possible improvements: a larger SetFit training budget, and a local language model that applies target-specific coding rules with retrieved human examples. It also measures precision versus coverage rather than assuming every model prediction is reliable.

Sentiment means **portrayal of Mamdani**, not general emotional tone. Only this task changes in stage 2. The original annotations, research charts, and topic-classification workflow are preserved.

## Local setup

Use Python 3.12 in a virtual environment. The dashboard still needs only `requirements.txt`: it displays committed reports without loading models. Run experiments separately on a machine with enough memory for the quantized language model and its context. Run SetFit and the LLM sequentially on a memory-constrained machine.

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-ml.txt
python -m pip install -r requirements-local-llm.txt
```

`llama-cpp-python` may compile locally, requiring CMake and a C/C++ compiler. On a Linux machine with GCC, a bounded source build can use:

```bash
CC=gcc CXX=g++ CMAKE_BUILD_PARALLEL_LEVEL=2 python -m pip install llama-cpp-python==0.3.36 --no-binary=llama-cpp-python
```

No paid API or API key is used. The first run downloads [Qwen3 4B GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF), revision `bc640142c66e1fdd12af0bd68f40445458f3869b`, file `Qwen3-4B-Q4_K_M.gguf`. Model weights stay in the local Hugging Face cache. The language model is pretrained externally; the 484 annotations provide examples in its prompt, rather than fine-tuning its weights.

## Reproduce the experiment

The original model comparison must exist and match the annotation file's SHA-256.

```bash
MEDIA_LENS_CPU_THREADS=4 python -m learning reliability-setfit
python -m learning reliability-llm --threads 6
python -m learning reliability-summary --target-precision 0.90 --minimum-accepted 20
```

SetFit compares 30 steps, 120 steps, and 120 steps with a smaller embedding learning rate, across the original three development folds. Selection uses pooled development macro-F1. This is a bounded search, not an exhaustive optimization or evidence that training longer must help.

The local LLM compares portrayal rules alone with the same rules plus six examples: two per human sentiment label, retrieved using word/bigram TF-IDF similarity. Training-only retrieval prevents evaluation articles or their labels from appearing in the prompt. Article text is serialized and chat control tokens are escaped. The LLM's thinking mode is disabled for these classification prompts.

The scoring implementation examines the next-token probabilities for A, B, C, and U (Negative, Neutral, Positive, and Uncertain), including single-token alternatives with a leading space. It normalizes over these answer codes. It does not measure the probability that the prediction is correct. U is retained as Uncertain and is never silently converted to Neutral. Context overflow also becomes Uncertain.

Completed LLM predictions and SetFit folds are cached under ignored `ml-runs/reliability-cache/`. Interrupted experiments resume without repeating completed work. Cache keys include model revision, prompt/configuration, dataset identity where relevant, and package versions. After changing scoring semantics, update the runtime recipe to invalidate previous results.

Outputs go to `reports/reliability/`:

| File | Contents |
| --- | --- |
| `setfit_tuning.json` | Three-fold development metrics and selected configuration |
| `*_development.csv` | Out-of-fold SetFit predictions and scores |
| `llm_benchmark.json` | Prompt comparison, model revision, validation and diagnostic metrics |
| `*_validation.csv` | LLM predictions, prompt hashes, and retrieved training IDs |
| `llm_diagnostic.csv` | Predictions on the previously examined historical test partition |
| `summary.json` | Paired comparison, precision/coverage curves, Wilson intervals, and per-label support |
| `RESULTS.md` | Measured results and interpretation |

## What these scores establish

The paired comparison uses **258 training articles and the same 129 validation articles** for the frozen-embedding baseline, selected SetFit configuration, and each LLM prompt. SetFit is selected across all three development folds, so the 129-row result is a development diagnostic, not a new held-out result. LLM examples and prompt selection use development data only.

The original **97 test articles have already been examined** in stage 1. Their reuse is explicitly labeled historical diagnostic. Neither that partition nor thresholds selected on the 129 validation articles certify precision on future articles. A fresh, independently human-labeled sample from a later period is still needed.

The acceptance rule searches score thresholds for at least 90% observed precision with at least 20 accepted validation articles, choosing the largest qualifying coverage. The report records the result as exploratory and keeps `automation_ready=false`. If none qualifies, it says so. Wilson 95% intervals describe sampling uncertainty at each point but do not correct for searching thresholds on the same validation set. Per-label counts expose cases where good overall precision comes mostly from a single label.

A separate agreement check accepts only labels where the selected LLM and selected SetFit configuration agree. It reports coverage and precision rather than assuming agreement eliminates mistakes: the two models can make the same error.

Do not copy a validation threshold directly into future batch inference: refitting a classifier on more annotations or changing the LLM's example pool can change its score distribution. The historical diagnostics apply the validation-selected rule without retuning it on those diagnostic labels and report the resulting precision and coverage.

Scraping can run without assigning every article a sentiment. Store all collected articles; unresolved labels can remain pending or excluded from labeled trend totals, with their count and coverage shown. A rejected article is not Neutral. Review a small random audit sample as well as uncertain cases to detect confidently wrong predictions and changes in later news coverage. Human-reviewed examples can then improve the next model version.

## Try predictions on a new batch

```bash
python -m learning llm-predict --input examples/new_articles.csv --output artifacts/local_llm_predictions.csv
# Optionally request a supporting quotation; this adds generation time:
python -m learning llm-predict --input new_articles.csv --output artifacts/new_predictions.csv --explain
```

The demonstration headlines are fabricated format examples, not evaluation cases. Input requires nonempty `title`, preferably `description`; fields such as `published_at`, `url`, and recorded human labels are preserved. Prediction adds a separate `predicted_sentiment`, raw model score, score type, model identity/revision, prompt hash, retrieval count, and `review_required=True`.

For new articles, retrieval uses all original human annotations. `--examples 0`, `3`, or `6` controls the prompt. Optional evidence is retained only if it is an exact quotation from the supplied text; a supporting quote does not verify a label. Predictions do not overwrite the input or any human annotation file and are not automatically published to the dashboard.

Read [RESULTS.md](../reports/reliability/RESULTS.md) before connecting a model to continuous collection. If this small local model remains weak, that is a measured limit of this configuration, not proof that the dataset cannot support any useful model. A stronger model, clearer original coding boundaries, and a fresh evaluation sample remain separate possible improvements.
