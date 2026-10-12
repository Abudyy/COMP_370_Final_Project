from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from learning.cli import metrics, versions, write_json
from learning.data import ANNOTATIONS, LABELS, ROOT, article_text, development_folds, file_hash, headline_groups, load_annotations
from learning.llm import MODEL_FILE, MODEL_REPO, MODEL_REVISION, RULES, ExampleRetriever, LocalLLM, prompt_for, prompt_hash
from learning.models import BASE_MODEL, BASE_REVISION, SetFitClassifier, build_model, prediction_scores
from learning.selection import choose_threshold, coverage_curve, coverage_point

OUTPUT = ROOT / "reports/reliability"
CACHE = ROOT / "ml-runs/reliability-cache"
CONFIGS = [
    {"name": "setfit_30", "steps": 30, "body_learning_rate": 2e-5},
    {"name": "setfit_120", "steps": 120, "body_learning_rate": 2e-5},
    {"name": "setfit_120_low_lr", "steps": 120, "body_learning_rate": 1e-5},
]


def protocol(annotations, original_report):
    df = load_annotations(annotations)
    prior = json.loads((original_report / "summary.json").read_text())
    if file_hash(annotations) != prior["dataset_sha256"]:
        raise ValueError("The original comparison must match the current annotations.")
    manifest = pd.read_csv(original_report / "splits.csv", keep_default_na=False)
    lookup = manifest.set_index("article_id")
    if set(lookup.index) != set(df.article_id):
        raise ValueError("The split manifest does not match this dataset.")
    split = lookup.loc[df.article_id, "split"].to_numpy()
    dev, test = np.flatnonzero(split == "development"), np.flatnonzero(split == "test")
    groups = headline_groups(df)
    folds = development_folds(df, dev, groups, "sentiment", prior["seed"])
    return df, prior, groups, dev, test, folds


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def fit_setfit(config, seed):
    return SetFitClassifier(seed=seed, max_steps=config["steps"], body_learning_rate=config["body_learning_rate"])


def tune(args):
    df, prior, groups, dev, test, folds = protocol(args.annotations, args.original_report)
    args.output.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    report = {"complete": False, "dataset_sha256": prior["dataset_sha256"], "seed": prior["seed"],
              "versions": versions(), "rows": len(df), "development_rows": len(dev),
              "base_model": BASE_MODEL, "base_revision": BASE_REVISION, "candidates": {},
              "task": "sentiment", "selection_rule": "Pooled macro-F1 over three development folds."}
    for config in CONFIGS:
        predictions = np.empty(len(dev), dtype=object)
        scores = np.zeros(len(dev))
        for number, (training, validation) in enumerate(folds):
            key = fingerprint({"dataset": prior["dataset_sha256"], "config": config, "fold": number,
                               "seed": prior["seed"], "versions": report["versions"], "revision": BASE_REVISION})
            path = CACHE / f"{key}.json"
            if path.exists():
                result = json.loads(path.read_text())
                print(f"{config['name']} fold {number + 1}: using cached evaluation", flush=True)
            else:
                print(f"{config['name']} fold {number + 1}: training {config['steps']} steps", flush=True)
                model = fit_setfit(config, prior["seed"])
                model.fit(df.text.iloc[dev[training]].tolist(), df.polarity.iloc[dev[training]].to_numpy())
                pred = model.predict(df.text.iloc[dev[validation]].tolist())
                probabilities, _ = prediction_scores(model, df.text.iloc[dev[validation]].tolist())
                result = {"predictions": pred.tolist(), "scores": probabilities.tolist()}
                write_json(path, result)
            predictions[validation] = result["predictions"]
            scores[validation] = result["scores"]
        evaluated = metrics(df.polarity.iloc[dev], predictions, LABELS["sentiment"])
        report["candidates"][config["name"]] = {"config": config, "development": evaluated}
        table = pd.DataFrame({"article_id": df.article_id.iloc[dev].to_numpy(), "row": dev,
                              "true_label": df.polarity.iloc[dev].to_numpy(), "prediction": predictions, "score": scores})
        table.to_csv(args.output / f"{config['name']}_development.csv", index=False)
        write_json(args.output / "setfit_tuning.json", report)
        print(f"{config['name']}: development macro-F1 {evaluated['macro_f1']:.3f}", flush=True)
    selected = max(report["candidates"], key=lambda name: report["candidates"][name]["development"]["macro_f1"])
    report["selected_model"] = selected
    report["complete"] = True
    write_json(args.output / "setfit_tuning.json", report)
    print(f"Broader SetFit comparison complete: {selected}", flush=True)


def cached_prediction(llm, prompt, cache, provenance):
    key = fingerprint({**provenance, "prompt_sha256": prompt_hash(prompt)})
    path = cache / (key + ".json")
    if path.exists():
        return json.loads(path.read_text())
    start = time.monotonic()
    result = llm.score(prompt)
    result["elapsed_seconds"] = time.monotonic() - start
    result["prompt_sha256"] = prompt_hash(prompt)
    write_json(path, result)
    return result


def evaluate_llm(llm, df, training, evaluation, examples, cache, phase):
    retriever = ExampleRetriever(df.iloc[training])
    provenance = {"model": MODEL_REPO, "revision": MODEL_REVISION, "file": MODEL_FILE,
                  "runtime": "native-logits-grouped-answer-probability-v1", "versions": versions()}
    rows = []
    for position, index in enumerate(evaluation):
        article = df.iloc[index]
        retrieved = retriever.retrieve(article.text, examples)
        assert all(example.article_id not in set(df.article_id.iloc[evaluation]) for example in retrieved)
        prompt = prompt_for(article.title, article.description, retrieved)
        result = cached_prediction(llm, prompt, cache, provenance)
        rows.append({"article_id": article.article_id, "row": int(index), "true_label": article.polarity,
                     "prediction": result["prediction"], "score": result["score"], "reason": result["reason"],
                     "prompt_tokens": result["prompt_tokens"], "prompt_sha256": result["prompt_sha256"],
                     "example_ids": "|".join(example.article_id for example in retrieved)})
        if position % 10 == 0 or position + 1 == len(evaluation):
            print(f"{phase}, {examples} examples: {position + 1}/{len(evaluation)} evaluated", flush=True)
    return pd.DataFrame(rows)


def benchmark_llm(args):
    df, prior, groups, dev, test, folds = protocol(args.annotations, args.original_report)
    training, validation = folds[0]
    training, validation = dev[training], dev[validation]
    args.output.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    llm = LocalLLM(threads=args.threads)
    report = {"complete": False, "dataset_sha256": prior["dataset_sha256"], "task": "sentiment",
              "versions": versions(), "runtime_recipe": "native-logits-grouped-answer-probability-v1",
              "context_tokens": llm.context, "threads": args.threads, "thinking_mode": False,
              "model_repo": MODEL_REPO, "revision": MODEL_REVISION, "filename": MODEL_FILE,
              "score_kind": "uncalibrated_probability_over_answer_codes", "rules_sha256": prompt_hash(RULES),
              "validation_rows": len(validation), "training_rows": len(training), "diagnostic_rows": len(test),
              "validation_article_ids": df.article_id.iloc[validation].tolist(), "candidates": {},
              "test_status": "Previously examined historical partition; diagnostic, not a fresh independent test.",
              "retrieval": "Balanced word/bigram TF-IDF examples drawn exclusively from the training partition."}
    for count in args.examples:
        table = evaluate_llm(llm, df, training, validation, count, CACHE, "validation")
        name = f"local_llm_{count}_examples"
        table.to_csv(args.output / f"{name}_validation.csv", index=False)
        report["candidates"][name] = {"examples": count,
                                      "validation": metrics(table.true_label, table.prediction, LABELS["sentiment"]),
                                      "uncertain_fraction": float(table.prediction.eq("Uncertain").mean())}
        write_json(args.output / "llm_benchmark.json", report)
    selected = max(report["candidates"], key=lambda name: report["candidates"][name]["validation"]["macro_f1"])
    report["selected_model"] = selected
    count = report["candidates"][selected]["examples"]
    table = evaluate_llm(llm, df, dev, test, count, CACHE, "historical diagnostic")
    table.to_csv(args.output / "llm_diagnostic.csv", index=False)
    report["diagnostic"] = metrics(table.true_label, table.prediction, LABELS["sentiment"])
    report["complete"] = True
    write_json(args.output / "llm_benchmark.json", report)
    print(f"Local LLM benchmark complete: {selected}", flush=True)


def summarize(args):
    if not 0 < args.target_precision <= 1 or args.minimum_accepted < 1:
        raise ValueError("Use a precision target between 0 and 1 and a positive minimum accepted count.")
    df, prior, groups, dev, test, folds = protocol(args.annotations, args.original_report)
    tuning = json.loads((args.output / "setfit_tuning.json").read_text())
    llm = json.loads((args.output / "llm_benchmark.json").read_text())
    if not tuning["complete"] or not llm["complete"]:
        raise ValueError("Finish both experiments before creating the combined report.")
    if tuning["dataset_sha256"] != prior["dataset_sha256"] or llm["dataset_sha256"] != prior["dataset_sha256"]:
        raise ValueError("Experiment data hashes must match.")
    training, validation = folds[0]
    validation_rows = dev[validation]
    y = df.polarity.iloc[validation_rows].to_numpy()
    candidates = {}
    selected_setfit = tuning["selected_model"]
    sft = pd.read_csv(args.output / f"{selected_setfit}_development.csv").set_index("row").loc[validation_rows]
    candidates[selected_setfit] = {"prediction": sft.prediction.to_numpy(), "score": sft.score.to_numpy()}
    for name in llm["candidates"]:
        table = pd.read_csv(args.output / f"{name}_validation.csv").set_index("row").loc[validation_rows]
        candidates[name] = {"prediction": table.prediction.to_numpy(), "score": table.score.to_numpy()}
    # A paired baseline uses the same training/validation partition as every LLM prompt.
    baseline = build_model("embedding_lr", prior["seed"])
    baseline.fit(df.text.iloc[dev[training]].tolist(), df.polarity.iloc[dev[training]].to_numpy())
    pred = baseline.predict(df.text.iloc[validation_rows].tolist())
    scores, _ = prediction_scores(baseline, df.text.iloc[validation_rows].tolist())
    candidates["embedding_lr"] = {"prediction": pred, "score": scores}
    report = {"complete": True, "dataset_sha256": prior["dataset_sha256"], "task": "sentiment",
              "validation_rows": len(validation_rows), "diagnostic_rows": len(test), "candidates": {},
              "target_precision": args.target_precision, "minimum_accepted": args.minimum_accepted,
              "automation_ready": False,
              "limitations": ["Only one 129-row development fold is paired with the local LLM.",
                              "The 97-row historical diagnostic partition has already been examined.",
                              "Validation-selected thresholds and their intervals are exploratory, not independent certification.",
                              "A fresh human-labeled article sample is needed before ongoing automatic labeling."]}
    for name, result in candidates.items():
        curve = coverage_curve(y, result["prediction"], result["score"], LABELS["sentiment"])
        chosen = choose_threshold(curve, args.target_precision, args.minimum_accepted)
        report["candidates"][name] = {"validation": metrics(y, result["prediction"], LABELS["sentiment"]),
                                     "coverage_curve": curve, "validation_operating_point": chosen}
    # Agreement is a separate abstention strategy, not evidence that errors are independent.
    llm_selected = candidates[llm["selected_model"]]
    agreement = np.asarray(llm_selected["prediction"]) == np.asarray(candidates[selected_setfit]["prediction"])
    agreed_predictions = np.where(agreement, llm_selected["prediction"], "Uncertain")
    report["agreement_check"] = {"models": [selected_setfit, llm["selected_model"]],
                                 "validation": coverage_point(y, agreed_predictions, np.ones(len(y)), 0, LABELS["sentiment"])}
    selected = max(report["candidates"], key=lambda name: report["candidates"][name]["validation"]["macro_f1"])
    report["selected_by_validation_macro_f1"] = selected
    # Diagnostics for the separately selected SetFit configuration, using development rows only for fitting.
    config = tuning["candidates"][selected_setfit]["config"]
    key = fingerprint({"dataset": prior["dataset_sha256"], "config": config, "role": "historical-diagnostic",
                       "seed": prior["seed"], "versions": versions(), "revision": BASE_REVISION})
    path = CACHE / (key + ".json")
    if path.exists():
        diagnostic = json.loads(path.read_text())
    else:
        model = fit_setfit(config, prior["seed"])
        model.fit(df.text.iloc[dev].tolist(), df.polarity.iloc[dev].to_numpy())
        predictions = model.predict(df.text.iloc[test].tolist())
        scores, _ = prediction_scores(model, df.text.iloc[test].tolist())
        diagnostic = {"prediction": predictions.tolist(), "score": scores.tolist()}
        write_json(path, diagnostic)
    diagnostics = {selected_setfit: diagnostic}
    llm_table = pd.read_csv(args.output / "llm_diagnostic.csv").set_index("row").loc[test]
    diagnostics[llm["selected_model"]] = {"prediction": llm_table.prediction.to_numpy(), "score": llm_table.score.to_numpy()}
    for name, result in diagnostics.items():
        report["candidates"][name]["historical_diagnostic"] = metrics(df.polarity.iloc[test], result["prediction"], LABELS["sentiment"])
        operating = report["candidates"][name]["validation_operating_point"]
        if operating is not None:
            report["candidates"][name]["diagnostic_operating_point"] = coverage_point(
                df.polarity.iloc[test], result["prediction"], result["score"], operating["threshold"], LABELS["sentiment"])
    agreed = np.asarray(diagnostics[selected_setfit]["prediction"]) == np.asarray(diagnostics[llm["selected_model"]]["prediction"])
    predictions = np.where(agreed, diagnostics[selected_setfit]["prediction"], "Uncertain")
    report["agreement_check"]["historical_diagnostic"] = coverage_point(
        df.polarity.iloc[test], predictions, np.ones(len(test)), 0, LABELS["sentiment"])
    write_json(args.output / "summary.json", report)
    example_counts = ", ".join(str(result["examples"]) for result in llm["candidates"].values())
    lines = ["# Sentiment reliability experiments", "",
             "The local LLM uses Qwen3 4B Q4_K_M with a pinned revision, explicit target-portrayal rules, "
             f"and {example_counts} balanced TF-IDF-retrieved human examples. Retrieval is confined to training data.", "",
             "## SetFit tuning", "",
             "SetFit settings are selected by pooled macro-F1 across three development folds (387 articles).", "",
             "| Configuration | Development accuracy | Development macro-F1 |",
             "| --- | ---: | ---: |"]
    for name, result in tuning["candidates"].items():
        evaluated = result["development"]
        lines.append(f"| {name} | {evaluated['accuracy']:.1%} | {evaluated['macro_f1']:.3f} |")
    lines += ["", f"Selected SetFit configuration: `{selected_setfit}`.", "", "## Paired development comparison", "",
             "The following paired comparison "
             f"uses the same {len(validation_rows)} validation articles and training partition for all candidates.", "",
             "| Candidate | Validation accuracy | Macro-F1 | Accepted coverage at target precision |",
             "| --- | ---: | ---: | ---: |"]
    for name, result in report["candidates"].items():
        operating = result["validation_operating_point"]
        coverage = f"{operating['coverage']:.1%} ({operating['accepted']} articles)" if operating else "No qualifying threshold"
        lines.append(f"| {name} | {result['validation']['accuracy']:.1%} | {result['validation']['macro_f1']:.3f} | {coverage} |")
    agreement = report["agreement_check"]["validation"]
    agreement_precision = f"{agreement['precision']:.1%}" if agreement["precision"] is not None else "undefined"
    lines += ["", f"Accepting only labels where selected SetFit and the selected LLM agree gives "
              f"{agreement_precision} observed validation precision at {agreement['coverage']:.1%} coverage "
              f"({agreement['accepted']} articles). This is exploratory; the two models can share errors.",
              "", f"Precision target: {args.target_precision:.0%}; minimum accepted validation articles: {args.minimum_accepted}.",
              "A qualifying threshold is an exploratory validation result, not a claim of guaranteed precision. "
              "summary.json includes Wilson intervals and per-label support. The same validation data selected these thresholds, "
              "so the intervals do not account for threshold selection.", "", "## Historical diagnostics", "",
              "The original 97-row test has already been examined. These results are explicitly diagnostic, "
              "not fresh independent validation of a new model or an acceptance rule.", "",
              "| Candidate | Diagnostic accuracy | Diagnostic macro-F1 | Precision at validation threshold | Accepted coverage |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for name, result in report["candidates"].items():
        if "historical_diagnostic" in result:
            diagnostic = result["historical_diagnostic"]
            point = result.get("diagnostic_operating_point")
            precision = f"{point['precision']:.1%}" if point and point["precision"] is not None else "No qualifying validation threshold"
            coverage = f"{point['coverage']:.1%} ({point['accepted']} articles)" if point else "—"
            lines.append(f"| {name} | {diagnostic['accuracy']:.1%} | {diagnostic['macro_f1']:.3f} | {precision} | {coverage} |")
    lines += ["", "## Interpretation", "",
              "Uncertain predictions are never silently converted to Neutral. Rejected predictions remain stored "
              "and must be counted when showing sentiment coverage; selective acceptance can bias trends. "
              "Model scores are uncalibrated and differ between SetFit and the restricted-token LLM scorer.", "",
              "No existing human labels or research charts are replaced. Topic classification is unchanged in this experiment. "
              "Ongoing collection and automated publication are not enabled by these historical benchmarks.", ""]
    (args.output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Combined reliability report: {args.output / 'RESULTS.md'}", flush=True)


def predict_llm(args):
    if args.output.resolve() in {args.input.resolve(), ANNOTATIONS.resolve(), args.annotations.resolve()}:
        raise ValueError("Prediction output must not overwrite input or human annotations.")
    df = pd.read_csv(args.input, keep_default_na=False, dtype=str)
    if "title" not in df or df.empty or not df.title.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Input needs non-empty headlines.")
    annotations = load_annotations(args.annotations)
    retriever = ExampleRetriever(annotations)
    model = LocalLLM(args.threads)
    rows = []
    for article in df.to_dict(orient="records"):
        description = article.get("description", "")
        text = article_text([article["title"]], [description])[0]
        examples = retriever.retrieve(text, args.examples)
        prompt = prompt_for(article["title"], description, examples)
        result = model.score(prompt)
        evidence = model.explain(article["title"], description, result["prediction"]) if args.explain and result["reason"] == "scored" else ""
        article.update({"predicted_sentiment": result["prediction"], "sentiment_model_score": result["score"],
                        "score_kind": "uncalibrated_probability_over_answer_codes", "label_source": "local_llm",
                        "model_revision": MODEL_REVISION, "model_repo": MODEL_REPO, "model_file": MODEL_FILE,
                        "annotations_sha256": file_hash(args.annotations),
                        "retrieval_count": args.examples, "prompt_sha256": prompt_hash(prompt),
                        "retrieved_example_ids": "|".join(example.article_id for example in examples),
                        "evidence": evidence, "review_required": True})
        rows.append(article)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} experimental local-LLM predictions", flush=True)


def add_commands(sub):
    for name, function in [("reliability-setfit", tune), ("reliability-llm", benchmark_llm), ("reliability-summary", summarize)]:
        p = sub.add_parser(name)
        p.add_argument("--annotations", type=Path, default=ANNOTATIONS)
        p.add_argument("--original-report", type=Path, default=ROOT / "reports/model_comparison")
        p.add_argument("--output", type=Path, default=OUTPUT)
        if name == "reliability-llm":
            p.add_argument("--threads", type=int, default=6)
            p.add_argument("--examples", nargs="+", type=int, choices=[0, 3, 6], default=[0, 6])
        if name == "reliability-summary":
            p.add_argument("--target-precision", type=float, default=.9)
            p.add_argument("--minimum-accepted", type=int, default=20)
        p.set_defaults(function=function)
    p = sub.add_parser("llm-predict")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--examples", type=int, choices=[0, 3, 6], default=6)
    p.add_argument("--explain", action="store_true")
    p.set_defaults(function=predict_llm)
