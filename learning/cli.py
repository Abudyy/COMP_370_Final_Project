from __future__ import annotations

import argparse
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from learning.data import (ANNOTATIONS, LABELS, ROOT, TARGET_COLUMNS, article_text, development_folds,
                           experiment_split, file_hash, headline_groups, load_annotations)
from learning.models import BASE_MODEL, BASE_REVISION, MODEL_NAMES, build_model, prediction_scores

DEFAULT_REPORTS = ROOT / "reports/model_comparison"
DEFAULT_ARTIFACTS = ROOT / "artifacts/classifiers"


def metrics(y, predicted, labels):
    return {"accuracy": float(accuracy_score(y, predicted)),
            "macro_f1": float(f1_score(y, predicted, labels=labels, average="macro", zero_division=0)),
            "per_class": classification_report(y, predicted, labels=labels, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, predicted, labels=labels).tolist(), "labels": labels}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def versions():
    names = ["scikit-learn", "numpy", "pandas", "torch", "setfit", "sentence-transformers", "transformers"]
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def compare(args):
    df = load_annotations(args.annotations)
    groups = headline_groups(df)
    dev, test = experiment_split(df, groups, args.seed)
    report = {"complete": False, "dataset_sha256": file_hash(args.annotations), "rows": len(df),
              "development_rows": len(dev), "test_rows": len(test), "seed": args.seed,
              "created_at": datetime.now(timezone.utc).isoformat(), "versions": versions(),
              "base_model": args.base_model, "base_model_revision": BASE_REVISION if args.base_model == BASE_MODEL else None,
              "setfit_steps": args.setfit_steps,
              "headline_group_threshold": .88, "headline_groups": len(set(groups)),
              "sentiment_definition": "Portrayal of Mamdani: favorable / neutral / unfavorable",
              "selection_rule": "Highest pooled macro-F1 on three development folds; final test not used for selection.",
              "automation_ready": False, "tasks": {}}
    args.output.mkdir(parents=True, exist_ok=True)
    split_manifest = df[["article_id", "id"]].copy()
    split_manifest["headline_group"] = groups
    split_manifest["split"] = "development"
    split_manifest.loc[test, "split"] = "test"
    leaderboard, test_predictions, review = [], [], []

    for target, column in TARGET_COLUMNS.items():
        y = df[column].to_numpy()
        folds = development_folds(df, dev, groups, target, args.seed)
        split_manifest[f"{target}_development_fold"] = -1
        task = {"class_counts": df[column].value_counts().to_dict(), "candidates": {}}
        report["tasks"][target] = task
        candidate_predictions = {}
        for name in args.models:
            print(f"{target}: evaluating {name} on development folds", flush=True)
            predicted = np.empty(len(dev), dtype=object)
            fold_scores = []
            try:
                for fold_id, (train, valid) in enumerate(folds):
                    model = build_model(name, args.seed, args.base_model, args.setfit_steps)
                    model.fit(df.text.iloc[dev[train]].tolist(), y[dev[train]])
                    predicted[valid] = model.predict(df.text.iloc[dev[valid]].tolist())
                    fold_scores.append(metrics(y[dev[valid]], predicted[valid], LABELS[target])["macro_f1"])
                    split_manifest.loc[dev[valid], f"{target}_development_fold"] = fold_id
                    print(f"  fold {fold_id + 1}: macro-F1 {fold_scores[-1]:.3f}", flush=True)
                scores = metrics(y[dev], predicted, LABELS[target])
                task["candidates"][name] = {"status": "evaluated", "development": scores,
                                             "fold_macro_f1": fold_scores}
                candidate_predictions[name] = predicted.copy()
                leaderboard.append({"task": target, "model": name, "development_accuracy": scores["accuracy"],
                                    "development_macro_f1": scores["macro_f1"], "evaluated_rows": len(dev)})
            except (ImportError, OSError) as error:
                if name not in {"embedding_lr", "setfit"}:
                    raise
                task["candidates"][name] = {"status": "unavailable", "reason": str(error)[:600]}
                print(f"  unavailable: {error}", flush=True)
            write_json(args.output / "summary.json", report)
        eligible = [name for name in args.models if name != "majority" and name in candidate_predictions]
        if not eligible:
            raise ValueError("No trainable candidate completed evaluation.")
        selected = max(eligible, key=lambda name: task["candidates"][name]["development"]["macro_f1"])
        task["selected_model"] = selected
        print(f"{target}: selected {selected}; fitting development data for the held-out test", flush=True)
        model = build_model(selected, args.seed, args.base_model, args.setfit_steps)
        model.fit(df.text.iloc[dev].tolist(), y[dev])
        predicted = model.predict(df.text.iloc[test].tolist())
        task["held_out_test"] = metrics(y[test], predicted, LABELS[target])
        dummy = build_model("majority", args.seed)
        dummy.fit(df.text.iloc[dev].tolist(), y[dev])
        task["held_out_majority"] = metrics(y[test], dummy.predict(df.text.iloc[test].tolist()), LABELS[target])
        scores, score_kind = prediction_scores(model, df.text.iloc[test].tolist())
        for index, prediction, score in zip(test, predicted, scores):
            test_predictions.append({"article_id": df.article_id.iloc[index], "task": target, "model": selected,
                                     "outlet_id": df.id.iloc[index], "title": df.title.iloc[index],
                                     "true_label": y[index], "predicted_label": prediction,
                                     "model_score": float(score), "score_kind": score_kind})
        wrong = dev[candidate_predictions[selected] != y[dev]]
        # Inspect development errors first. Looking repeatedly at test errors turns the test into development data.
        for index in wrong[:40]:
            local = int(np.where(dev == index)[0][0])
            review.append({"article_id": df.article_id.iloc[index], "task": target, "outlet_id": df.id.iloc[index],
                           "title": df.title.iloc[index], "description": df.description.iloc[index],
                           "recorded_label": y[index], "development_prediction": candidate_predictions[selected][local],
                           "reviewed_label": "", "review_notes": ""})
        write_json(args.output / "summary.json", report)
    report["complete"] = True
    write_json(args.output / "summary.json", report)
    split_manifest.to_csv(args.output / "splits.csv", index=False)
    pd.DataFrame(leaderboard).to_csv(args.output / "leaderboard.csv", index=False)
    pd.DataFrame(test_predictions).to_csv(args.output / "test_predictions.csv", index=False)
    pd.DataFrame(review).to_csv(args.output / "review_candidates.csv", index=False)
    write_results(args.output, report)
    print(f"Comparison complete. Read {args.output / 'RESULTS.md'}", flush=True)


def write_results(output, report):
    lines = ["# First model comparison", "", f"Dataset: {report['rows']} original human annotations.", "",
             f"Development: {report['development_rows']} rows; held-out test: {report['test_rows']} rows.", "",
             "Candidates are selected using three development folds and macro-F1. The selected model is then evaluated "
             "on the held-out test. Near-identical headline groups stay in one partition. TF-IDF is fitted inside each training fold.", "",
             "| Task | Selected model | Development macro-F1 | Test accuracy | Test macro-F1 | Test majority accuracy |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for target, task in report["tasks"].items():
        selected = task["selected_model"]
        lines.append(f"| {target} | {selected} | {task['candidates'][selected]['development']['macro_f1']:.3f} | "
                     f"{task['held_out_test']['accuracy']:.1%} | {task['held_out_test']['macro_f1']:.3f} | "
                     f"{task['held_out_majority']['accuracy']:.1%} |")
    lines += ["", "## Scope of these results", "",
              "These results measure held-out articles from the same historical collection, not future-period performance. "
              "The test is small, so scores have substantial uncertainty. Near-identical headline matching does not identify "
              "all shared events or syndication. There is no temporal holdout because publication dates are absent.", "",
              "Sentiment means portrayal of Mamdani (confirmed by the user), not general emotional tone. Existing labels "
              "are unchanged. Codebook consistency and inter-rater agreement have not been independently established.", "",
              "Automatic labeling has not been validated by this benchmark. Predictions are marked as requiring review; "
              "model scores are uncalibrated. Validate on newly collected articles and inspect precision/recall before "
              "setting an acceptance threshold. Model predictions are not new human training labels.", "",
              "Use review_candidates.csv for development-error inspection. test_predictions.csv is an audit artifact; "
              "avoid repeatedly tuning against those test errors. If the test informs future changes, create a fresh test set.", "",
              "## Files", "", "- summary.json: configuration, metrics, class reports, confusion matrices, and package versions.",
              "- leaderboard.csv: development comparison only.", "- splits.csv: reproducible partitions and fold assignments.",
              "- test_predictions.csv: held-out predictions from the selected models.",
              "- review_candidates.csv: up to 40 development mistakes per task, with blank review fields.", ""]
    stress_path = output / "source_stress.json"
    if stress_path.exists():
        stress = json.loads(stress_path.read_text())
        if stress.get("comparison_created_at") == report["created_at"]:
            lines += ["## Exploratory check on excluded outlets", "",
                      "| Task | Accuracy | Macro-F1 |", "| --- | ---: | ---: |"]
            for target, task in stress["tasks"].items():
                lines.append(f"| {target} | {task['metrics']['accuracy']:.1%} | {task['metrics']['macro_f1']:.3f} |")
            lines += ["", "These five folds exclude entire outlets and overlapping near-identical headlines from training. "
                      "The selected models are unchanged. This is exploratory historical evidence, not a future-period test.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def train(args):
    report = json.loads((args.report / "summary.json").read_text())
    if not report.get("complete") or report["dataset_sha256"] != file_hash(args.annotations):
        raise ValueError("Run a complete comparison on this annotation file before training.")
    df = load_annotations(args.annotations)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset_sha256": report["dataset_sha256"], "trained_rows": len(df),
                "trained_at": datetime.now(timezone.utc).isoformat(), "versions": versions(),
                "model_version": "experiment-" + report["dataset_sha256"][:10] + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                "automation_ready": False, "base_model": report["base_model"],
                "base_model_revision": report.get("base_model_revision"),
                "comparison_created_at": report["created_at"],
                "sentiment_definition": report["sentiment_definition"], "tasks": {}}
    for target, column in TARGET_COLUMNS.items():
        name = report["tasks"][target]["selected_model"]
        print(f"Fitting {target}: {name} on {len(df)} human-labeled rows", flush=True)
        model = build_model(name, report["seed"], report["base_model"], report["setfit_steps"])
        model.fit(df.text.tolist(), df[column].to_numpy())
        joblib.dump(model, args.output / f"{target}.joblib")
        manifest["tasks"][target] = {"model": name, "labels": LABELS[target],
                                     "held_out_test": report["tasks"][target]["held_out_test"]}
    write_json(args.output / "manifest.json", manifest)
    print(f"Experimental artifacts saved in {args.output}", flush=True)


def source_check(args):
    report = json.loads((args.report / "summary.json").read_text())
    if not report.get("complete") or report["dataset_sha256"] != file_hash(args.annotations):
        raise ValueError("Run a complete comparison on this annotation file before the outlet check.")
    df = load_annotations(args.annotations)
    groups = headline_groups(df)
    folds = list(GroupKFold(n_splits=5).split(df, groups=df.id))
    result = {"dataset_sha256": report["dataset_sha256"], "rows": len(df), "tasks": {},
              "comparison_created_at": report["created_at"],
              "purpose": "Exploratory outlet generalization check; not used to select models and not a future-period test."}
    for target, column in TARGET_COLUMNS.items():
        name = report["tasks"][target]["selected_model"]
        y = df[column].to_numpy()
        predicted = np.empty(len(df), dtype=object)
        details = []
        for fold_id, (training, validation) in enumerate(folds):
            # Remove near-identical headlines from training even if they belong to a different outlet.
            training = training[~np.isin(groups[training], groups[validation])]
            if set(y[training]) != set(LABELS[target]):
                raise ValueError(f"An outlet-check training fold is missing {target} classes.")
            assert set(df.id.iloc[training]).isdisjoint(df.id.iloc[validation])
            assert set(groups[training]).isdisjoint(groups[validation])
            print(f"{target}: outlet check fold {fold_id + 1} with {name}", flush=True)
            model = build_model(name, report["seed"], report["base_model"], report["setfit_steps"])
            model.fit(df.text.iloc[training].tolist(), y[training])
            predicted[validation] = model.predict(df.text.iloc[validation].tolist())
            details.append({"excluded_outlets": sorted(df.id.iloc[validation].unique()),
                            "training_rows": len(training), "test_rows": len(validation),
                            "accuracy": float(accuracy_score(y[validation], predicted[validation]))})
        result["tasks"][target] = {"model": name, "metrics": metrics(y, predicted, LABELS[target]), "folds": details}
    write_json(args.report / "source_stress.json", result)
    write_results(args.report, report)
    print(f"Outlet check saved in {args.report / 'source_stress.json'}", flush=True)


def predict(args):
    if args.input.resolve() == args.output.resolve() or args.output.resolve() == ANNOTATIONS.resolve():
        raise ValueError("Prediction output must not overwrite input data or the original annotations.")
    df = pd.read_csv(args.input, keep_default_na=False)
    if "title" not in df or df.title.astype(str).str.strip().eq("").any() or df.empty:
        raise ValueError("Input CSV needs non-empty title values.")
    descriptions = df.description if "description" in df else pd.Series([""] * len(df))
    texts = article_text(df.title, descriptions)
    manifest = json.loads((args.artifacts / "manifest.json").read_text())
    for target in TARGET_COLUMNS:
        model = joblib.load(args.artifacts / f"{target}.joblib")
        df[f"predicted_{target}"] = model.predict(texts)
        scores, kind = prediction_scores(model, texts)
        df[f"{target}_model_score"] = scores
        df[f"{target}_score_kind"] = kind
    df["label_source"] = "model"
    df["model_version"] = manifest["model_version"]
    df["review_required"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} experimental predictions to {args.output}; every row requires review.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Compare, train, and run experimental article classifiers.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("compare")
    p.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    p.add_argument("--output", type=Path, default=DEFAULT_REPORTS)
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--setfit-steps", type=int, default=30)
    p.set_defaults(function=compare)
    p = sub.add_parser("train")
    p.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORTS)
    p.add_argument("--output", type=Path, default=DEFAULT_ARTIFACTS)
    p.set_defaults(function=train)
    p = sub.add_parser("source-check")
    p.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORTS)
    p.set_defaults(function=source_check)
    p = sub.add_parser("predict")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    p.set_defaults(function=predict)
    args = parser.parse_args()
    args.function(args)
