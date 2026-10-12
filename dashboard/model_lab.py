"""Display the committed benchmark; importing this never trains or downloads a model."""
import hashlib
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_model_lab(root):
    st.markdown("### How well can models reproduce the human labels?")
    st.caption("This is a fixed evaluation of the historical annotations. Exploration filters do not change this benchmark.")
    folder = root / "reports/model_comparison"
    path = folder / "summary.json"
    if not path.exists():
        st.info("The model comparison is being prepared.")
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("complete"):
        st.info("Model evaluation is still in progress.")
        return
    if hashlib.sha256((root / "data/annotation/final_full_annotated.csv").read_bytes()).hexdigest() != report["dataset_sha256"]:
        st.warning("The annotations have changed since this benchmark. These scores describe an earlier dataset.")
        return
    st.info("Experimental predictions. These scores describe held-out articles from the historical sample; "
            "performance on newly published articles has not been established.")
    left, right = st.columns(2)
    left.metric("Development articles", report["development_rows"], border=True)
    right.metric("Held-out test articles", report["test_rows"], border=True)
    st.caption("Models are compared using three development folds. The selected model is then tested on a separate "
               "partition. Near-identical headlines stay together. Macro-F1 gives each label equal weight.")
    task_name = st.radio("Prediction task", ["Sentiment", "Topic"], horizontal=True, key="benchmark_task")
    task = report["tasks"][task_name.lower()]
    candidates = []
    for name, results in task["candidates"].items():
        if results["status"] == "evaluated":
            candidates.append({"Model": name, "Development accuracy": results["development"]["accuracy"],
                               "Development macro-F1": results["development"]["macro_f1"],
                               "Selected": name == task["selected_model"]})
    st.dataframe(pd.DataFrame(candidates), hide_index=True, width="stretch",
                 column_config={"Development accuracy": st.column_config.NumberColumn(format="%.3f"),
                                "Development macro-F1": st.column_config.NumberColumn(format="%.3f")})
    test = task["held_out_test"]
    st.markdown(f"#### Held-out result: {task['selected_model']}")
    a, b, c = st.columns(3)
    a.metric("Test accuracy", f"{test['accuracy']:.1%}", border=True)
    b.metric("Test macro-F1", f"{test['macro_f1']:.3f}", border=True)
    c.metric("Majority-label accuracy", f"{task['held_out_majority']['accuracy']:.1%}", border=True)
    st.caption("The majority baseline predicts the most common training label for every test article.")
    chart, table = st.columns([1.2, 1])
    with chart:
        fig = go.Figure(go.Heatmap(z=test["confusion_matrix"], x=test["labels"], y=test["labels"],
                                  colorscale=[[0, "#F5F2EC"], [1, "#6C5BA7"]], texttemplate="%{z}",
                                  hovertemplate="Human label: %{y}<br>Prediction: %{x}<br>%{z} articles<extra></extra>"))
        fig.update_layout(height=380, xaxis_title="Predicted label", yaxis_title="Human label",
                          paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=15, b=15))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch", key=f"benchmark_confusion_{task_name}")
    with table:
        rows = [{"Label": label, "Precision": test["per_class"][label]["precision"],
                 "Recall": test["per_class"][label]["recall"], "F1": test["per_class"][label]["f1-score"],
                 "Test articles": int(test["per_class"][label]["support"])} for label in test["labels"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={col: st.column_config.NumberColumn(format="%.3f") for col in ["Precision", "Recall", "F1"]})
    st.caption("The test sample is small. These results do not establish inter-rater agreement or future-period reliability.")
    stress_path = folder / "source_stress.json"
    if stress_path.exists():
        stress = json.loads(stress_path.read_text())
        if (stress.get("dataset_sha256") == report["dataset_sha256"]
                and stress.get("comparison_created_at") == report["created_at"]
                and stress["tasks"][task_name.lower()]["model"] == task["selected_model"]):
            st.markdown("#### Generalizing to outlets excluded from training")
            stress_metrics = stress["tasks"][task_name.lower()]["metrics"]
            a, b = st.columns(2)
            a.metric("Excluded-outlet accuracy", f"{stress_metrics['accuracy']:.1%}", border=True)
            b.metric("Excluded-outlet macro-F1", f"{stress_metrics['macro_f1']:.3f}", border=True)
            st.caption("Exploratory five-fold check: each outlet is evaluated using a model trained on other outlets. "
                       "Near-identical headlines are excluded from training. This check does not establish future-period performance.")
    with st.expander("Examples where the model disagreed during development"):
        review_path = folder / "review_candidates.csv"
        if review_path.exists():
            review = pd.read_csv(review_path, keep_default_na=False)
            subset = review[review.task == task_name.lower()]
            st.caption("A disagreement can reflect a model mistake, ambiguous wording, or inconsistent coding. "
                       "Original labels are preserved.")
            st.dataframe(subset[["outlet_id", "title", "recorded_label", "development_prediction"]],
                         hide_index=True, width="stretch")
    if task_name == "Sentiment":
        render_reliability(root, report["dataset_sha256"])


def render_reliability(root, dataset_hash):
    path = root / "reports/reliability/summary.json"
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("complete") or report.get("dataset_sha256") != dataset_hash:
        return
    st.divider()
    st.markdown("### Can we trust automatic sentiment labels?")
    st.caption("A local language model applies portrayal rules, with or without retrieved human examples. "
               "A broader SetFit search tests longer training and a smaller learning rate.")
    st.info(f"This comparison uses the same {report['validation_rows']} development articles for every model. "
            "The original test articles have already been examined; a fresh sample is needed before automatic publication.")
    names = {"embedding_lr": "Frozen embeddings + logistic regression", "setfit_30": "SetFit · 30 steps",
             "setfit_120": "SetFit · 120 steps", "setfit_120_low_lr": "SetFit · 120 steps, smaller learning rate",
             "local_llm_0_examples": "Local LLM · rules only", "local_llm_3_examples": "Local LLM · 3 human examples",
             "local_llm_6_examples": "Local LLM · 6 human examples"}
    target = report["target_precision"]
    rows = []
    for name, result in report["candidates"].items():
        operating = result["validation_operating_point"]
        rows.append({"Model": names.get(name, name), "Validation accuracy": f"{result['validation']['accuracy']:.1%}",
                     "Macro-F1": round(result["validation"]["macro_f1"], 3),
                     f"Coverage at {target:.0%} precision": f"{operating['coverage']:.1%}" if operating else "No qualifying threshold"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"The precision target means correct labels among accepted predictions. At least "
               f"{report['minimum_accepted']} validation articles must be accepted. Uncertain counts as an incorrect "
               "raw prediction and is excluded from accepted labels.")
    selected = st.selectbox("Inspect precision and coverage", list(report["candidates"]),
                            index=list(report["candidates"]).index(report["selected_by_validation_macro_f1"]),
                            format_func=lambda name: names.get(name, name), key="reliability_model")
    result = report["candidates"][selected]
    curve = sorted((p for p in result["coverage_curve"] if p["accepted"]), key=lambda p: p["coverage"])
    fig = go.Figure(go.Scatter(
        x=[p["coverage"] for p in curve], y=[p["precision"] for p in curve], mode="lines+markers",
        line=dict(color="#6C5BA7"), marker=dict(size=5),
        customdata=[[p["threshold"], p["accepted"], p["interval"][0], p["interval"][1]] for p in curve],
        hovertemplate="Coverage: %{x:.1%}<br>Precision: %{y:.1%}<br>Accepted: %{customdata[1]}"
                      "<br>Score threshold: %{customdata[0]:.9f}<br>95% interval: %{customdata[2]:.1%}–%{customdata[3]:.1%}<extra></extra>"))
    fig.add_hline(y=target, line_dash="dot", line_color="#426E62", annotation_text=f"{target:.0%} target")
    fig.update_layout(height=340, xaxis_title="Share of articles accepted", yaxis_title="Correct labels among accepted articles",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=20, b=10))
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_yaxes(range=[0, 1.04], tickformat=".0%")
    st.plotly_chart(fig, width="stretch", key="reliability_coverage")
    operating = result["validation_operating_point"]
    supported = [p for p in curve if p["accepted"] >= report["minimum_accepted"]]
    point = operating or (max(supported, key=lambda p: (p["precision"], p["accepted"])) if supported else None)
    if operating is None:
        st.warning(f"This model has no evaluated threshold reaching {target:.0%} precision with at least "
                   f"{report['minimum_accepted']} accepted validation articles.")
    if point:
        a, b, c = st.columns(3)
        a.metric("Observed precision" if operating else "Best supported validation precision", f"{point['precision']:.1%}", border=True)
        b.metric("Articles accepted", f"{point['accepted']} / {report['validation_rows']}", border=True)
        c.metric("Accepted coverage", f"{point['coverage']:.1%}", border=True)
        st.caption(f"{point['correct']} accepted labels are correct. Wilson 95% interval: "
                   f"{point['interval'][0]:.1%}–{point['interval'][1]:.1%}. This threshold was explored on the same "
                   "validation articles; the interval does not account for threshold selection.")
        diagnostic = result.get("diagnostic_operating_point")
        if diagnostic and diagnostic["precision"] is not None:
            st.warning(f"The same validation-selected threshold gave {diagnostic['precision']:.1%} precision on "
                       f"{diagnostic['accepted']} accepted historical diagnostic articles "
                       f"({diagnostic['coverage']:.1%} coverage). This does not establish reliability on newly published articles.")
        with st.expander("Which labels are accepted?"):
            st.dataframe(pd.DataFrame([
                {"Predicted label": label, "Articles accepted": values["accepted"],
                 "Observed precision": f"{values['precision']:.1%}" if values["precision"] is not None else "—"}
                for label, values in point["per_class"].items()]), hide_index=True, width="stretch")
    st.caption("Answer scores are uncalibrated. A score near 1 is not proof of a correct label. "
               "Rejected articles must remain visible as unresolved; accepting a subset can change the apparent sentiment trend.")
    agreement = report.get("agreement_check", {}).get("validation")
    if agreement and agreement["precision"] is not None:
        st.caption(f"Agreement check: accepting only labels where selected SetFit and the selected LLM agree gives "
                   f"{agreement['precision']:.1%} observed precision at {agreement['coverage']:.1%} coverage "
                   f"({agreement['accepted']} articles). Agreement can still include shared mistakes.")
    with st.expander("Historical diagnostic results"):
        diagnostics = [{"Model": names.get(name, name), "Diagnostic accuracy": f"{values['historical_diagnostic']['accuracy']:.1%}",
                        "Macro-F1": round(values["historical_diagnostic"]["macro_f1"], 3)}
                       for name, values in report["candidates"].items() if "historical_diagnostic" in values]
        st.caption(f"These {report['diagnostic_rows']} articles were examined in the earlier model comparison. "
                   "These are historical diagnostics, not independent evidence for the new model or thresholds.")
        st.dataframe(pd.DataFrame(diagnostics), hide_index=True, width="stretch")
