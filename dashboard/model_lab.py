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
