import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from learning.cli import compare, predict, source_check, train
from learning.data import (ANNOTATIONS, LABELS, article_text, development_folds, experiment_split,
                           file_hash, headline_groups, load_annotations)
from learning.models import build_model, prediction_scores


@pytest.fixture(scope="module")
def dataset():
    df = load_annotations()
    return df, headline_groups(df)


def test_placeholders_are_removed_without_dropping_real_reporting_words():
    texts = article_text(["A headline", "A video headline"],
                         ["No summary available in snippet. (Could be video or online-only content)",
                          "A video shows residents discussing housing."])
    assert texts[0] == "A headline"
    assert "video shows" in texts[1]


def test_duplicate_headlines_never_cross_development_test_or_validation(dataset):
    df, groups = dataset
    dev, test = experiment_split(df, groups)
    assert len(dev) + len(test) == 484
    assert set(dev).isdisjoint(test)
    assert set(groups[dev]).isdisjoint(groups[test])
    assert len(set(groups)) < len(df)
    for target in LABELS:
        folds = development_folds(df, dev, groups, target)
        validation_rows = []
        for training, validation in folds:
            assert set(groups[dev[training]]).isdisjoint(groups[dev[validation]])
            assert set(dev[training]).isdisjoint(test)
            validation_rows.extend(validation)
        assert sorted(validation_rows) == list(range(len(dev)))


def test_holdout_assignment_is_reproducible(dataset):
    df, groups = dataset
    first = experiment_split(df, groups, 42)
    second = experiment_split(df, groups, 42)
    np.testing.assert_array_equal(first[1], second[1])


def test_vectorizer_cannot_see_unseen_test_words():
    model = build_model("tfidf_word_lr")
    text = ["housing costs rising", "housing costs falling", "public transit changes", "public transit proposal"]
    model.fit(text, ["Negative", "Positive", "Neutral", "Neutral"])
    model.predict(["neverobservedtoken appears here"])
    assert "neverobservedtoken" not in model.steps[0][1].vocabulary_
    scores, kind = prediction_scores(model, ["housing costs rising"])
    assert kind == "uncalibrated_probability"
    assert 0 <= scores[0] <= 1


def test_invalid_annotations_fail_early(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"title": ["A title"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="Missing columns"):
        load_annotations(bad)


def test_training_and_prediction_workflow_preserves_human_labels(tmp_path):
    before = file_hash(ANNOTATIONS)
    report = tmp_path / "report"
    compare(argparse.Namespace(annotations=ANNOTATIONS, output=report,
                              models=["majority", "tfidf_word_lr"], seed=42,
                              base_model="unused-for-offline-test", setfit_steps=1))
    summary = json.loads((report / "summary.json").read_text())
    assert summary["complete"]
    assert summary["automation_ready"] is False
    for task in summary["tasks"].values():
        assert task["selected_model"] == "tfidf_word_lr"
        assert len(task["held_out_test"]["confusion_matrix"]) in {3, 6}
    source_check(argparse.Namespace(annotations=ANNOTATIONS, report=report))
    stress = json.loads((report / "source_stress.json").read_text())
    assert stress["comparison_created_at"] == summary["created_at"]
    assert all(len(task["folds"]) == 5 for task in stress["tasks"].values())
    artifacts = tmp_path / "models"
    train(argparse.Namespace(annotations=ANNOTATIONS, report=report, output=artifacts))
    source = tmp_path / "new.csv"
    output = tmp_path / "predictions.csv"
    pd.DataFrame({"title": ["Residents discuss Mamdani's housing proposal"], "description": ["Public debate continues."],
                  "published_at": ["2026-10-12T08:00:00Z"], "polarity": ["original-value"]}).to_csv(source, index=False)
    predict(argparse.Namespace(input=source, output=output, artifacts=artifacts))
    result = pd.read_csv(output)
    assert result.polarity.iloc[0] == "original-value"
    assert result.published_at.iloc[0] == "2026-10-12T08:00:00Z"
    assert result.predicted_sentiment.iloc[0] in LABELS["sentiment"]
    assert result.predicted_topic.iloc[0] in LABELS["topic"]
    assert result.review_required.iloc[0]
    assert result.label_source.iloc[0] == "model"
    assert result.model_version.str.startswith("experiment-").all()
    assert file_hash(ANNOTATIONS) == before
    with pytest.raises(ValueError, match="must not overwrite"):
        predict(argparse.Namespace(input=source, output=source, artifacts=artifacts))
    with pytest.raises(ValueError, match="must not overwrite"):
        predict(argparse.Namespace(input=source, output=ANNOTATIONS, artifacts=artifacts))


def test_changed_dataset_cannot_reuse_an_old_model_comparison(tmp_path):
    report = tmp_path / "report"
    report.mkdir()
    (report / "summary.json").write_text(json.dumps({"complete": True, "dataset_sha256": "different"}))
    with pytest.raises(ValueError, match="complete comparison"):
        train(argparse.Namespace(annotations=ANNOTATIONS, report=report, output=tmp_path / "models"))
