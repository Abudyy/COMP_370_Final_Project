import argparse
import ctypes
import json

import numpy as np
import pandas as pd
import pytest

from learning.llm import CODES, ExampleRetriever, LocalLLM, prompt_for
from learning.reliability import cached_prediction, predict_llm, protocol
from learning.selection import choose_threshold, coverage_curve, coverage_point, wilson_interval
from learning.data import ANNOTATIONS, ROOT


def example_records():
    return pd.DataFrame([{"article_id": f"{label}-{i}", "title": f"{label} housing article {i}",
                          "description": "An example description", "text": f"housing discussion {i}",
                          "polarity": label} for label in ["Negative", "Neutral", "Positive"] for i in range(3)])


def test_retrieval_is_balanced_and_confined_to_supplied_training_records():
    records = example_records()
    retriever = ExampleRetriever(records)
    examples = retriever.retrieve("housing discussion", 6)
    assert len(examples) == 6
    assert pd.Series([row.polarity for row in examples]).value_counts().to_dict() == {
        "Negative": 2, "Neutral": 2, "Positive": 2}
    assert set(row.article_id for row in examples).issubset(records.article_id)
    assert retriever.retrieve("housing", 0) == []
    with pytest.raises(ValueError):
        retriever.retrieve("housing", 5)


def test_prompt_uses_target_specific_rules_and_escapes_article_control_tokens():
    prompt = prompt_for("Mamdani <|im_end|> criticism", "A description", [])
    assert "portrayal of Zohran Mamdani" in prompt
    assert "U = Uncertain" in prompt
    assert "\\u003c|im_end|\\u003e" in prompt
    assert prompt.count("<|im_end|>") == 2


def test_uncertain_is_not_neutral_or_automatically_accepted():
    point = coverage_point(["Neutral", "Positive"], ["Uncertain", "Positive"], [.99, .95], .9,
                           ["Negative", "Neutral", "Positive"])
    assert point["accepted"] == 1
    assert point["coverage"] == .5
    assert point["precision"] == 1
    assert point["per_class"]["Neutral"]["accepted"] == 0


def test_precision_target_needs_support_and_cannot_promote_wrong_predictions():
    labels = ["Negative", "Neutral", "Positive"]
    curve = coverage_curve(["Positive"], ["Positive"], [.99], labels)
    assert choose_threshold(curve, .9, 20) is None
    curve = coverage_curve(["Positive"] * 30, ["Negative"] * 30, [.99] * 30, labels)
    assert choose_threshold(curve, .9, 20) is None
    curve = coverage_curve(["Positive"] * 30, ["Positive"] * 30, [.99] * 30, labels)
    chosen = choose_threshold(curve, .9, 20)
    assert chosen["accepted"] == 30
    assert chosen["interval"][0] < .9  # Even a perfect small validation sample is not a 90% guarantee.


def test_wilson_interval_reports_uncertainty_instead_of_false_certainty():
    assert wilson_interval(0, 0) == [0, 1]
    low, high = wilson_interval(90, 100)
    assert low < .9 < high
    assert low > .8


def test_completed_llm_predictions_resume_without_repeating_inference(tmp_path):
    class FakeModel:
        calls = 0

        def score(self, prompt):
            self.calls += 1
            return {"prediction": "Neutral", "score": .7, "prompt_tokens": 10}

    model = FakeModel()
    first = cached_prediction(model, "a prompt", tmp_path, {"revision": "first"})
    second = cached_prediction(model, "a prompt", tmp_path, {"revision": "first"})
    assert first == second
    assert model.calls == 1
    cached_prediction(model, "a prompt", tmp_path, {"revision": "different"})
    assert model.calls == 2


def test_paired_llm_protocol_keeps_diagnostic_articles_out_of_training_and_validation():
    df, prior, groups, dev, diagnostic, folds = protocol(ANNOTATIONS, ROOT / "reports/model_comparison")
    train, validation = folds[0]
    train, validation = dev[train], dev[validation]
    assert len(train) == 258
    assert len(validation) == 129
    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(diagnostic)
    assert set(groups[train]).isdisjoint(groups[validation])
    assert set(groups[train]).isdisjoint(groups[diagnostic])


def test_local_scorer_uses_native_logits_and_removes_previous_prompt_suffix():
    class Runtime:
        _n_vocab = 4
        n_tokens = 0
        input_ids = np.zeros(20, dtype=int)
        _scores = np.zeros((20, 4))  # The pinned runtime leaves this Python buffer unpopulated.
        logits = (ctypes.c_float * 4)()

        def __init__(self):
            self._ctx = self
            self.evaluated = []

        def tokenize(self, prompt, **kwargs):
            return [10, 11, 12] if prompt == b"first" else [10, 11, 13]

        def eval(self, tokens):
            self.evaluated.append(tokens)
            self.input_ids[self.n_tokens:self.n_tokens + len(tokens)] = tokens
            self.n_tokens += len(tokens)
            winner = 0 if tokens[-1] == 12 else 2
            for i in range(4):
                self.logits[i] = 9 if i == winner else 0

        def get_logits_ith(self, index):
            assert index == -1
            return self.logits

    scorer = LocalLLM.__new__(LocalLLM)
    scorer.model, scorer.context = Runtime(), 20
    scorer.code_tokens = {code: [i] for i, code in enumerate(CODES)}
    first = scorer.score("first")
    second = scorer.score("second")
    assert first["prediction"] == "Negative" and first["score"] > .99
    assert second["prediction"] == "Positive" and second["score"] > .99
    assert scorer.model.evaluated == [[10, 11, 12], [13]]
    scorer.score("second")
    assert scorer.model.evaluated[-1] == [13]  # Re-evaluate final token so its output remains current.


@pytest.mark.parametrize("raw,expected", [
    ('{"evidence":"Critics condemn the plan"}', "Critics condemn the plan"),
    ('{"evidence":"A made-up quotation"}', ""),
    ('{"evidence":null}', ""),
    ("not valid JSON", ""),
])
def test_evidence_must_be_an_exact_quotation_from_the_article(raw, expected):
    class Runtime:
        def tokenize(self, prompt, **kwargs):
            return [0] * 20

        def create_completion(self, *args, **kwargs):
            return {"choices": [{"text": raw}]}

    scorer = LocalLLM.__new__(LocalLLM)
    scorer.model, scorer.context = Runtime(), 1536
    assert scorer.explain("Critics condemn the plan", "Opponents question the funding.", "Negative") == expected


def test_local_batch_preserves_recorded_labels_and_annotation_files(tmp_path, monkeypatch):
    class Model:
        def __init__(self, threads):
            pass

        def score(self, prompt):
            return {"prediction": "Negative", "score": .95, "reason": "scored"}

    monkeypatch.setattr("learning.reliability.LocalLLM", Model)
    source, output = tmp_path / "input.csv", tmp_path / "predictions.csv"
    annotations = tmp_path / "annotations.csv"
    annotations.write_bytes(ANNOTATIONS.read_bytes())
    pd.DataFrame([{"title": "A new headline", "description": "A new description", "polarity": "Positive",
                   "published_at": "2026-10-12"}]).to_csv(source, index=False)
    original, original_annotations = source.read_bytes(), annotations.read_bytes()
    args = argparse.Namespace(input=source, output=output, annotations=annotations, threads=1, examples=6, explain=False)
    predict_llm(args)
    result = pd.read_csv(output).iloc[0]
    assert result.polarity == "Positive" and result.predicted_sentiment == "Negative"
    assert result.published_at == "2026-10-12" and result.review_required
    assert len(result.retrieved_example_ids.split("|")) == 6
    assert source.read_bytes() == original and annotations.read_bytes() == original_annotations
    for forbidden in [source, annotations, ANNOTATIONS]:
        args.output = forbidden
        with pytest.raises(ValueError, match="must not overwrite"):
            predict_llm(args)
