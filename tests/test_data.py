from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard.data import (ROOT, SENTIMENTS, TOPICS, canonical_id, distribution,
                            export_csv, filter_corpus, fit_language, load_corpus, safe_url, top_terms)


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


def test_annotation_integrity(corpus):
    original = pd.read_csv(ROOT / "data/annotation/final_full_annotated.csv")
    assert len(corpus) == len(original) == 484
    assert corpus.topic.tolist() == original.coding.tolist()
    assert corpus.sentiment.tolist() == original.polarity.tolist()
    assert corpus.outlet.nunique() == 15
    assert corpus.record_id.is_unique
    assert "name" not in corpus.columns


def test_known_counts(corpus):
    assert corpus.sentiment.value_counts().to_dict() == {"Neutral": 191, "Negative": 166, "Positive": 127}
    assert corpus.topic.value_counts()["Election"] == 102
    assert corpus.outlet.value_counts()["New York Post"] == 182


def test_source_aliases_and_conservative_url_join(corpus):
    assert canonical_id("nypost") == "ny-post"
    assert canonical_id("cbc-news") == "ca-national-news"
    assert canonical_id("nydailynews") == "ny-daily-news"
    assert len(corpus) == 484  # Raw URL joins must not multiply annotations.
    assert corpus.url.ne("").sum() == 481
    assert corpus.url.eq("").sum() == 3
    assert corpus.loc[corpus.url.ne(""), "url"].str.startswith(("http://", "https://")).all()


def test_simulated_scores_do_not_silently_drop_unmapped_outlets(corpus):
    unmapped = corpus[corpus.simulated_group == "Unmapped"]
    assert len(unmapped) == 17
    assert set(unmapped.outlet) == {"Los Angeles Times"}
    assert unmapped.simulated_bias_score.isna().all()


def test_filters_are_literal_and_compose(corpus):
    selected = filter_corpus(corpus, ["New York Post"], ["Opinion"], ["Negative"])
    assert not selected.empty
    assert selected.outlet.eq("New York Post").all()
    assert selected.topic.eq("Opinion").all()
    assert selected.sentiment.eq("Negative").all()
    assert filter_corpus(corpus, [], TOPICS, SENTIMENTS).empty
    assert filter_corpus(corpus, corpus.outlet.unique().tolist(), TOPICS, SENTIMENTS, "[unlikely-pattern]").empty
    trump = filter_corpus(corpus, corpus.outlet.unique().tolist(), TOPICS, SENTIMENTS, "TRUMP")
    assert len(trump) > 0
    assert (trump.title + " " + trump.description).str.contains("trump", case=False, regex=False).all()


def test_within_group_denominators(corpus):
    data = distribution(corpus, "outlet")
    np.testing.assert_allclose(data.groupby("outlet")["share"].sum(), 100)
    assert data["count"].sum() == len(corpus)
    assert set(data.sentiment) == set(SENTIMENTS)
    abc_negative = data[(data.outlet == "ABC News") & (data.sentiment == "Negative")].iloc[0]
    assert abc_negative["total"] == 11
    assert abc_negative["count"] == 2
    assert abc_negative["share"] == pytest.approx(100 * 2 / 11)


def test_language_reproduces_original_saved_scores(corpus):
    matrix, vocabulary = fit_language(corpus)
    category = None
    saved = {}
    for line in (ROOT / "data/annotation/tfidf_top_words.txt").read_text().splitlines():
        if not line.strip():
            continue
        if line.startswith("- "):
            term, score = line[2:].split(": ")
            saved[category][term] = float(score)
        else:
            category = line.strip()
            saved[category] = {}
    for category, expected in saved.items():
        rows = corpus.loc[corpus.topic == category, "record_id"].to_numpy()
        computed = top_terms(matrix, vocabulary, rows, top_n=10).set_index("term").score.to_dict()
        assert set(computed) == set(expected)
        for term, score in expected.items():
            assert computed[term] == pytest.approx(score, abs=1e-12)
    hidden = top_terms(matrix, vocabulary, corpus.record_id.to_numpy(), top_n=25,
                       hide_entities=True, hide_boilerplate=True)
    assert "mamdani" not in hidden.term.tolist()
    assert "snippet" not in hidden.term.tolist()
    assert top_terms(matrix, vocabulary, []).empty


def test_csv_export_does_not_expose_annotator_or_allow_formulas(corpus):
    sample = corpus.head(1).copy()
    sample.loc[sample.index[0], "title"] = " =HYPERLINK(\"bad\")"
    sample.loc[sample.index[0], "description"] = "+SUM(1,1)"
    exported = export_csv(sample).decode("utf-8-sig")
    assert "simulated_bias_score" not in exported.splitlines()[0]
    assert "name" not in exported.splitlines()[0]
    assert "' =HYPERLINK" in exported
    assert "'+SUM" in exported


def test_non_web_source_url_is_rejected():
    assert safe_url("javascript:alert(1)") == ""
    assert safe_url("file:///etc/passwd") == ""
    assert safe_url("https://example.com/story") == "https://example.com/story"
