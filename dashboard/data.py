"""Load the original research data without changing its annotations."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
SENTIMENTS = ["Negative", "Neutral", "Positive"]
TOPICS = ["Election", "Policy", "Social Response", "Opinion", "Political Identity", "Political Interaction"]
OUTLETS = {
    "ny-post": "New York Post", "ny-daily-news": "NY Daily News",
    "the-wall-street-journal": "The Wall Street Journal",
    "new-york-magazine": "New York Magazine", "the-seattle-times": "The Seattle Times",
    "nbc-news": "NBC News", "los-angeles-times": "Los Angeles Times",
    "business-insider": "Business Insider", "abc-news": "ABC News",
    "ca-national-news": "CBC News", "the-new-york-times": "The New York Times",
    "usa-today": "USA Today", "fox-news": "Fox News", "the-atlantic": "The Atlantic",
    "gothamist": "Gothamist",
}
ALIASES = {"nypost": "ny-post", "nydailynews": "ny-daily-news", "cbc-news": "ca-national-news"}
BOILERPLATE_TERMS = {"summary", "snippet", "available", "summary available", "available snippet",
                     "snippet video", "video online", "online", "online content", "content", "video"}
ENTITY_TERMS = {"mamdani", "zohran", "zohran mamdani", "elect zohran"}


def canonical_id(value: str) -> str:
    return ALIASES.get(str(value).strip(), str(value).strip())


def title_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def safe_url(value: str) -> str:
    value = str(value).strip()
    parsed = urlparse(value)
    return value if parsed.scheme in {"https", "http"} and parsed.netloc else ""


def load_corpus(root: Path = ROOT) -> pd.DataFrame:
    df = pd.read_csv(root / "data/annotation/final_full_annotated.csv", keep_default_na=False)
    required = {"id", "title", "description", "coding", "polarity"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing annotation columns: {sorted(required - set(df.columns))}")
    df = df.drop(columns=["name"], errors="ignore").rename(columns={"coding": "topic", "polarity": "sentiment"})
    for col in ["id", "topic", "sentiment"]:
        df[col] = df[col].str.strip()
    if not set(df.sentiment).issubset(SENTIMENTS) or not set(df.topic).issubset(TOPICS):
        raise ValueError("The annotation file contains an unrecognized topic or sentiment label.")
    df["id"] = df.id.map(canonical_id)
    df["outlet"] = df.id.map(OUTLETS).fillna(df.id)
    df["record_id"] = np.arange(len(df))

    # Join URLs only on normalized outlet + title. Never guess a link from a similar headline.
    raw = json.loads((root / "data/reformated_articles_final.json").read_text(encoding="utf-8"))["articles"]
    urls: dict[tuple[str, str], set[str]] = {}
    for record in raw:
        key = (canonical_id(record.get("id", "")), title_key(record.get("title", "")))
        url = safe_url(record.get("url", ""))
        if url:
            urls.setdefault(key, set()).add(url)
    df["url"] = [next(iter(candidates)) if len(candidates) == 1 else ""
                 for candidates in (urls.get((row.id, title_key(row.title)), set()) for row in df.itertuples())]

    scores = json.loads((root / "data/media_bias.json").read_text(encoding="utf-8"))["sources"]
    score_lookup = {r["journal_name"]: r["simulated_bias_score"] for r in scores}
    df["simulated_bias_score"] = df.outlet.map(score_lookup)
    df["simulated_group"] = df.simulated_bias_score.map(
        lambda x: "Unmapped" if pd.isna(x) else "Left" if x <= -3 else "Right" if x >= 3 else "Center"
    )
    return df


def filter_corpus(df: pd.DataFrame, outlets: list[str], topics: list[str], sentiments: list[str],
                  query: str = "") -> pd.DataFrame:
    mask = df.outlet.isin(outlets) & df.topic.isin(topics) & df.sentiment.isin(sentiments)
    if query.strip():
        text = df.title + " " + df.description
        mask &= text.str.contains(query.strip(), case=False, regex=False, na=False)
    return df.loc[mask].copy()


def distribution(df: pd.DataFrame, group: str, values: str = "sentiment") -> pd.DataFrame:
    order = SENTIMENTS if values == "sentiment" else TOPICS
    table = pd.crosstab(df[group], df[values]).reindex(columns=order, fill_value=0)
    long = table.rename_axis(group).reset_index().melt(id_vars=group, var_name=values, value_name="count")
    totals = df.groupby(group).size().rename("total")
    long = long.join(totals, on=group)
    long["share"] = 100 * long["count"] / long["total"]
    return long


def fit_language(df: pd.DataFrame):
    # Preserve the original analysis: full-corpus vocabulary, English stop words,
    # unigram + bigram tokens, and the original New_York substitution.
    texts = (df.title.fillna("") + " " + df.description.fillna(""))
    texts = texts.str.replace("New York", "New_York", regex=False).str.replace("new york", "New_York", regex=False)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer.get_feature_names_out()


def top_terms(matrix, vocabulary, record_ids, top_n: int = 15,
              hide_entities: bool = False, hide_boilerplate: bool = False) -> pd.DataFrame:
    if not len(record_ids):
        return pd.DataFrame(columns=["term", "score"])
    scores = np.asarray(matrix[np.asarray(record_ids, dtype=int)].mean(axis=0)).ravel()
    hidden = (ENTITY_TERMS if hide_entities else set()) | (BOILERPLATE_TERMS if hide_boilerplate else set())
    ranked = [i for i in np.argsort(-scores, kind="stable") if scores[i] > 0 and vocabulary[i] not in hidden][:top_n]
    return pd.DataFrame({"term": vocabulary[ranked], "score": scores[ranked]})


def export_csv(df: pd.DataFrame) -> bytes:
    columns = ["outlet", "title", "description", "topic", "sentiment", "url"]
    exported = df[columns].copy()
    # Keep spreadsheet programs from treating arbitrary source text as a formula.
    for col in columns:
        exported[col] = exported[col].map(lambda value: "'" + value if isinstance(value, str)
                                         and re.match(r"^[\s]*[=+@-]", value) else value)
    return exported.to_csv(index=False).encode("utf-8-sig")
