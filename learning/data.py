from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "data/annotation/final_full_annotated.csv"
LABELS = {
    "sentiment": ["Negative", "Neutral", "Positive"],
    "topic": ["Election", "Opinion", "Policy", "Political Identity", "Political Interaction", "Social Response"],
}
TARGET_COLUMNS = {"sentiment": "polarity", "topic": "coding"}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def article_text(titles, descriptions):
    titles = pd.Series(titles).fillna("").astype(str)
    descriptions = pd.Series(descriptions).fillna("").astype(str)
    # Remove the collection placeholder, not genuine words such as "video" in reporting.
    placeholder = descriptions.str.match(r"^\s*No summary available in snippet\.", case=False)
    descriptions = descriptions.mask(placeholder, "")
    return (titles.str.strip() + " " + descriptions.str.strip()).str.strip().tolist()


def load_annotations(path: Path = ANNOTATIONS) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    required = {"id", "title", "description", "coding", "polarity"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")
    if df.empty or df.title.astype(str).str.strip().eq("").any():
        raise ValueError("Annotations must contain non-empty headlines.")
    for target, col in TARGET_COLUMNS.items():
        df[col] = df[col].str.strip()
        unexpected = set(df[col]) - set(LABELS[target])
        if unexpected or set(LABELS[target]) - set(df[col]):
            raise ValueError(f"{target}: expected all labels {LABELS[target]}; unexpected labels: {sorted(unexpected)}")
    df["article_id"] = [hashlib.sha256((str(r.id) + "\0" + str(r.title) + "\0" + str(r.description)).encode()).hexdigest()[:16]
                        for r in df.itertuples()]
    if not df.article_id.is_unique:
        raise ValueError("Duplicate annotation records must be resolved before evaluation.")
    df["text"] = article_text(df.title, df.description)
    return df.reset_index(drop=True)


def headline_groups(df: pd.DataFrame, threshold: float = .88) -> np.ndarray:
    """Conservative duplicate groups, not a claim to identify every shared news event."""
    titles = df.title.str.casefold().map(lambda s: re.sub(r"\s+", " ", s).strip())
    features = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(titles)
    similarity = cosine_similarity(features)
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(df)):
        for j in range(i):
            if similarity[i, j] >= threshold:
                root_i, root_j = find(i), find(j)
                parent[max(root_i, root_j)] = min(root_i, root_j)
    components = {}
    for i in range(len(df)):
        components.setdefault(find(i), []).append(df.article_id.iloc[i])
    names = {root: hashlib.sha256("|".join(sorted(ids)).encode()).hexdigest()[:16]
             for root, ids in components.items()}
    return np.array([names[find(i)] for i in range(len(df))])


def experiment_split(df, groups, seed=42):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    development, test = next(splitter.split(df.text, df.polarity, groups))
    for target, col in TARGET_COLUMNS.items():
        if set(df[col].iloc[development]) != set(LABELS[target]):
            raise ValueError(f"Development split is missing {target} classes.")
    assert not set(groups[development]) & set(groups[test])
    return development, test


def development_folds(df, indices, groups, target, seed=42):
    y = df[TARGET_COLUMNS[target]].iloc[indices].to_numpy()
    local_groups = groups[indices]
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(indices)), y, local_groups))
    for train, valid in folds:
        if set(y[train]) != set(LABELS[target]):
            raise ValueError(f"A development fold is missing {target} classes.")
        assert not set(local_groups[train]) & set(local_groups[valid])
    return folds
