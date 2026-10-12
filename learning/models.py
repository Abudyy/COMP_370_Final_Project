from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_NAMES = ["majority", "tfidf_word_lr", "tfidf_char_svm", "embedding_lr", "setfit"]


@lru_cache(maxsize=2)
def frozen_encoder(model_name):
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    return SentenceTransformer(model_name, device="cpu", revision=BASE_REVISION if model_name == BASE_MODEL else None)


class EmbeddingClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, model_name=BASE_MODEL, seed=42):
        self.model_name = model_name
        self.seed = seed

    def features(self, texts):
        return frozen_encoder(self.model_name).encode(list(texts), batch_size=32,
                                                      normalize_embeddings=True, show_progress_bar=False)

    def fit(self, X, y):
        self.head_ = LogisticRegression(C=1, max_iter=2000, class_weight="balanced", random_state=self.seed)
        self.head_.fit(self.features(X), y)
        self.classes_ = self.head_.classes_
        return self

    def predict(self, X):
        return self.head_.predict(self.features(X))

    def predict_proba(self, X):
        return self.head_.predict_proba(self.features(X))


class SetFitClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, model_name=BASE_MODEL, seed=42, max_steps=30):
        self.model_name = model_name
        self.seed = seed
        self.max_steps = max_steps

    def fit(self, X, y):
        from datasets import Dataset
        from setfit import SetFitModel, Trainer, TrainingArguments
        from transformers import set_seed
        import torch
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        set_seed(self.seed)
        self.classes_ = np.unique(y)
        lookup = {label: i for i, label in enumerate(self.classes_)}
        dataset = Dataset.from_dict({"text": list(X), "label": [lookup[label] for label in y]})
        # Reload the pretrained body for each fold; never reuse a body fine-tuned on other folds.
        self.model_ = SetFitModel.from_pretrained(self.model_name, device="cpu", labels=self.classes_.tolist(),
                                                head_params={"class_weight": "balanced", "max_iter": 2000},
                                                revision=BASE_REVISION if self.model_name == BASE_MODEL else None)
        args = TrainingArguments(batch_size=16, num_epochs=1, max_steps=self.max_steps,
                                 seed=self.seed, report_to="none", logging_strategy="no", save_strategy="no",
                                 use_amp=False, show_progress_bar=False)
        Trainer(model=self.model_, args=args, train_dataset=dataset).train()
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(list(X)))

    def predict_proba(self, X):
        probabilities = self.model_.predict_proba(list(X))
        return probabilities.cpu().numpy() if hasattr(probabilities, "cpu") else np.asarray(probabilities)


def build_model(name, seed=42, base_model=BASE_MODEL, setfit_steps=30):
    if name == "majority":
        # DummyClassifier needs a 2D input; vectorization remains confined to the training fold.
        return make_pipeline(TfidfVectorizer(min_df=1), DummyClassifier(strategy="most_frequent"))
    if name == "tfidf_word_lr":
        return make_pipeline(TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2),
                             LogisticRegression(C=1, max_iter=2000, class_weight="balanced", random_state=seed))
    if name == "tfidf_char_svm":
        return make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
                             LinearSVC(C=1, class_weight="balanced", random_state=seed))
    if name == "embedding_lr":
        return EmbeddingClassifier(base_model, seed)
    if name == "setfit":
        return SetFitClassifier(base_model, seed, setfit_steps)
    raise ValueError(f"Unknown model: {name}")


def prediction_scores(model, texts):
    """Return a model score, never a claim about calibrated correctness probability."""
    if hasattr(model, "predict_proba"):
        return np.max(model.predict_proba(texts), axis=1), "uncalibrated_probability"
    margins = model.decision_function(texts)
    return np.max(margins, axis=1), "decision_margin"
