"""Selective classification metrics; uncertainty is never relabeled as Neutral."""
from math import sqrt

import numpy as np


def wilson_interval(correct, total, z=1.96):
    if not total:
        return [0.0, 1.0]
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def coverage_point(y, predicted, scores, threshold, labels):
    y, predicted, scores = np.asarray(y), np.asarray(predicted), np.asarray(scores)
    accepted = (scores >= threshold) & np.isin(predicted, labels)
    total = int(accepted.sum())
    correct = int(((predicted == y) & accepted).sum())
    per_class = {}
    for label in labels:
        selected = accepted & (predicted == label)
        n = int(selected.sum())
        hits = int(((predicted == y) & selected).sum())
        per_class[label] = {"accepted": n, "precision": hits / n if n else None,
                            "interval": wilson_interval(hits, n)}
    return {"threshold": float(threshold), "accepted": total, "correct": correct,
            "coverage": total / len(y) if len(y) else 0,
            "precision": correct / total if total else None,
            "interval": wilson_interval(correct, total), "per_class": per_class}


def coverage_curve(y, predicted, scores, labels):
    scores, predicted = np.asarray(scores), np.asarray(predicted)
    known = scores[np.isfinite(scores) & np.isin(predicted, labels)]
    # Every distinct observed score defines a possible accepted subset. A coarse grid can miss
    # useful subsets, particularly when restricted answer-code probabilities saturate near 1.
    thresholds = np.unique(np.concatenate([[0.0, 1.0], known]))
    return [coverage_point(y, predicted, scores, t, labels) for t in thresholds]


def choose_threshold(curve, target_precision=.9, minimum_accepted=20):
    eligible = [p for p in curve if p["accepted"] >= minimum_accepted
                and p["precision"] is not None and p["precision"] >= target_precision]
    # This is an exploratory validation operating point, not certification of population precision.
    return max(eligible, key=lambda p: (p["coverage"], -p["threshold"])) if eligible else None
