"""Metric implementations for §7.3/§7.4 (numpy; no sklearn dependency)."""

from __future__ import annotations

import numpy as np


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.abs(pred - true).mean())


def r2(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = float(((true - pred) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def binary_metrics(logit_or_score: np.ndarray, true: np.ndarray, threshold: float = 0.0):
    """accuracy, precision, recall, F1 at `threshold`; AUROC over all scores."""
    pred = (logit_or_score > threshold).astype(np.float64)
    true = true.astype(np.float64)
    tp = float((pred * true).sum())
    fp = float((pred * (1 - true)).sum())
    fn = float(((1 - pred) * true).sum())
    tn = float(((1 - pred) * (1 - true)).sum())
    acc = (tp + tn) / max(len(true), 1)
    prec = tp / (tp + fp) if tp + fp > 0 else float("nan")
    rec = tp / (tp + fn) if tp + fn > 0 else float("nan")
    f1 = (
        2 * prec * rec / (prec + rec)
        if prec == prec and rec == rec and (prec + rec) > 0
        else float("nan")
    )
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auroc": auroc(logit_or_score, true),
        "positives": int(true.sum()),
        "n": len(true),
    }


def auroc(score: np.ndarray, true: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U), ties handled by average rank."""
    pos = true > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    sorted_scores = score[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
