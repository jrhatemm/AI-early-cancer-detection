"""Evaluation metrics (numpy only): AUC, sensitivity, specificity, best threshold."""
import numpy as np


def roc_auc(y_true, y_score):
    """AUC via the rank-sum (Mann-Whitney) formula, ties get average ranks."""
    y, s = np.asarray(y_true).astype(int), np.asarray(y_score, dtype=float)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    ranks = np.empty(len(s))
    ranks[np.argsort(s, kind="mergesort")] = np.arange(1, len(s) + 1)
    _, inv = np.unique(s, return_inverse=True)
    ranks = (np.bincount(inv, weights=ranks) / np.bincount(inv))[inv]
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def confusion(y_true, y_score, thr=0.5):
    y, pred = np.asarray(y_true).astype(int), (np.asarray(y_score) >= thr).astype(int)
    return (int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum()),
            int(((pred == 0) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum()))  # tp fp tn fn


def summary(y_true, y_score, thr=0.5):
    tp, fp, tn, fn = confusion(y_true, y_score, thr)
    div = lambda a, b: a / b if b else 0.0
    sens, spec, prec = div(tp, tp + fn), div(tn, tn + fp), div(tp, tp + fp)
    return {"auc": roc_auc(y_true, y_score), "sensitivity": sens, "specificity": spec,
            "precision": prec, "f1": div(2 * prec * sens, prec + sens),
            "accuracy": div(tp + tn, tp + fp + tn + fn), "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def youden_threshold(y_true, y_score):
    """Threshold maximising sensitivity + specificity - 1 (use it to set RISK_HIGH)."""
    best_t, best_j = 0.5, -1.0
    for t in np.unique(np.asarray(y_score)):
        m = summary(y_true, y_score, t)
        j = m["sensitivity"] + m["specificity"] - 1
        if j > best_j:
            best_t, best_j = float(t), j
    return best_t
