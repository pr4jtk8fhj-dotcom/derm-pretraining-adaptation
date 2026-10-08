"""Shared evaluation code: threshold locked on validation, metrics, patient/lesion-clustered bootstrap."""
import math

import numpy as np
from sklearn.metrics import roc_auc_score


def threshold_at_sensitivity(y, s, target=0.95):
    """Largest threshold t such that sensitivity of (s >= t) on (y, s) is >= target."""
    pos = np.sort(np.asarray(s)[np.asarray(y) == 1])[::-1]
    if len(pos) == 0:
        raise ValueError("no positives")
    k = math.ceil(target * len(pos))
    return float(pos[k - 1])


def metrics_at(y, s, t):
    y, s = np.asarray(y), np.asarray(s)
    flag = s >= t
    tp = int((flag & (y == 1)).sum()); fn = int((~flag & (y == 1)).sum())
    tn = int((~flag & (y == 0)).sum()); fp = int((flag & (y == 0)).sum())
    return {
        "n": len(y), "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
        "auroc": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan"),
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "false_negatives": fn, "false_positives": fp, "tp": tp, "tn": tn,
        "flag_rate": float(flag.mean()), "threshold": float(t),
    }


def _group_index(groups):
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    return np.split(order, np.cumsum(np.bincount(inv))[:-1])


def bootstrap(y, s, t, groups, n_boot=1000, seed=0, keys=("auroc", "sensitivity", "specificity", "flag_rate", "false_negatives")):
    """Percentile 95% CI resampling whole groups (patient, or lesion where patient is missing).
    The threshold t is held fixed (locked on validation): CIs do not include threshold-selection variability."""
    y, s = np.asarray(y), np.asarray(s)
    idx = _group_index(groups)
    rng = np.random.default_rng(seed)
    vals = {k: [] for k in keys}
    skipped = 0
    for _ in range(n_boot):
        pick = np.concatenate([idx[g] for g in rng.integers(0, len(idx), len(idx))])
        yy = y[pick]
        if yy.min() == yy.max():
            skipped += 1
            continue
        m = metrics_at(yy, s[pick], t)
        for k in keys:
            vals[k].append(m[k])
    ci = {k: ([float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] if v else [float("nan")] * 2)
          for k, v in vals.items()}
    ci["n_boot"], ci["n_skipped_single_class"], ci["n_groups"] = n_boot, skipped, len(idx)
    return ci


def paired_bootstrap(y, s1, t1, s2, t2, groups, n_boot=1000, seed=0, keys=("auroc", "sensitivity", "specificity", "flag_rate")):
    """CI of (method 2 - method 1) on the same test images, same group resamples."""
    y, s1, s2 = np.asarray(y), np.asarray(s1), np.asarray(s2)
    idx = _group_index(groups)
    rng = np.random.default_rng(seed)
    diffs = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = np.concatenate([idx[g] for g in rng.integers(0, len(idx), len(idx))])
        if y[pick].min() == y[pick].max():
            continue
        m1, m2 = metrics_at(y[pick], s1[pick], t1), metrics_at(y[pick], s2[pick], t2)
        for k in keys:
            diffs[k].append(m2[k] - m1[k])
    p1, p2 = metrics_at(y, s1, t1), metrics_at(y, s2, t2)
    return {k: {"diff": p2[k] - p1[k], "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
            for k, v in diffs.items()}


def evaluate_locked(val_y, val_s, test_y, test_s, test_groups, target=0.95, n_boot=1000, seed=0):
    t = threshold_at_sensitivity(val_y, val_s, target)
    return {"target_sensitivity": target, "threshold": t,
            "val": metrics_at(val_y, val_s, t),
            "test": metrics_at(test_y, test_s, t),
            "test_ci95": bootstrap(test_y, test_s, t, test_groups, n_boot, seed) if n_boot else None}
