"""Metrics for the test months.

Scores are turned into decisions the same way for every model: in each month the
top 10% of LSOAs by score are flagged. This matches how an analyst would use a
ranked list, and keeps the comparison fair between models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from crimepred import config


def flag_top_share(df, scores, share=config.FLAG_SHARE):
    """Flags the highest-scoring share of LSOAs separately in every month."""
    if not 0 < share < 1:
        raise ValueError("share must be between 0 and 1")
    tmp = pd.DataFrame({"month": df["month"].to_numpy(), "score": scores})
    rank = tmp.groupby("month")["score"].rank(method="first", ascending=False)
    size = tmp.groupby("month")["score"].transform("size")
    limit = np.ceil(size * share)
    return (rank <= limit).astype(int).to_numpy()


def score_metrics(y, scores, flags):
    y = np.asarray(y)
    tn, fp, fn, tp = confusion_matrix(y, flags, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y, flags, zero_division=0),
        "recall": recall_score(y, flags, zero_division=0),
        "f1": f1_score(y, flags, zero_division=0),
        "pr_auc": average_precision_score(y, scores),
        "roc_auc": roc_auc_score(y, scores),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def evaluate_models(test, score_map):
    """One row of metrics per model on the full test period."""
    rows = []
    for name, scores in score_map.items():
        flags = flag_top_share(test, scores)
        rows.append({"model": name, **score_metrics(test["target"], scores, flags)})
    return pd.DataFrame(rows)


def evaluate_by_month(test, score_map):
    """Same metrics, split by test month, to check results are stable."""
    rows = []
    for name, scores in score_map.items():
        flags = flag_top_share(test, scores)
        for month in sorted(test["month"].unique()):
            mask = (test["month"] == month).to_numpy()
            m = score_metrics(test["target"].to_numpy()[mask], scores[mask], flags[mask])
            rows.append({"model": name, "month_t": month, **m})
    return pd.DataFrame(rows)


def bootstrap_difference(test, scores_a, scores_b, n_boot=1000, seed=config.RANDOM_STATE):
    """Confidence interval for (model A - model B), resampling whole LSOAs.

    Resampling LSOAs rather than single rows keeps the three months of one area
    together, because they are not independent of each other.
    """
    rng = np.random.default_rng(seed)
    lsoas = test["lsoa"].to_numpy()
    unique = np.unique(lsoas)
    index_of = pd.Series(np.arange(len(test))).groupby(lsoas).apply(np.array)
    y = test["target"].to_numpy()
    flags_a, flags_b = flag_top_share(test, scores_a), flag_top_share(test, scores_b)
    diffs = {"pr_auc": [], "f1": []}
    for _ in range(n_boot):
        pick = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate(index_of.loc[pick].to_numpy())
        yb = y[idx]
        if yb.sum() == 0:
            continue
        diffs["pr_auc"].append(average_precision_score(yb, scores_a[idx]) - average_precision_score(yb, scores_b[idx]))
        diffs["f1"].append(f1_score(yb, flags_a[idx]) - f1_score(yb, flags_b[idx]))
    out = {}
    for metric, values in diffs.items():
        values = np.array(values)
        out[metric] = {
            "mean_diff": float(values.mean()),
            "ci_low": float(np.percentile(values, 2.5)),
            "ci_high": float(np.percentile(values, 97.5)),
            "share_above_zero": float((values > 0).mean()),
        }
    return out


def round_floats(df, digits=4):
    return df.apply(lambda c: c.round(digits) if c.dtype.kind == "f" else c)
