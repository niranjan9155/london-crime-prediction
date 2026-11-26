"""Baselines, logistic regression and XGBoost, with tuning on the validation months."""

import itertools

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from crimepred import config

# Count-like columns are skewed, so the logistic model sees log(1 + x) instead.
COUNT_COLS = ["crimes", "lag1", "lag2", "mean_3m", "nb_crimes", "nb_mean_3m"]


# ---------------------------------------------------------------- baseline
def previous_month_scores(df):
    """Baseline: this month's London rank is the guess for next month."""
    return df["rank_pct"].to_numpy()


# ---------------------------------------------------------------- logistic regression
def _signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def make_logistic(features, c_value):
    logged = [f for f in features if f in COUNT_COLS or f == "change_1m"]
    rest = [f for f in features if f not in logged]
    prep = ColumnTransformer([
        ("log", Pipeline([("log", FunctionTransformer(_signed_log)), ("scale", StandardScaler())]), logged),
        ("scale", StandardScaler(), rest),
    ])
    model = LogisticRegression(C=c_value, class_weight="balanced", max_iter=5000)
    return Pipeline([("prep", prep), ("model", model)])


def tune_logistic(train, valid, features, c_values=(0.01, 0.1, 1.0, 10.0)):
    """Picks the regularisation strength with the best validation PR-AUC."""
    rows = []
    for c in c_values:
        model = make_logistic(features, c).fit(train[features], train["target"])
        score = average_precision_score(valid["target"], model.predict_proba(valid[features])[:, 1])
        rows.append({"C": c, "valid_pr_auc": score})
    results = pd.DataFrame(rows)
    best_c = results.loc[results["valid_pr_auc"].idxmax(), "C"]
    return float(best_c), results


# ---------------------------------------------------------------- XGBoost
XGB_GRID = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "min_child_weight": [1, 5, 10, 20],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [1.0, 5.0, 10.0],
}


def sample_grid(n_configs, seed=config.RANDOM_STATE):
    """Random subset of the grid. The same subset is used for every feature set."""
    keys = list(XGB_GRID)
    combos = list(itertools.product(*(XGB_GRID[k] for k in keys)))
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(combos), size=min(n_configs, len(combos)), replace=False)
    return [dict(zip(keys, combos[i])) for i in sorted(picked)]


def make_xgb(params, n_estimators, pos_weight, early_stopping=None):
    return XGBClassifier(
        n_estimators=n_estimators,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        scale_pos_weight=pos_weight,
        early_stopping_rounds=early_stopping,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        **params,
    )


def positive_weight(y):
    """Ratio of negative to positive rows, so the rare class is not ignored."""
    y = np.asarray(y)
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def tune_xgb(train, valid, features, n_configs=30, max_rounds=2000, patience=50):
    """Tries each setting on the training months and scores it on the validation months."""
    pos_weight = positive_weight(train["target"])
    rows = []
    for i, params in enumerate(sample_grid(n_configs)):
        model = make_xgb(params, max_rounds, pos_weight, early_stopping=patience)
        model.fit(train[features], train["target"],
                  eval_set=[(valid[features], valid["target"])], verbose=False)
        score = average_precision_score(valid["target"], model.predict_proba(valid[features])[:, 1])
        rows.append({"config": i, **params, "best_rounds": model.best_iteration + 1, "valid_pr_auc": score})
    results = pd.DataFrame(rows).sort_values("valid_pr_auc", ascending=False).reset_index(drop=True)
    best = results.iloc[0]
    best_params = {k: best[k] for k in XGB_GRID}
    best_params["max_depth"] = int(best_params["max_depth"])
    return best_params, int(best["best_rounds"]), results


def fit_final_xgb(data, features, params, rounds):
    """Refits on training + validation months with the chosen settings."""
    model = make_xgb(params, rounds, positive_weight(data["target"]))
    return model.fit(data[features], data["target"], verbose=False)
