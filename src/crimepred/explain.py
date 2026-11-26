"""SHAP values for the final XGBoost model."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from crimepred import config

FEATURE_LABELS = {
    "crimes": "Crimes this month",
    "lag1": "Crimes last month",
    "lag2": "Crimes two months ago",
    "mean_3m": "3-month average",
    "change_1m": "Change since last month",
    "rank_pct": "London rank this month",
    "high_now": "High this month",
    "high_count_3m": "High months (last 3)",
    "share_theft": "Theft share",
    "share_violence": "Violence share",
    "month_sin": "Target month (sin)",
    "month_cos": "Target month (cos)",
    "nb_crimes": "Neighbours: crimes",
    "nb_mean_3m": "Neighbours: 3-month avg",
    "nb_high_now": "Neighbours: share high",
    "n_neighbours": "Number of neighbours",
}


def shap_values(model, X, sample_size=4000, seed=config.RANDOM_STATE):
    """Explains a random sample of rows (the full test set gives the same picture, only slower)."""
    sample = X.sample(min(sample_size, len(X)), random_state=seed)
    explainer = shap.TreeExplainer(model)
    return explainer(sample)


def importance_table(explanation):
    """Mean absolute SHAP value per feature, largest first."""
    values = np.abs(explanation.values).mean(axis=0).astype(float)
    table = pd.DataFrame({"feature": explanation.feature_names, "mean_abs_shap": values})
    table["label"] = table["feature"].map(FEATURE_LABELS)
    table["share"] = table["mean_abs_shap"] / table["mean_abs_shap"].sum()
    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def save_plots(explanation, folder):
    """Beeswarm (direction of effect) and bar chart (size of effect)."""
    named = shap.Explanation(
        values=explanation.values,
        base_values=explanation.base_values,
        data=explanation.data,
        feature_names=[FEATURE_LABELS.get(f, f) for f in explanation.feature_names],
    )
    shap.plots.beeswarm(named, max_display=16, show=False)
    plt.title("SHAP values on test months (XGBoost with neighbour features)")
    plt.tight_layout()
    plt.savefig(folder / "shap_beeswarm.png", dpi=200)
    plt.close("all")

    shap.plots.bar(named, max_display=16, show=False)
    plt.title("Mean |SHAP| per feature")
    plt.tight_layout()
    plt.savefig(folder / "shap_bar.png", dpi=200)
    plt.close("all")
