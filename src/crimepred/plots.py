"""Charts and maps saved to outputs/figures."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.metrics import precision_recall_curve

MAROON = "#9B003C"
MAP_NOTE = ("Blank area in the centre: City of London LSOAs 001A-001C, policed by the "
            "City of London Police and not included in Met Police data.")
COLOURS = {
    "Previous-month rule": "#888888",
    "Logistic regression": "#3B6EA8",
    "XGBoost (own history)": "#E39B2D",
    "XGBoost (with neighbours)": MAROON,
}


def monthly_trend(totals, path):
    """London total recorded crime per month."""
    series = totals.groupby("month")["crimes"].sum()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(range(len(series)), series.values, color=MAROON, linewidth=2)
    ax.set_xticks(range(0, len(series), 3))
    ax.set_xticklabels([f"{m[4:]}/{m[2:4]}" for m in series.index[::3]])
    ax.set_ylabel("Recorded crimes")
    ax.set_title("London recorded crime per month (all LSOAs)")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def pr_curves(test, score_map, path):
    """Precision-recall curve for each model on the test months."""
    fig, ax = plt.subplots(figsize=(6, 5))
    base_rate = test["target"].mean()
    for name, scores in score_map.items():
        precision, recall, _ = precision_recall_curve(test["target"], scores)
        ax.plot(recall, precision, label=name, color=COLOURS.get(name), linewidth=1.8)
    ax.axhline(base_rate, color="black", linestyle=":", linewidth=1, label=f"Random ({base_rate:.0%} high)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall on test months (Jul-Sep 2026)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def confusion(metrics_row, path, title):
    """2 x 2 confusion matrix from the flagged decisions."""
    grid = np.array([[metrics_row["tn"], metrics_row["fp"]], [metrics_row["fn"], metrics_row["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(grid, cmap="Reds")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{grid[i, j]:,}", ha="center", va="center",
                    color="white" if grid[i, j] > grid.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["Not flagged", "Flagged"])
    ax.set_yticks([0, 1], ["Not high", "High"])
    ax.set_xlabel("Model decision (top 10%)")
    ax.set_ylabel("Actual next month")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def risk_map(gdf, values, outline, path, title):
    """Choropleth of predicted probability for one month."""
    data = gdf.merge(values, on="lsoa", how="left")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    outline.boundary.plot(ax=ax, color="#444444", linewidth=0.6)
    data.plot(column="probability", ax=ax, cmap="Reds", linewidth=0, legend=True,
              legend_kwds={"label": "Predicted probability of being high next month", "shrink": 0.6},
              vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_axis_off()
    fig.text(0.01, 0.01, MAP_NOTE, fontsize=7, color="#555555")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def outcome_map(gdf, values, outline, path, title):
    """Where the flagged top 10% matched reality and where it did not."""
    data = gdf.merge(values, on="lsoa", how="left")
    labels = {0: "Correctly not flagged", 1: "Missed (high, not flagged)",
              2: "False alarm (flagged, not high)", 3: "Correctly flagged"}
    high, flagged = data["target"] == 1, data["flag"] == 1
    data["outcome"] = np.select(
        [high & flagged, ~high & flagged, high & ~flagged], [3, 2, 1], default=0
    )
    colours = ["#EEEEEE", "#E39B2D", "#3B6EA8", MAROON]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    data.plot(column="outcome", ax=ax, cmap=ListedColormap(colours), vmin=-0.5, vmax=3.5, linewidth=0)
    outline.boundary.plot(ax=ax, color="#444444", linewidth=0.6)
    counts = data["outcome"].value_counts()
    handles = [Patch(color=colours[k], label=f"{labels[k]} ({counts.get(k, 0):,})") for k in labels]
    ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=True)
    ax.set_title(title)
    ax.set_axis_off()
    fig.text(0.01, 0.01, MAP_NOTE, fontsize=7, color="#555555")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def feature_gain(model, features, labels, path):
    """XGBoost's own gain-based importance, for comparison with SHAP."""
    gain = model.get_booster().get_score(importance_type="gain")
    values = np.array([gain.get(f, 0.0) for f in features])
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh([labels.get(features[i], features[i]) for i in order], values[order] / values.sum(), color=MAROON)
    ax.set_xlabel("Share of total gain")
    ax.set_title("XGBoost gain importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
