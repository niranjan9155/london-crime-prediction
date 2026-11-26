"""Builds the model table: one row per LSOA and month t, with features from month t
and earlier, and a label taken from month t+1."""

import numpy as np
import pandas as pd

from crimepred import config, spatial

# Features that only use the LSOA's own history and the calendar.
OWN_FEATURES = [
    "crimes", "lag1", "lag2", "mean_3m", "change_1m", "rank_pct", "high_now",
    "high_count_3m", "share_theft", "share_violence", "month_sin", "month_cos",
]
# Features that describe the surrounding LSOAs.
NEIGHBOUR_FEATURES = ["nb_crimes", "nb_mean_3m", "nb_high_now", "n_neighbours"]

ALL_FEATURES = OWN_FEATURES + NEIGHBOUR_FEATURES


def next_month(month):
    """'202612' -> '202701'."""
    year, mon = int(month[:4]), int(month[4:])
    return f"{year + mon // 12}{mon % 12 + 1:02d}"


def add_high_flag(panel, quantile=config.HIGH_QUANTILE):
    """Marks LSOA-months in the top share of London for that month.

    The cut-off is worked out separately for every month, so the label always
    means "among the highest in London that month".
    """
    cutoff = panel.groupby("month")["crimes"].transform(lambda s: s.quantile(quantile))
    out = panel.copy()
    out["cutoff"] = cutoff
    out["high_now"] = (out["crimes"] >= cutoff).astype(int)
    out["rank_pct"] = out.groupby("month")["crimes"].rank(pct=True)
    return out


def add_own_history(panel):
    """Lags, rolling means and change, computed inside each LSOA."""
    out = panel.sort_values(["lsoa", "month"]).copy()
    g = out.groupby("lsoa")
    out["lag1"] = g["crimes"].shift(1)
    out["lag2"] = g["crimes"].shift(2)
    out["mean_3m"] = (out["crimes"] + out["lag1"] + out["lag2"]) / 3
    out["change_1m"] = out["crimes"] - out["lag1"]
    out["high_count_3m"] = (
        out["high_now"] + g["high_now"].shift(1) + g["high_now"].shift(2)
    )
    return out


def add_target(panel):
    """Label = 1 when the LSOA is high in the following month."""
    out = panel.sort_values(["lsoa", "month"]).copy()
    g = out.groupby("lsoa")
    out["target_month"] = out["month"].map(next_month)
    out["crimes_next"] = g["crimes"].shift(-1)
    out["target"] = g["high_now"].shift(-1)
    return out


def add_calendar(panel):
    """Month of the target month as a point on a circle, so December sits next to January."""
    out = panel.copy()
    mon = out["target_month"].str[4:].astype(int)
    out["month_sin"] = np.sin(2 * np.pi * mon / 12)
    out["month_cos"] = np.cos(2 * np.pi * mon / 12)
    return out


def add_neighbours(panel, pairs):
    """Average recent crime and high-month share of the surrounding LSOAs."""
    nb = spatial.neighbour_mean(panel, pairs, ["crimes", "mean_3m", "high_now"])
    out = panel.merge(nb, on=["lsoa", "month"], how="left")
    counts = spatial.neighbour_summary(pairs, panel["lsoa"].unique())
    return out.merge(counts, on="lsoa", how="left")


def build_model_table(totals, shares, pairs):
    """Joins everything and keeps only rows that have full history and a known label."""
    panel = totals.merge(shares, on=["lsoa", "month"], how="left")
    panel = add_high_flag(panel)
    panel = add_own_history(panel)
    panel = add_target(panel)
    panel = add_calendar(panel)
    panel = add_neighbours(panel, pairs)
    table = panel.dropna(subset=["lag2", "target"]).copy()
    table["target"] = table["target"].astype(int)
    table["crimes_next"] = table["crimes_next"].astype(int)
    if table[ALL_FEATURES].isna().any().any():
        bad = table[ALL_FEATURES].columns[table[ALL_FEATURES].isna().any()].tolist()
        raise ValueError(f"Missing values left in features: {bad}")
    return table.sort_values(["month", "lsoa"]).reset_index(drop=True)


def split_by_time(table):
    """Splits rows by month t, so the test months always come after training."""
    def between(lo, hi):
        return table[(table["month"] >= lo) & (table["month"] <= hi)]
    return between(*config.TRAIN_MONTHS), between(*config.VALID_MONTHS), between(*config.TEST_MONTHS)
