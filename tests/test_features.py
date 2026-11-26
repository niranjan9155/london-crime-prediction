import numpy as np
import pandas as pd
import pytest

from crimepred import config, features


def make_panel(n_lsoas=20, months=("202601", "202602", "202603", "202604", "202605")):
    """Each LSOA has a steady level so the expected values are easy to work out."""
    rows = []
    for i in range(n_lsoas):
        for k, m in enumerate(months):
            rows.append({"lsoa": f"L{i:02d}", "month": m, "crimes": i * 10 + k})
    return pd.DataFrame(rows)


def test_next_month_rolls_over_year():
    assert features.next_month("202612") == "202701"
    assert features.next_month("202601") == "202602"


def test_high_flag_is_top_ten_percent_each_month():
    panel = features.add_high_flag(make_panel())
    per_month = panel.groupby("month")["high_now"].sum()
    # 20 LSOAs: the 90th percentile cut-off keeps the top 2.
    assert (per_month == 2).all()
    assert set(panel.loc[panel["high_now"] == 1, "lsoa"]) == {"L18", "L19"}


def test_lags_stay_inside_each_lsoa():
    panel = features.add_own_history(features.add_high_flag(make_panel()))
    first = panel[panel["month"] == "202601"]
    assert first["lag1"].isna().all()  # nothing before the first month
    row = panel[(panel["lsoa"] == "L03") & (panel["month"] == "202603")].iloc[0]
    assert row["lag1"] == 31 and row["lag2"] == 30
    assert row["mean_3m"] == pytest.approx((32 + 31 + 30) / 3)


def test_target_comes_from_the_next_month():
    panel = features.add_target(features.add_high_flag(make_panel()))
    row = panel[(panel["lsoa"] == "L19") & (panel["month"] == "202602")].iloc[0]
    assert row["target_month"] == "202603"
    assert row["crimes_next"] == 192
    assert row["target"] == 1
    last = panel[panel["month"] == "202605"]
    assert last["target"].isna().all()  # no label beyond the data


def test_model_table_keeps_only_complete_rows():
    panel = make_panel()
    shares = panel[["lsoa", "month"]].assign(share_theft=0.5, share_violence=0.2)
    # Ring of neighbours: each LSOA touches the next one.
    ids = sorted(panel["lsoa"].unique())
    pairs = pd.DataFrame(
        [(a, b) for i, a in enumerate(ids) for b in (ids[i - 1], ids[(i + 1) % len(ids)])],
        columns=["lsoa", "neighbour"],
    )
    table = features.build_model_table(panel, shares, pairs)
    # Months 1-2 lack two lags, month 5 lacks a label, so only months 3-4 remain.
    assert sorted(table["month"].unique()) == ["202603", "202604"]
    assert not table[features.ALL_FEATURES].isna().any().any()
    row = table[(table["lsoa"] == "L05") & (table["month"] == "202603")].iloc[0]
    assert row["nb_crimes"] == pytest.approx((42 + 62) / 2)
    assert row["n_neighbours"] == 2


def test_month_encoding_is_on_a_circle():
    panel = pd.DataFrame({"target_month": ["202612", "202601", "202606"]})
    out = features.add_calendar(panel)
    assert np.allclose(out["month_sin"] ** 2 + out["month_cos"] ** 2, 1)


def test_time_split_has_no_overlap(monkeypatch):
    months = [f"2026{m:02d}" for m in range(1, 10)]
    table = pd.DataFrame({"month": months, "x": range(9)})
    monkeypatch.setattr(config, "TRAIN_MONTHS", ("202601", "202603"))
    monkeypatch.setattr(config, "VALID_MONTHS", ("202604", "202606"))
    monkeypatch.setattr(config, "TEST_MONTHS", ("202607", "202609"))
    train, valid, test = features.split_by_time(table)
    assert train["month"].max() < valid["month"].min()
    assert valid["month"].max() < test["month"].min()
    assert len(train) + len(valid) + len(test) == 9


def test_real_split_months_are_in_order():
    assert config.TRAIN_MONTHS[1] < config.VALID_MONTHS[0]
    assert config.VALID_MONTHS[1] < config.TEST_MONTHS[0]
