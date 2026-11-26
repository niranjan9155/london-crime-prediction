import pandas as pd
import pytest

from crimepred import data


def test_month_columns_are_sorted(raw_crime):
    assert data.month_columns(raw_crime) == ["202601", "202602", "202603", "202604"]


def test_monthly_totals_add_up_crime_types(raw_crime):
    totals = data.monthly_totals(raw_crime).set_index(["lsoa", "month"])["crimes"]
    assert totals[("E1", "202601")] == 3
    assert totals[("E2", "202603")] == 6
    assert len(totals) == 2 * 4


def test_group_shares(raw_crime):
    shares = data.group_shares(raw_crime).set_index(["lsoa", "month"])
    assert shares.loc[("E1", "202601"), "share_theft"] == pytest.approx(1 / 3)
    assert shares.loc[("E1", "202601"), "share_violence"] == pytest.approx(2 / 3)
    # A month with no crime at all gets a share of 0, not a division error.
    assert shares.loc[("E1", "202603"), "share_theft"] == 0


def test_check_rejects_duplicates(raw_crime):
    doubled = pd.concat([raw_crime, raw_crime.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicated"):
        data.check_crime(doubled)


def test_check_rejects_negative_counts(raw_crime):
    bad = raw_crime.copy()
    bad.loc[0, "202601"] = -1
    with pytest.raises(ValueError, match="negative"):
        data.check_crime(bad)
