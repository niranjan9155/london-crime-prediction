import numpy as np
import pandas as pd
import pytest

from crimepred import evaluate, models


def make_test_frame(n_per_month=50, months=("202607", "202608")):
    rng = np.random.default_rng(0)
    rows = []
    for m in months:
        for i in range(n_per_month):
            rows.append({"lsoa": f"L{i:02d}", "month": m, "target": int(i < 5),
                         "rank_pct": rng.random()})
    return pd.DataFrame(rows)


def test_flag_top_share_flags_exactly_ten_percent_per_month():
    df = make_test_frame()
    flags = evaluate.flag_top_share(df, np.arange(len(df), dtype=float))
    per_month = pd.Series(flags).groupby(df["month"].to_numpy()).sum()
    assert (per_month == 5).all()


def test_flag_top_share_rejects_bad_share():
    df = make_test_frame()
    with pytest.raises(ValueError):
        evaluate.flag_top_share(df, np.zeros(len(df)), share=1.5)


def test_perfect_scores_give_perfect_metrics():
    df = make_test_frame()
    scores = df["target"].to_numpy().astype(float)
    flags = evaluate.flag_top_share(df, scores)
    m = evaluate.score_metrics(df["target"], scores, flags)
    assert m["precision"] == 1 and m["recall"] == 1 and m["f1"] == 1
    assert m["pr_auc"] == pytest.approx(1)
    assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == len(df)


def test_bootstrap_difference_of_identical_models_is_zero():
    df = make_test_frame()
    scores = np.linspace(0, 1, len(df))
    result = evaluate.bootstrap_difference(df, scores, scores, n_boot=50)
    assert result["pr_auc"]["mean_diff"] == pytest.approx(0)
    assert result["f1"]["ci_low"] == pytest.approx(0)


def test_sample_grid_is_repeatable():
    assert models.sample_grid(10) == models.sample_grid(10)
    assert len(models.sample_grid(10)) == 10


def test_positive_weight():
    assert models.positive_weight([0, 0, 0, 1]) == 3


def test_previous_month_scores_use_rank():
    df = make_test_frame()
    assert np.array_equal(models.previous_month_scores(df), df["rank_pct"].to_numpy())
