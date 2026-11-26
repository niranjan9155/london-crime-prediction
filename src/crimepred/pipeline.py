"""Runs the full project from raw files to results.

Run with:  python -m crimepred.pipeline
"""

import json
import time

import pandas as pd

from crimepred import config, data, evaluate, explain, features, models, plots, spatial

NAMES = {
    "rule": "Previous-month rule",
    "logit": "Logistic regression",
    "xgb_own": "XGBoost (own history)",
    "xgb_full": "XGBoost (with neighbours)",
}


def log(msg, start=[time.time()]):
    print(f"[{time.time() - start[0]:6.1f}s] {msg}", flush=True)


def save_table(df, name):
    path = config.TABLES / name
    evaluate.round_floats(df).to_csv(path, index=False)
    return path


def main(n_configs=30):
    for folder in (config.DATA_PROCESSED, config.FIGURES, config.TABLES):
        folder.mkdir(parents=True, exist_ok=True)

    # 1. Data preparation
    log("Loading crime file")
    raw = data.load_crime()
    save_table(data.data_quality_summary(raw), "data_quality.csv")
    totals = data.monthly_totals(raw)
    shares = data.group_shares(raw)

    log("Loading boundaries and finding neighbours")
    gdf = spatial.load_boundaries()
    code_check = spatial.check_codes(totals["lsoa"].unique(), gdf)
    save_table(code_check.rename("count").reset_index().rename(columns={"index": "check"}), "code_check.csv")
    if code_check["crime LSOAs without boundary"] > 0:
        raise ValueError("Some crime LSOAs have no boundary; neighbour features would be wrong")
    pairs = spatial.neighbour_pairs(gdf)
    nb_counts = spatial.neighbour_summary(pairs, gdf["lsoa"])

    log("Building model table")
    table = features.build_model_table(totals, shares, pairs)
    table.to_parquet(config.DATA_PROCESSED / "model_table.parquet", index=False)
    train, valid, test = features.split_by_time(table)
    split = pd.DataFrame([
        {"part": name, "first_month_t": part["month"].min(), "last_month_t": part["month"].max(),
         "rows": len(part), "high_share": part["target"].mean()}
        for name, part in [("train", train), ("validation", valid), ("test", test)]
    ])
    save_table(split, "split_summary.csv")
    plots.monthly_trend(totals, config.FIGURES / "london_monthly_trend.png")

    own, full = features.OWN_FEATURES, features.ALL_FEATURES
    train_valid = pd.concat([train, valid])

    # 2. Models
    log("Tuning logistic regression")
    best_c, logit_tuning = models.tune_logistic(train, valid, full)
    save_table(logit_tuning, "tuning_logistic.csv")
    logit = models.make_logistic(full, best_c).fit(train_valid[full], train_valid["target"])

    log(f"Tuning XGBoost on own-history features ({n_configs} settings)")
    own_params, own_rounds, own_tuning = models.tune_xgb(train, valid, own, n_configs=n_configs)
    save_table(own_tuning, "tuning_xgb_own.csv")
    log(f"Tuning XGBoost with neighbour features ({n_configs} settings)")
    full_params, full_rounds, full_tuning = models.tune_xgb(train, valid, full, n_configs=n_configs)
    save_table(full_tuning, "tuning_xgb_full.csv")

    log("Refitting final models on train + validation months")
    xgb_own = models.fit_final_xgb(train_valid, own, own_params, own_rounds)
    xgb_full = models.fit_final_xgb(train_valid, full, full_params, full_rounds)

    # 3. Evaluation on the untouched test months
    log("Scoring test months")
    scores = {
        NAMES["rule"]: models.previous_month_scores(test),
        NAMES["logit"]: logit.predict_proba(test[full])[:, 1],
        NAMES["xgb_own"]: xgb_own.predict_proba(test[own])[:, 1],
        NAMES["xgb_full"]: xgb_full.predict_proba(test[full])[:, 1],
    }
    overall = evaluate.evaluate_models(test, scores)
    save_table(overall, "test_metrics.csv")
    save_table(evaluate.evaluate_by_month(test, scores), "test_metrics_by_month.csv")

    log("Bootstrap comparisons (1,000 resamples of LSOAs)")
    comparisons = {
        "XGBoost with neighbours vs own history": (NAMES["xgb_full"], NAMES["xgb_own"]),
        "XGBoost with neighbours vs previous-month rule": (NAMES["xgb_full"], NAMES["rule"]),
        "XGBoost with neighbours vs logistic regression": (NAMES["xgb_full"], NAMES["logit"]),
    }
    boot_rows = []
    for label, (a, b) in comparisons.items():
        result = evaluate.bootstrap_difference(test, scores[a], scores[b])
        for metric, stats in result.items():
            boot_rows.append({"comparison": label, "metric": metric, **stats})
    boot = pd.DataFrame(boot_rows)
    save_table(boot, "bootstrap_comparisons.csv")

    # 4. Predictions file for the test months
    best_flags = evaluate.flag_top_share(test, scores[NAMES["xgb_full"]])
    names = data.lsoa_names(raw)
    preds = test[["lsoa", "month", "target_month", "crimes", "crimes_next", "target"]].copy()
    preds["probability"] = scores[NAMES["xgb_full"]]
    preds["flag"] = best_flags
    preds = names.merge(preds, on="lsoa").sort_values(["month", "probability"], ascending=[True, False])
    save_table(preds, "test_predictions.csv")

    # 5. Figures and explanation
    log("Plotting curves and maps")
    plots.pr_curves(test, scores, config.FIGURES / "pr_curves_test.png")
    best_row = overall.set_index("model").loc[NAMES["xgb_full"]]
    plots.confusion(best_row, config.FIGURES / "confusion_xgb_full.png",
                    "XGBoost with neighbours, test months")
    last_t = test["month"].max()
    last = preds[preds["month"] == last_t]
    target_label = pd.to_datetime(features.next_month(last_t), format="%Y%m").strftime("%B %Y")
    outline = spatial.london_outline(gdf)
    map_gdf = gdf.assign(geometry=gdf.geometry.simplify(5))
    plots.risk_map(map_gdf, last[["lsoa", "probability"]], outline,
                   config.FIGURES / "risk_map_latest.png",
                   f"Predicted risk of being a high-crime LSOA in {target_label}")
    plots.outcome_map(map_gdf, last[["lsoa", "target", "flag"]], outline,
                      config.FIGURES / "outcome_map_latest.png",
                      f"Top-10% flags vs actual high LSOAs, {target_label}")
    plots.feature_gain(xgb_full, full, explain.FEATURE_LABELS, config.FIGURES / "xgb_gain_importance.png")

    log("Computing SHAP values")
    expl = explain.shap_values(xgb_full, test[full])
    shap_table = explain.importance_table(expl)
    save_table(shap_table, "shap_importance.csv")
    explain.save_plots(expl, config.FIGURES)

    # 6. Short summary for the report
    summary = {
        "data": {
            "lsoas": int(totals["lsoa"].nunique()),
            "months": [totals["month"].min(), totals["month"].max()],
            "neighbour_pairs": int(len(pairs) // 2),
            "mean_neighbours": round(float(nb_counts["n_neighbours"].mean()), 2),
        },
        "split": split.to_dict(orient="records"),
        "chosen_settings": {
            "logistic_C": best_c,
            "xgb_own": {**own_params, "rounds": own_rounds},
            "xgb_full": {**full_params, "rounds": full_rounds},
        },
        "test_metrics": evaluate.round_floats(overall).to_dict(orient="records"),
        "bootstrap": evaluate.round_floats(boot).to_dict(orient="records"),
        "top_shap_features": shap_table.head(5)[["feature", "share"]].round(4).to_dict(orient="records"),
    }
    (config.OUTPUTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log("Done")
    print(evaluate.round_floats(overall, 3).to_string(index=False))
    return summary


if __name__ == "__main__":
    main()
