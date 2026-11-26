"""Dashboard for viewing the project results.

It only reads the files in outputs/, so the pipeline does not need to be run first.

Run with:  streamlit run dashboard.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MAP_FILE = OUT / "dashboard" / "lsoa_map.geojson"
BEST = "XGBoost (with neighbours)"
RULE = "Previous-month rule"
OUTCOME_COLOURS = {
    "Correctly flagged": "#9B003C",
    "False alarm": "#3B6EA8",
    "Missed": "#E39B2D",
    "Correctly not flagged": "#E6E6E6",
}

st.set_page_config(page_title="London crime prediction", layout="wide")


# ------------------------------------------------------------------ loading
@st.cache_data
def load_tables():
    preds = pd.read_csv(OUT / "tables" / "test_predictions.csv",
                        dtype={"month": str, "target_month": str, "lsoa": str})
    preds["borough_name"] = preds["lsoa_name"].str.rsplit(" ", n=1).str[0]
    high, flagged = preds["target"] == 1, preds["flag"] == 1
    preds["outcome"] = "Correctly not flagged"
    preds.loc[high & flagged, "outcome"] = "Correctly flagged"
    preds.loc[~high & flagged, "outcome"] = "False alarm"
    preds.loc[high & ~flagged, "outcome"] = "Missed"
    metrics = pd.read_csv(OUT / "tables" / "test_metrics.csv")
    by_month = pd.read_csv(OUT / "tables" / "test_metrics_by_month.csv", dtype={"month_t": str})
    boot = pd.read_csv(OUT / "tables" / "bootstrap_comparisons.csv")
    shap_table = pd.read_csv(OUT / "tables" / "shap_importance.csv")
    quality = pd.read_csv(OUT / "tables" / "data_quality.csv")
    split = pd.read_csv(OUT / "tables" / "split_summary.csv")
    return preds, metrics, by_month, boot, shap_table, quality, split


@st.cache_data
def load_map():
    if not MAP_FILE.exists():
        # The map file is made by the pipeline; build it here if it is missing.
        from crimepred import dashboard_data
        dashboard_data.build_map_file(MAP_FILE)
    return json.loads(MAP_FILE.read_text())


@st.cache_data
def area_centres(_geo):
    """Rough centre point of each LSOA, used to zoom the map to the chosen boroughs."""
    rows = []
    for f in _geo["features"]:
        g = f["geometry"]
        ring = g["coordinates"][0] if g["type"] == "Polygon" else g["coordinates"][0][0]
        lons, lats = zip(*ring)
        rows.append((f["properties"]["lsoa"], sum(lats) / len(lats), sum(lons) / len(lons)))
    return pd.DataFrame(rows, columns=["lsoa", "lat", "lon"]).set_index("lsoa")


def month_label(yyyymm):
    return pd.to_datetime(yyyymm, format="%Y%m").strftime("%B %Y")


required = [OUT / "tables" / "test_predictions.csv", OUT / "tables" / "test_metrics.csv"]
if not all(p.exists() for p in required):
    st.error("Results are missing. Run `python -m crimepred.pipeline` first.")
    st.stop()

preds, metrics, by_month, boot, shap_table, quality, split = load_tables()
geo = load_map()

# ------------------------------------------------------------------ sidebar
st.sidebar.header("Options")
target_months = sorted(preds["target_month"].unique())
chosen_target = st.sidebar.selectbox(
    "Month being predicted", target_months, index=len(target_months) - 1, format_func=month_label
)
layer = st.sidebar.radio("Map shows", ["Predicted risk", "Flags vs actual outcome"])
boroughs = sorted(preds["borough_name"].unique())
chosen_boroughs = st.sidebar.multiselect("Boroughs (empty = all London)", boroughs)
top_n = st.sidebar.slider("Rows in the ranked list", 10, 100, 25, step=5)
st.sidebar.caption(
    "Model: XGBoost with neighbour features, trained on Jan 2025 to Jun 2026. "
    "Each month the top 10% of LSOAs by predicted risk are flagged."
)

month_df = preds[preds["target_month"] == chosen_target].copy()
month_t = month_df["month"].iloc[0]
view = month_df[month_df["borough_name"].isin(chosen_boroughs)] if chosen_boroughs else month_df

# ------------------------------------------------------------------ header
st.title("Predicting next-month high-crime LSOAs in London")
st.caption(
    "An LSOA is **high** when its recorded crime is in the top 10% of London LSOAs that month. "
    "Predictions are decision support for analysts, not automatic policing decisions. "
    "Data: Metropolitan Police Service (London Datastore) and ONS 2021 LSOA boundaries, Open Government Licence."
)

# ------------------------------------------------------------------ headline numbers
row = by_month[(by_month["model"] == BEST) & (by_month["month_t"] == month_t)].iloc[0]
rule = by_month[(by_month["model"] == RULE) & (by_month["month_t"] == month_t)].iloc[0]
st.subheader(f"How well did the model do for {month_label(chosen_target)}?")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Precision", f"{row.precision:.3f}", f"{row.precision - rule.precision:+.3f} vs rule")
c2.metric("Recall", f"{row.recall:.3f}", f"{row.recall - rule.recall:+.3f} vs rule")
c3.metric("F1", f"{row.f1:.3f}", f"{row.f1 - rule.f1:+.3f} vs rule")
c4.metric("PR-AUC", f"{row.pr_auc:.3f}", f"{row.pr_auc - rule.pr_auc:+.3f} vs rule")
c5.metric("High LSOAs found", f"{int(row.tp)} of {int(row.tp + row.fn)}", f"{int(row.tp - rule.tp):+d} vs rule")
st.caption("Scores are for all of London. 'vs rule' compares with simply repeating this month's ranking.")

# ------------------------------------------------------------------ map
st.subheader(f"Map for {month_label(chosen_target)}")
hover = {"lsoa_name": True, "crimes": True, "crimes_next": True, "probability": ":.2f",
         "outcome": True, "lsoa": False}
if chosen_boroughs:
    pts = area_centres(geo).loc[view["lsoa"]]
    centre = {"lat": float(pts["lat"].mean()), "lon": float(pts["lon"].mean())}
    zoom = 11.3 if len(chosen_boroughs) == 1 else 10.3
else:
    centre, zoom = {"lat": 51.49, "lon": -0.11}, 9.4
common = dict(geojson=geo, locations="lsoa", featureidkey="properties.lsoa",
              map_style="carto-positron", center=centre, zoom=zoom, opacity=0.75, hover_data=hover,
              labels={"crimes": f"Crimes {month_label(month_t)}",
                      "crimes_next": f"Crimes {month_label(chosen_target)}",
                      "probability": "Predicted risk", "lsoa_name": "LSOA", "outcome": "Outcome"})
if layer == "Predicted risk":
    fig = px.choropleth_map(view, color="probability", color_continuous_scale="Reds",
                            range_color=(0, 1), **common)
else:
    fig = px.choropleth_map(view, color="outcome", color_discrete_map=OUTCOME_COLOURS,
                            category_orders={"outcome": list(OUTCOME_COLOURS)}, **common)
fig.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0),
                  legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
fig.update_traces(marker_line_width=0.2)
st.plotly_chart(fig, width="stretch", config={"scrollZoom": False})
counts = view["outcome"].value_counts()
st.caption(
    " · ".join(f"{k}: {counts.get(k, 0):,}" for k in list(OUTCOME_COLOURS)[:3])
    + ". Drag to move the map; use the borough filter to zoom in. The blank area in the centre is the City of London (001A to 001C), which is not in Met Police data."
)

# ------------------------------------------------------------------ ranked list
st.subheader(f"Highest predicted risk for {month_label(chosen_target)}")
ranked = view.sort_values("probability", ascending=False).head(top_n)
table = ranked[["lsoa_name", "borough_name", "probability", "crimes", "crimes_next", "outcome"]].rename(columns={
    "lsoa_name": "LSOA",
    "borough_name": "Borough",
    "probability": "Predicted risk",
    "crimes": f"Crimes in {month_label(month_t)}",
    "crimes_next": f"Actual crimes in {month_label(chosen_target)}",
    "outcome": "Outcome",
})
table.index = range(1, len(table) + 1)
st.dataframe(table, width="stretch", height=min(38 + 35 * len(table), 560),
             column_config={"Predicted risk": st.column_config.NumberColumn(format="%.3f")})

# ------------------------------------------------------------------ detail tabs
tab1, tab2, tab3, tab4 = st.tabs(["Model comparison", "What drives predictions", "Data", "Method"])

with tab1:
    st.markdown("**All test months (July to September 2026), every model flags its top 10% each month**")
    show = metrics[["model", "precision", "recall", "f1", "pr_auc", "roc_auc", "tp", "fp", "fn"]].copy()
    show.columns = ["Model", "Precision", "Recall", "F1", "PR-AUC", "ROC-AUC", "Correct flags", "False alarms", "Missed"]
    st.dataframe(show.style.format({c: "{:.3f}" for c in ["Precision", "Recall", "F1", "PR-AUC", "ROC-AUC"]}),
                 hide_index=True, width="stretch")
    a, b = st.columns(2)
    with a:
        st.markdown("**Bootstrap differences (1,000 resamples of LSOAs, 95% interval)**")
        bt = boot.copy()
        bt["95% interval"] = bt.apply(lambda r: f"{r.ci_low:+.4f} to {r.ci_high:+.4f}", axis=1)
        bt = bt.rename(columns={"comparison": "Comparison", "metric": "Metric", "mean_diff": "Difference"})
        st.dataframe(bt[["Comparison", "Metric", "Difference", "95% interval"]], hide_index=True,
                     width="stretch")
        st.markdown(
            "- The models clearly beat the previous-month rule.\n"
            "- Neighbour features add only a very small gain.\n"
            "- Logistic regression performs at the same level as XGBoost."
        )
    with b:
        st.image(str(OUT / "figures" / "pr_curves_test.png"), width="stretch")

with tab2:
    a, b = st.columns(2)
    with a:
        bar = px.bar(shap_table.sort_values("share"), x="share", y="label", orientation="h",
                     labels={"share": "Share of total |SHAP|", "label": ""},
                     color_discrete_sequence=["#9B003C"])
        bar.update_layout(height=520, margin=dict(l=0, r=0, t=30, b=0), title="Feature importance (SHAP)")
        bar.update_xaxes(tickformat=".0%")
        st.plotly_chart(bar, width="stretch")
    with b:
        st.image(str(OUT / "figures" / "shap_beeswarm.png"), width="stretch")

with tab3:
    a, b = st.columns([2, 1])
    with a:
        st.image(str(OUT / "figures" / "london_monthly_trend.png"), width="stretch")
    with b:
        st.markdown("**Crime file checks**")
        st.dataframe(quality.astype(str), hide_index=True, width="stretch")
    st.markdown("**Time split (rows are LSOA-months)**")
    sp = split.copy()
    sp["high_share"] = sp["high_share"].map("{:.1%}".format)
    st.dataframe(sp, hide_index=True, width="stretch")

with tab4:
    st.markdown(
        """
- **Unit:** one row per LSOA and month *t*; features use month *t* and earlier, the label comes from month *t+1*.
- **Label:** high when the LSOA's total recorded crime is in the top 10% of London LSOAs for that month.
- **Features:** recent crime (last three months, change, London rank), crime mix (theft and violence shares),
  calendar (target month), and neighbours (average crime of touching LSOAs, share of neighbours that are high).
- **Split by time:** train Jan 2025 to Mar 2026, validate Apr to Jun 2026, test Jul to Sep 2026.
- **Models:** previous-month rule, logistic regression, XGBoost with and without neighbour features.
- **Evaluation:** precision, recall, F1, PR-AUC (main metric), ROC-AUC; bootstrap over LSOAs for differences.
- **Limitations:** recorded crime is not all crime; only 24 months of data; one test period.
"""
    )
