# Predicting next-month high-crime LSOAs in London

MSc Data Science and Artificial Intelligence research project, Sheffield Hallam University.

This repository holds the full machine-learning pipeline for the project: data preparation,
modelling, evaluation and explanation. It predicts which London Lower layer Super Output Areas
(LSOAs) will be among the **top 10% for recorded crime in the following month**, using each area's
recent crime, the calendar and crime in neighbouring areas.

The outputs are meant as **decision support** for analysts reviewing areas. They are not a tool
for automatic policing decisions.

## Research question

> How effectively can XGBoost predict which London LSOAs will have high recorded-crime levels
> in the following month, using recent crime patterns, calendar information and
> neighbouring-area features?

Supporting questions:

1. Do neighbouring-area crime features improve prediction compared with an area's own past crime?
2. Which recent, calendar and neighbouring-area features have the greatest influence on predictions?
3. How useful and understandable are the outputs to crime analysts and related professionals?
   (answered through the questionnaire; responses are not stored in this repository)

## Data

| Dataset | Source | Licence |
|---|---|---|
| MPS LSOA Level Crime (most recent 24 months), Nov 2024 to Oct 2026, published 06/11/2026 | [London Datastore](https://data.london.gov.uk/dataset/mps-recorded-crime-geographic-breakdown-exy3m) | Open Government Licence v2.0 |
| LSOA (December 2021) boundaries, full resolution | [ONS Open Geography Portal](https://geoportal.statistics.gov.uk/) | Open Government Licence v3.0 |

Contains public sector information licensed under the Open Government Licence. Contains OS data
© Crown copyright and database right.

Facts about the crime file (from `outputs/tables/data_quality.csv`):

- 106,219 rows: one row per LSOA and crime sub-group, one column per month
- 4,991 LSOAs, 33 boroughs, 10 crime groups, 29 sub-groups, 24 months
- no missing values and no duplicate rows
- very skewed: in October 2026 the median LSOA had 10 crimes and the busiest had 725

Three City of London LSOAs (001A to 001C) are policed by the City of London Police, so they are not
in the Met Police file. They appear as a blank area on the maps.

The data files are not committed. They are downloaded by the script below.

## Quick start: open the dashboard

The dashboard shows the results that are already in `outputs/`, so it works without downloading
data or re-running the models. It needs Python 3.11 or newer.

**Windows:** double-click `run_dashboard.bat`, or run these commands in PowerShell from the project folder:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m streamlit run dashboard.py
```

**macOS / Linux:** `bash run_dashboard.sh`

The first setup takes a few minutes. The dashboard then opens in the browser at
http://localhost:8501. It has:

- headline scores for the chosen month compared with the previous-month rule
- an interactive map of predicted risk, or of correct flags, false alarms and missed areas
- a ranked list of the highest-risk LSOAs, with a borough filter
- tabs with the model comparison, SHAP feature importance, data checks and the method

## Re-running the full pipeline

```bash
python -m crimepred.download    # crime file + boundaries into data/raw (about 85 MB)
python -m crimepred.pipeline    # full run, about 6 minutes on a laptop
pytest                          # 25 unit tests
```

(On Windows, put `.venv\Scripts\python` in place of `python`, as above.)

All results in `outputs/` are produced by `crimepred.pipeline`. Random seeds are fixed, so a
second run gives the same numbers.

## Method

**Unit of analysis.** One row per LSOA and month *t*. Features only use month *t* and earlier;
the label comes from month *t+1*.

**Label.** An LSOA-month is *high* when its total recorded crime is in the top 10% of all London
LSOAs for that month. The cut-off is worked out separately for each month (24 to 30 crimes; 29 in
October 2026). About 10% of rows are high.

**Features.**

| Group | Features |
|---|---|
| Own history | crimes this month, last month and two months ago, 3-month average, change since last month, London rank this month, high this month, number of high months in the last three |
| Crime mix | share of theft, share of violence against the person |
| Calendar | target month as sine and cosine |
| Neighbours | average crimes and 3-month average of touching LSOAs, share of neighbours that are high, number of neighbours |

Neighbours are LSOAs that share a border or a corner (queen contiguity), found from the 2021
boundaries with a 1 m tolerance. Each LSOA has 6 neighbours on average and none has zero.

**Time split.** Rows are split by month, never at random, so no future information reaches training.

| Part | Months *t* | Rows | High share |
|---|---|---|---|
| Train | Jan 2025 to Mar 2026 | 74,865 | 10.4% |
| Validation | Apr 2026 to Jun 2026 | 14,973 | 10.3% |
| Test | Jul 2026 to Sep 2026 | 14,973 | 10.2% |

**Models.**

1. Previous-month rule: this month's London rank is used as next month's score.
2. Logistic regression on all features (log-scaled counts, balanced class weights, C tuned on validation).
3. XGBoost on own-history features only.
4. XGBoost with neighbour features.

Both XGBoost models are tuned on the same 30 random settings, with early stopping on validation
PR-AUC and class weighting for the rare high class. The chosen settings are then refitted on
train + validation months and scored once on the test months.

**Evaluation.** Every model flags the top 10% of LSOAs by score in each test month, which is how an
analyst would use a ranked list. Reported metrics: precision, recall, F1, PR-AUC (main metric,
because the classes are imbalanced), ROC-AUC and the confusion matrix. Differences between models
are checked with a bootstrap of 1,000 resamples of whole LSOAs.

**Explanation.** SHAP values for the final XGBoost model on the test months, plus XGBoost gain importance.

## Results (test months, July to September 2026)

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| Previous-month rule | 0.758 | 0.743 | 0.750 | 0.848 | 0.971 |
| Logistic regression | 0.803 | 0.787 | 0.795 | 0.886 | 0.980 |
| XGBoost (own history) | 0.796 | 0.780 | 0.788 | 0.882 | 0.980 |
| XGBoost (with neighbours) | 0.799 | 0.782 | 0.790 | 0.884 | 0.980 |

Bootstrap differences (95% interval, `outputs/tables/bootstrap_comparisons.csv`):

| Comparison | PR-AUC difference | F1 difference |
|---|---|---|
| XGBoost with neighbours vs previous-month rule | +0.035 (+0.028 to +0.042) | +0.040 (+0.027 to +0.053) |
| XGBoost with neighbours vs own history | +0.0017 (-0.0002 to +0.0036) | +0.003 (-0.003 to +0.008) |
| XGBoost with neighbours vs logistic regression | -0.002 (-0.004 to +0.001) | -0.005 (-0.010 to +0.001) |

What this shows:

- The learned models clearly beat the previous-month rule. Across the three test months, XGBoost
  with neighbours finds 61 more high LSOAs correctly (about 20 per month) and raises 61 fewer
  false alarms.
- Adding neighbour features gives only a small gain over own history, and the interval just
  touches zero. Most of the signal is in the area's own recent crime.
- XGBoost and logistic regression perform at the same level. A simpler model is enough for this
  task with these features.
- SHAP shows the 3-month average (54% of total importance) and the London rank this month (16%)
  drive most predictions. The four neighbour features together account for about 5%.

## Outputs

| File | Content |
|---|---|
| `outputs/figures/risk_map_latest.png` | Predicted probability for October 2026 |
| `outputs/figures/outcome_map_latest.png` | Top-10% flags compared with actual high LSOAs, October 2026 |
| `outputs/figures/pr_curves_test.png` | Precision-recall curves for all models |
| `outputs/figures/confusion_xgb_full.png` | Confusion matrix, XGBoost with neighbours |
| `outputs/figures/shap_beeswarm.png`, `shap_bar.png` | SHAP explanations |
| `outputs/figures/xgb_gain_importance.png` | XGBoost gain importance |
| `outputs/figures/london_monthly_trend.png` | London total crime per month |
| `outputs/tables/test_metrics.csv`, `test_metrics_by_month.csv` | Test metrics overall and by month |
| `outputs/tables/test_predictions.csv` | Score, flag and actual outcome for every LSOA in the test months |
| `outputs/tables/tuning_*.csv` | Every setting tried during tuning |
| `outputs/summary.json` | Key numbers in one file |
| `outputs/dashboard/lsoa_map.geojson` | Simplified LSOA shapes used by the dashboard map |

## Project structure

```
src/crimepred/
  config.py      paths, sources, split months, settings
  download.py    downloads the crime file and boundaries
  data.py        checks the crime file, monthly totals, crime-group shares
  spatial.py     boundaries, neighbour pairs, neighbour averages
  features.py    features, labels and the time split
  models.py      baseline, logistic regression, XGBoost tuning and refit
  evaluate.py    top-10% flags, metrics, bootstrap comparisons
  explain.py     SHAP values and plots
  plots.py       charts and maps
  dashboard_data.py  small map file for the dashboard
  pipeline.py    runs everything in order
tests/           unit tests for data, features, neighbours, metrics and models
dashboard.py     Streamlit dashboard (reads outputs/ only)
run_dashboard.bat / run_dashboard.sh   one-step setup and launch
```

## Limitations

- Recorded crime is not all crime. Reporting and recording practices affect the counts, and
  models trained on police records can repeat existing patterns.
- Only 24 months are used, so seasonal effects are learned from roughly two cycles.
- The label is relative (top 10% each month), so it says where crime is concentrated, not whether
  it is rising overall.
- Results are for one test period (July to September 2026).

## Ethics

The project was reviewed under the Sheffield Hallam University UREC2 low-risk process. The crime and
boundary data are public and area-level, with no personal information. Questionnaire responses
from participants are kept on university storage only and are never added to this repository.
