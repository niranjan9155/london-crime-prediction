"""Project settings: file paths, data sources and the time split."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"

# Metropolitan Police Service, recorded crime by LSOA (London Datastore, Open Government Licence).
# This link points to the version published on 06/11/2026 (Nov 2024 to Oct 2026).
CRIME_URL = (
    "https://data.london.gov.uk/download/exy3m/n90/"
    "MPS%20LSOA%20Level%20Crime%20(most%20recent%2024%20months).csv"
)
CRIME_FILE = DATA_RAW / "MPS LSOA Level Crime (most recent 24 months).csv"

# ONS Lower layer Super Output Areas (December 2021), full resolution boundaries (Open Government Licence).
BOUNDARY_URL = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "LSOA_2021_EW_BFE_V10_RUC/FeatureServer/3/query"
)
BOUNDARY_FILE = DATA_RAW / "lsoa_2021_london.geojson"

# British National Grid, so distances and areas are in metres.
CRS = "EPSG:27700"

# An LSOA-month is "high" when its total is in the top 10% of London LSOAs for that month.
HIGH_QUANTILE = 0.90

# Months are written as YYYYMM. Each row uses month t to predict month t+1.
TRAIN_MONTHS = ("202501", "202603")
VALID_MONTHS = ("202604", "202606")
TEST_MONTHS = ("202607", "202609")

# Share of LSOAs flagged as high risk each month when turning scores into decisions.
FLAG_SHARE = 0.10

RANDOM_STATE = 42
