"""Loads the MPS crime file, checks it and turns it into one row per LSOA per month."""

import pandas as pd

from crimepred import config

ID_COLS = ["LSOA Code", "LSOA Name", "Borough", "Group", "SubGroup"]


def month_columns(df):
    """Returns the YYYYMM month columns in date order."""
    return sorted(c for c in df.columns if c.isdigit() and len(c) == 6)


def load_crime(path=None):
    """Reads the raw file (one row per LSOA and crime type, one column per month)."""
    df = pd.read_csv(path or config.CRIME_FILE, dtype={c: str for c in ID_COLS})
    check_crime(df)
    return df


def check_crime(df):
    """Stops early if the file does not look like the expected MPS layout."""
    missing = [c for c in ID_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Crime file is missing columns: {missing}")
    months = month_columns(df)
    # Two lags, the current month and the label month need at least 4 months.
    if len(months) < 4:
        raise ValueError("Crime file has fewer than 4 month columns")
    if df[months].isna().any().any():
        raise ValueError("Crime file has missing monthly counts")
    if (df[months] < 0).any().any():
        raise ValueError("Crime file has negative counts")
    dupes = df.duplicated(["LSOA Code", "Group", "SubGroup"]).sum()
    if dupes:
        raise ValueError(f"Crime file has {dupes} duplicated LSOA/crime-type rows")


def data_quality_summary(df):
    """Small table describing the raw file, saved for the report."""
    months = month_columns(df)
    totals = df.groupby("LSOA Code")[months].sum()
    return pd.DataFrame(
        {
            "item": [
                "rows", "columns", "LSOAs", "boroughs", "crime groups", "crime sub-groups",
                "first month", "last month", "missing values", "duplicate rows",
                "median crimes per LSOA in last month", "max crimes per LSOA in last month",
            ],
            "value": [
                len(df), df.shape[1], df["LSOA Code"].nunique(), df["Borough"].nunique(),
                df["Group"].nunique(), df["SubGroup"].nunique(), months[0], months[-1],
                int(df.isna().sum().sum()), int(df.duplicated().sum()),
                float(totals[months[-1]].median()), int(totals[months[-1]].max()),
            ],
        }
    )


def monthly_totals(df):
    """Adds up all crime types and returns a long table: LSOA, month, crimes."""
    months = month_columns(df)
    wide = df.groupby("LSOA Code")[months].sum()
    long = wide.reset_index().melt(id_vars="LSOA Code", var_name="month", value_name="crimes")
    long = long.rename(columns={"LSOA Code": "lsoa"})
    return long.sort_values(["lsoa", "month"]).reset_index(drop=True)


def group_shares(df, groups=("THEFT", "VIOLENCE AGAINST THE PERSON")):
    """Share of each LSOA-month total that comes from the given crime groups."""
    months = month_columns(df)
    total = df.groupby("LSOA Code")[months].sum()
    out = []
    for g in groups:
        part = df[df["Group"] == g].groupby("LSOA Code")[months].sum().reindex(total.index, fill_value=0)
        share = (part / total.where(total > 0)).fillna(0.0)
        name = "share_" + g.split()[0].lower()
        long = share.reset_index().melt(id_vars="LSOA Code", var_name="month", value_name=name)
        out.append(long.rename(columns={"LSOA Code": "lsoa"}).set_index(["lsoa", "month"]))
    return pd.concat(out, axis=1).reset_index()


def lsoa_names(df):
    """Lookup of LSOA code to name and borough code."""
    return df.drop_duplicates("LSOA Code")[["LSOA Code", "LSOA Name", "Borough"]].rename(
        columns={"LSOA Code": "lsoa", "LSOA Name": "lsoa_name", "Borough": "borough"}
    )
