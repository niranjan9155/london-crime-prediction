"""Boundaries and neighbour lists for London LSOAs."""

import geopandas as gpd
import pandas as pd

from crimepred import config

# Polygons closer than this (metres) count as touching. A small gap allowance
# is needed because shared edges in boundary files are not always exact.
TOUCH_TOLERANCE = 1.0


def load_boundaries(path=None):
    """Reads the LSOA polygons in British National Grid."""
    gdf = gpd.read_file(path or config.BOUNDARY_FILE)
    gdf = gdf.set_crs(config.CRS, allow_override=True) if gdf.crs is None else gdf.to_crs(config.CRS)
    gdf = gdf.rename(columns={"LSOA21CD": "lsoa", "LSOA21NM": "lsoa_name"})
    return gdf[["lsoa", "lsoa_name", "geometry"]].sort_values("lsoa").reset_index(drop=True)


def neighbour_pairs(gdf, tolerance=TOUCH_TOLERANCE):
    """Finds every pair of LSOAs that share a border or corner (queen contiguity).

    Returns a table with one row per direction (a -> b and b -> a).
    """
    buffered = gdf[["lsoa", "geometry"]].copy()
    buffered["geometry"] = buffered.geometry.buffer(tolerance / 2)
    pairs = gpd.sjoin(buffered, buffered, how="inner", predicate="intersects")
    pairs = pairs[pairs["lsoa_left"] != pairs["lsoa_right"]]
    pairs = pairs[["lsoa_left", "lsoa_right"]].rename(
        columns={"lsoa_left": "lsoa", "lsoa_right": "neighbour"}
    )
    return pairs.drop_duplicates().sort_values(["lsoa", "neighbour"]).reset_index(drop=True)


def neighbour_summary(pairs, all_lsoas):
    """Number of neighbours per LSOA, including any with none."""
    counts = pairs.groupby("lsoa").size().reindex(sorted(all_lsoas), fill_value=0)
    return counts.rename("n_neighbours").reset_index().rename(columns={"index": "lsoa"})


def neighbour_mean(panel, pairs, cols):
    """Average of the given columns over each LSOA's neighbours, month by month."""
    joined = pairs.merge(panel[["lsoa", "month"] + cols], left_on="neighbour", right_on="lsoa",
                         suffixes=("", "_nb"))
    joined = joined.drop(columns=["lsoa_nb"]) if "lsoa_nb" in joined.columns else joined
    means = joined.groupby(["lsoa", "month"])[cols].mean()
    means.columns = ["nb_" + c for c in cols]
    return means.reset_index()


def london_outline(gdf):
    """Single outline of London, used as a background on maps."""
    return gpd.GeoDataFrame(geometry=[gdf.geometry.union_all()], crs=gdf.crs)


def check_codes(crime_lsoas, gdf):
    """Reports crime LSOAs with no boundary and boundaries with no crime rows."""
    crime = set(crime_lsoas)
    bounds = set(gdf["lsoa"])
    return pd.Series({
        "crime LSOAs": len(crime),
        "boundary LSOAs": len(bounds),
        "crime LSOAs without boundary": len(crime - bounds),
        "boundaries without crime rows": len(bounds - crime),
    })
