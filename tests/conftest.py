"""Small hand-made inputs shared by the tests."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box


@pytest.fixture
def raw_crime():
    """Two LSOAs, two crime types, four months, in the same layout as the MPS file."""
    return pd.DataFrame(
        {
            "LSOA Code": ["E1", "E1", "E2", "E2"],
            "LSOA Name": ["Area 1", "Area 1", "Area 2", "Area 2"],
            "Borough": ["B1", "B1", "B2", "B2"],
            "Group": ["THEFT", "VIOLENCE AGAINST THE PERSON", "THEFT", "BURGLARY"],
            "SubGroup": ["OTHER THEFT", "ASSAULT", "OTHER THEFT", "RES BURGLARY"],
            "202601": [1, 2, 0, 0],
            "202602": [3, 1, 2, 2],
            "202603": [0, 0, 5, 1],
            "202604": [4, 4, 1, 0],
        }
    )


@pytest.fixture
def grid_boundaries():
    """3 x 3 grid of 100 m squares named G00 ... G22 (row, column)."""
    rows = []
    for r in range(3):
        for c in range(3):
            rows.append({"lsoa": f"G{r}{c}", "geometry": box(c * 100, r * 100, (c + 1) * 100, (r + 1) * 100)})
    return gpd.GeoDataFrame(rows, crs="EPSG:27700")
