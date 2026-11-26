"""Downloads the two public datasets used in the project.

Run with:  python -m crimepred.download
"""

import json

import pandas as pd
import requests

from crimepred import config

TIMEOUT = 120


def download_crime(force=False):
    """Saves the MPS LSOA crime file into data/raw."""
    path = config.CRIME_FILE
    if path.exists() and not force:
        print(f"Crime file already present: {path.name}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading MPS LSOA crime file ...")
    resp = requests.get(config.CRIME_URL, timeout=TIMEOUT)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    print(f"Saved {path.name} ({len(resp.content) / 1e6:.1f} MB)")
    return path


def _query_boundaries(codes):
    """Asks the ONS service for one batch of LSOA polygons."""
    quoted = ",".join(f"'{c}'" for c in codes)
    params = {
        "where": f"LSOA21CD IN ({quoted})",
        "outFields": "LSOA21CD,LSOA21NM",
        "outSR": "27700",
        "returnGeometry": "true",
        "f": "geojson",
    }
    resp = requests.post(config.BOUNDARY_URL, data=params, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if "features" not in data:
        raise RuntimeError(f"Unexpected reply from ONS service: {str(data)[:200]}")
    return data["features"]


def download_boundaries(force=False, batch_size=150):
    """Saves 2021 LSOA boundaries for every LSOA that appears in the crime file."""
    path = config.BOUNDARY_FILE
    if path.exists() and not force:
        print(f"Boundary file already present: {path.name}")
        return path
    codes = sorted(pd.read_csv(config.CRIME_FILE, usecols=["LSOA Code"])["LSOA Code"].unique())
    print(f"Downloading boundaries for {len(codes)} LSOAs ...")
    features = []
    for start in range(0, len(codes), batch_size):
        features.extend(_query_boundaries(codes[start:start + batch_size]))
        print(f"  {min(start + batch_size, len(codes))}/{len(codes)}")
    collection = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::27700"}},
        "features": features,
    }
    path.write_text(json.dumps(collection))
    print(f"Saved {path.name} with {len(features)} polygons")
    return path


if __name__ == "__main__":
    download_crime()
    download_boundaries()
