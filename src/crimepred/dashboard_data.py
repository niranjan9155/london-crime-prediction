"""Prepares a small map file for the dashboard.

The full boundary file is about 70 MB. Here we are simplifying the shapes and
converting them to latitude/longitude, so the dashboard map loads quickly.

Run with:  python -m crimepred.dashboard_data
"""

import json

from crimepred import config, spatial

MAP_FILE = config.OUTPUTS / "dashboard" / "lsoa_map.geojson"
SIMPLIFY_METRES = 8


def _round_coords(obj, digits=5):
    """Rounds every coordinate (5 decimals is about 1 m) to keep the file small."""
    if isinstance(obj, (list, tuple)):
        if obj and isinstance(obj[0], (int, float)):
            return [round(v, digits) for v in obj]
        return [_round_coords(v, digits) for v in obj]
    return obj


def build_map_file(path=MAP_FILE):
    gdf = spatial.load_boundaries()
    gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_METRES, preserve_topology=True)
    gdf = gdf.to_crs("EPSG:4326")
    data = json.loads(gdf[["lsoa", "lsoa_name", "geometry"]].to_json())
    for feature in data["features"]:
        feature["geometry"]["coordinates"] = _round_coords(feature["geometry"]["coordinates"])
        feature.pop("id", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")))
    print(f"Saved {path.name} ({path.stat().st_size / 1e6:.1f} MB, {len(data['features'])} areas)")
    return path


if __name__ == "__main__":
    build_map_file()
