import pandas as pd

from crimepred import spatial


def test_queen_neighbours_on_grid(grid_boundaries):
    pairs = spatial.neighbour_pairs(grid_boundaries)
    counts = pairs.groupby("lsoa").size()
    assert counts["G11"] == 8  # centre square touches all others
    assert counts["G00"] == 3  # corner square
    assert counts["G01"] == 5  # edge square


def test_neighbour_pairs_are_symmetric(grid_boundaries):
    pairs = spatial.neighbour_pairs(grid_boundaries)
    forward = set(zip(pairs["lsoa"], pairs["neighbour"]))
    backward = set(zip(pairs["neighbour"], pairs["lsoa"]))
    assert forward == backward
    assert not any(a == b for a, b in forward)


def test_small_gap_still_counts_as_touching(grid_boundaries):
    moved = grid_boundaries.copy()
    moved.loc[moved["lsoa"] == "G22", "geometry"] = moved.loc[moved["lsoa"] == "G22", "geometry"].translate(0.4, 0)
    pairs = spatial.neighbour_pairs(moved, tolerance=1.0)
    assert ("G22", "G21") in set(zip(pairs["lsoa"], pairs["neighbour"]))


def test_neighbour_mean(grid_boundaries):
    pairs = spatial.neighbour_pairs(grid_boundaries)
    panel = pd.DataFrame({"lsoa": grid_boundaries["lsoa"], "month": "202601",
                          "crimes": range(9)})
    means = spatial.neighbour_mean(panel, pairs, ["crimes"]).set_index("lsoa")
    # G00 touches G01 (1), G10 (3) and G11 (4).
    assert means.loc["G00", "nb_crimes"] == (1 + 3 + 4) / 3


def test_code_check(grid_boundaries):
    result = spatial.check_codes(["G00", "G01", "X99"], grid_boundaries)
    assert result["crime LSOAs without boundary"] == 1
    assert result["boundaries without crime rows"] == 7
