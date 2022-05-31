import cupy as cp
import pandas as pd
import shapely

import cudf
from cudf.testing._utils import assert_eq

import cuspatial


def shapely_pairwise_linestring_distance(data1, data2, offset1, offset2):
    """Compute pairwise linestring distances with shapely."""

    def make_linestring(group):
        return shapely.geometry.LineString([*zip(group["x"], group["y"])])

    ridx1 = pd.RangeIndex(len(data1))
    ridx2 = pd.RangeIndex(len(data2))
    groupid1 = ridx1.map(lambda i: offset1.searchsorted(i, side="right"))
    groupid2 = ridx2.map(lambda i: offset2.searchsorted(i, side="right"))

    data1["gid"] = groupid1
    data2["gid"] = groupid2

    linestrings1 = data1.groupby("gid").apply(make_linestring)
    linestrings2 = data2.groupby("gid").apply(make_linestring)

    linestring_pairs = pd.DataFrame({"s1": linestrings1, "s2": linestrings2})
    distances = linestring_pairs.apply(
        lambda row: row["s1"].distance(row["s2"]), axis=1
    )

    return distances.reset_index(drop=True)


def test_zero_pair():
    data1 = cudf.DataFrame(
        {
            "x": [],
            "y": [],
        }
    )
    data2 = cudf.DataFrame(
        {
            "x": [],
            "y": [],
        }
    )
    offset1 = cudf.Series([], dtype="int32")
    offset2 = cudf.Series([], dtype="int32")

    got = cuspatial.pairwise_linestring_distance(
        data1["x"], data2["y"], offset1, data2["x"], data2["y"], offset2
    )
    expected = cudf.Series([], dtype="float64")

    assert_eq(got, expected)


def test_one_pair():
    data1 = cudf.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
        }
    )
    data2 = cudf.DataFrame(
        {
            "x": [2.0, 3.0],
            "y": [2.0, 3.0],
        }
    )
    offset1 = cudf.Series([0], dtype="int32")
    offset2 = cudf.Series([0], dtype="int32")

    got = cuspatial.pairwise_linestring_distance(
        offset1, data1["x"], data1["y"], offset2, data2["x"], data2["y"]
    )
    expected = shapely_pairwise_linestring_distance(
        data1.to_pandas(),
        data2.to_pandas(),
        offset1.to_pandas(),
        offset2.to_pandas(),
    )

    assert_eq(got, expected)


def test_two_pairs():
    data1 = cudf.DataFrame(
        {
            "x": [0.0, 1.0, 5.0, 7.0, 8.0],
            "y": [0.0, 1.0, 10.2, 11.4, 12.8],
        }
    )
    data2 = cudf.DataFrame(
        {
            "x": [2.0, 3.0, -8.0, -10.0, -13.0, -3.0],
            "y": [2.0, 3.0, -8.0, -5.0, -15.0, -6.0],
        }
    )
    offset1 = cudf.Series([0, 3], dtype="int32")
    offset2 = cudf.Series([0, 2], dtype="int32")

    got = cuspatial.pairwise_linestring_distance(
        offset1, data1["x"], data1["y"], offset2, data2["x"], data2["y"]
    )
    expected = shapely_pairwise_linestring_distance(
        data1.to_pandas(),
        data2.to_pandas(),
        offset1.to_pandas(),
        offset2.to_pandas(),
    )

    assert_eq(got, expected)


def test_100_randomized_input():
    rng = cp.random.RandomState(0)

    max_linestring_points = 10
    size = 100

    offset1 = rng.randint(2, max_linestring_points, size=(size,))
    offset2 = rng.randint(2, max_linestring_points, size=(size,))

    offset1 = cp.cumsum(offset1)
    offset2 = cp.cumsum(offset2)

    num_points_1 = int(offset1[-1])
    num_points_2 = int(offset2[-1])

    offset1 = cp.concatenate((cp.zeros((1,)), offset1[:-1]))
    offset2 = cp.concatenate((cp.zeros((1,)), offset2[:-1]))

    points1_x = rng.uniform(-1, 1, (num_points_1,))
    points1_y = rng.uniform(-1, 1, (num_points_1,))

    points2_x = rng.uniform(0.5, 2.5, (num_points_2,))
    points2_y = rng.uniform(0.5, 2.5, (num_points_2,))

    got = cuspatial.pairwise_linestring_distance(
         offset1, points1_x, points1_y, offset2, points2_x, points2_y
    )
    expected = shapely_pairwise_linestring_distance(
        pd.DataFrame({"x": points1_x.get(), "y": points1_y.get()}),
        pd.DataFrame({"x": points2_x.get(), "y": points2_y.get()}),
        pd.Series(offset1.get()),
        pd.Series(offset2.get()),
    )

    assert_eq(got, expected)
