import pandas as pd
import shapely

import cudf
from cudf.testing._utils import assert_eq

import cuspatial


def shapely_pairwise_linestring_distance(data1, data2, offset1, offset2):
    def make_linestring(group):
        return shapely.geometry.LineString(
            [(group["x"][i], group["y"][i]) for i in range(group.shape[0])]
        )

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

    got = cuspatial.pairwise_polyline_distance(
        data1["x"], data2["y"], offset1, data2["x"], data2["y"], offset2
    )
    expected = shapely_pairwise_linestring_distance(
        data1.to_pandas(),
        data2.to_pandas(),
        offset1.to_pandas(),
        offset2.to_pandas(),
    )

    assert_eq(got, expected)
