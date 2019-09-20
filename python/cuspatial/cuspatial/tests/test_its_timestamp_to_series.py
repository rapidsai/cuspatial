# Copyright (c) 2019, NVIDIA CORPORATION.

import cudf
from cudf.tests.utils import assert_eq

from cuspatial.utils.traj_utils import its_timestamp_int64_to_datetime64ms


def test_zero():
    assert_eq(
        its_timestamp_int64_to_datetime64ms(cudf.Series([0]).data.mem),
        cudf.Series([0]).astype("datetime64[ms]"),
    )


def test_one():
    assert_eq(
        its_timestamp_int64_to_datetime64ms(cudf.Series([1]).data.mem),
        cudf.Series(["1971-01-01"]).astype("datetime64[ms]"),
    )


def test_zeroes():
    assert_eq(
        its_timestamp_int64_to_datetime64ms(cudf.Series([0, 0]).data.mem),
        cudf.Series([0, 0]).astype("datetime64[ms]"),
    )


def test_ones():
    assert_eq(
        its_timestamp_int64_to_datetime64ms(cudf.Series([1, 1]).data.mem),
        cudf.Series(["1971-01-01", "1971-01-01"]).astype("datetime64[ms]"),
    )
