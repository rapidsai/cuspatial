# Copyright (c) 2023, NVIDIA CORPORATION.

from cudf.core.dtypes import ListDtype


def point_dtype(base_dtype):
    return ListDtype(base_dtype)


def multipoint_dtype(base_dtype):
    return ListDtype(ListDtype(base_dtype))


def linestring_dtype(base_dtype):
    return ListDtype(ListDtype(ListDtype(base_dtype)))


def polygon_dtype(base_dtype):
    return ListDtype(ListDtype(ListDtype(ListDtype(base_dtype))))
