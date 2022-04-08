# Copyright (c) 2019, NVIDIA CORPORATION.
import operator

import rmm
from numba import cuda


@cuda.jit
def binarize(in_col, out, width):
    """Convert any positive integer to a binary array."""
    i = cuda.grid(1)
    if i < in_col.size:
        n = in_col[i]
        idx = width - 1

        out[i, idx] = operator.mod(n, 2)
        idx -= 1

        while n > 1:
            n = operator.rshift(n, 1)
            out[i, idx] = operator.mod(n, 2)
            idx -= 1


def apply_binarize(in_col, width):
    buf = rmm.DeviceBuffer(size=(in_col.size * width))
    out = cuda.as_cuda_array(buf).view("int8").reshape((in_col.size, width))
    if out.size > 0:
        out[:] = 0
        binarize.forall(out.size)(in_col, out, width)
    return out


def pip_bitmap_column_to_binary_array(polygon_bitmap_column, width):
    """Convert the bitmap output of point_in_polygon
    to an array of 0s and 1s.
    """
    binary_maps = apply_binarize(polygon_bitmap_column.data_array_view, width)
    return binary_maps
