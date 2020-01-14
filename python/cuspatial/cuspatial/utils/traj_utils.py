# Copyright 2019, NVIDIA Corporation

import rmm
from numba import cuda

import cudf


def get_ts_struct(ts):
    y = ts & 0x3F
    ts = ts >> 6
    m = ts & 0xF
    ts = ts >> 4
    d = ts & 0x1F
    ts = ts >> 5
    hh = ts & 0x1F
    ts = ts >> 5
    mm = ts & 0x3F
    ts = ts >> 6
    ss = ts & 0x3F
    ts = ts >> 6
    wd = ts & 0x8
    ts = ts >> 3
    yd = ts & 0x1FF
    ts = ts >> 9
    ms = ts & 0x3FF
    ts = ts >> 10
    pid = ts & 0x3FF

    return y, m, d, hh, mm, ss, wd, yd, ms, pid


@cuda.jit
def gpu_its_timestamp_int64_to_datetime64ms(its_timestamp_int64_col, out):
    i = cuda.grid(1)
    if i < its_timestamp_int64_col.size:
        ts = its_timestamp_int64_col[i]
        y = ts & 0x3F
        ts = ts >> 6
        m = ts & 0xF
        ts = ts >> 4
        d = ts & 0x1F
        ts = ts >> 5
        hh = ts & 0x1F
        ts = ts >> 5
        mm = ts & 0x3F
        ts = ts >> 6
        ss = ts & 0x3F
        ts = ts >> 6
        wd = ts & 0x8  # noqa: F841
        ts = ts >> 3
        yd = ts & 0x1FF  # noqa: F841
        ts = ts >> 9
        ms = ts & 0x3FF
        ts = ts >> 10
        pid = ts & 0x3FF  # noqa: F841
        time = ms
        time = time + ss * 1000
        time = time + mm * 60000
        time = time + hh * 3600000
        time = time + d * 86400000
        time = time + m * 2628000000
        time = time + y * 31536000000
        out[i] = time


def its_timestamp_int64_to_datetime64ms(its_timestamp_int64_col):
    out = rmm.device_array(its_timestamp_int64_col.size, dtype="int64")
    if out.size > 0:
        gpu_its_timestamp_int64_to_datetime64ms.forall(out.size)(
            its_timestamp_int64_col, out
        )
    return cudf.Series(out).astype("datetime64[ms]")
