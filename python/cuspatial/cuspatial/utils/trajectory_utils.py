# Copyright (c) 2020, NVIDIA CORPORATION.


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
