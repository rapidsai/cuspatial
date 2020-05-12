import numpy as np

from cudf.utils.dtypes import is_datetime_dtype


def normalize_point_columns(xs, ys):
    """
    Normalize the input columns by inferring a common floating point dtype.

    If the common dtype isn't a floating point dtype, promote the common dtype
    to float64.

    Parameters
    ----------
    {params}

    Returns
    -------
    tuple : the input columns cast to the inferred common floating point dtype
    """
    dtype = np.result_type(xs.dtype, ys.dtype)
    if not np.issubdtype(dtype, np.floating):
        dtype = np.float32 if dtype.itemsize <= 4 else np.float64
    return xs.astype(dtype), ys.astype(dtype)


def normalize_timestamp_column(ts, fallback_dtype="datetime64[ms]"):
    """
    Normalize the input timestamp column to one of the cuDF timestamp dtypes.

    If the input column's dtype isn't an np.datetime64, cast the column to the
    supplied `fallback_dtype` parameter.

    Parameters
    ----------
    {params}

    Returns
    -------
    column : the input column
    """
    return ts if is_datetime_dtype(ts.dtype) else ts.astype(fallback_dtype)


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
