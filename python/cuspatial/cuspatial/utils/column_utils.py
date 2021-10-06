# Copyright (c) 2020, NVIDIA CORPORATION.

import numpy as np

from cudf.api.types import is_datetime_dtype


def normalize_point_columns(*cols):
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
    dtype = np.result_type(*cols)
    if not np.issubdtype(dtype, np.floating):
        dtype = np.float32 if dtype.itemsize <= 4 else np.float64
    return (x.astype(dtype) for x in cols)


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
