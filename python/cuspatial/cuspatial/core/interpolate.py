# Copyright (c) 2020, NVIDIA CORPORATION.
import warnings

import cupy as cp
import numpy as np

from cudf import DataFrame, Series

from cuspatial._lib.interpolate import (
    cubicspline_coefficients,
    cubicspline_interpolate,
)


def _cubic_spline_coefficients(x, y, ids, prefix_sums):
    x_c = x._column
    y_c = y._column
    ids_c = ids._column
    prefix_c = prefix_sums._column
    return DataFrame._from_data(
        *cubicspline_coefficients(x_c, y_c, ids_c, prefix_c)
    )


def _cubic_spline_fit(points, points_ids, prefixes, original_t, c):
    points_c = points._column
    points_ids_c = points_ids._column
    prefixes_c = prefixes._column
    original_t_c = original_t._column
    result_column = cubicspline_interpolate(
        points_c, points_ids_c, prefixes_c, original_t_c, c
    )
    return result_column


class CubicSpline:
    """
    Fits each column of the input Series `y` to a hermetic cubic spline.

    ``cuspatial.CubicSpline`` supports basic usage identical to
    scipy.interpolate.CubicSpline::

        curve = cuspatial.CubicSpline(x, y)
        new_points = curve(np.linspace(x.min, x.max, 50))

    Parameters
    ----------
    x : cudf.Series
        1-D array containing values of the independent variable.
        Values must be real, finite and in strictly increasing order.
    y : cudf.Series
        Array containing values of the dependent variable.
    ids (Optional) : cudf.Series
        ids of each spline
    size (Optional) : cudf.Series
        fixed size of each spline
    offset (Optional) : cudf.Series
        alternative to `size`, allows splines of varying
        length. Not yet fully supported.

    Returns
    -------
    CubicSpline : callable `o`
        ``o.c`` contains the coefficients that can be used to compute new
        points along the spline fitting the original ``t`` data. ``o(n)``
        interpolates the spline coordinates along new input values ``n``.

    Note
    ----
    cuSpatial will outperform scipy when many splines are
    fit simultaneously. Data must be arranged in a structure of arrays (SoA)
    format, and the exclusive `offset` of the separate curves must also be
    passed to the function.

    Example
    -------
    # The following example only serves to demonstrate the
    # cuspatial.CubicSpline API. cuSpatial does not out perform scipy when
    # fitting a small number of curves as shown.
    >>> import cuspatial, cudf
    >>> import numpy as np
    >>> def f(x):
    ...     return x**3+4*x**2-7*x+1
    ...
    >>> x = np.array([0, 1, 2, 3, 4] + [10, 11, 12, 13, 14])
    >>> y = map(f, x)
    >>> x, y = cudf.Series(x, dtype='f4'), cudf.Series(y, dtype='f4')
    >>> offset = cudf.Series([0, 5, 10], dtype='i4')
    >>> curve = cuspatial.CubicSpline(x, y, offset=offset)
    UserWarning: fitting a small number of curves on device may suffer from
    kernel launch overheads.
    >>> x_sample1 = [*np.arange(-1, 5, 0.3)]
    >>> x_sample2 = [*np.arange(11, 12, 0.1)]
    >>> curve_ids = cudf.Series(
    ...  [0]*len(x_sample1) + [1]*len(x_sample2), dtype='i4'
    ... )
    >>> x_sample = cudf.Series(x_sample1 + x_sample2, dtype='f4')
    >>> y_sampled = curve(x_sample, curve_ids)
    >>> y_sampled1 = y_sampled[0:len(x_sample1)]
    >>> y_sampled2 = y_sampled[len(x_sample1):]
    """

    def __init__(self, x, y, ids=None, size=None, offset=None):
        # error protections:
        if len(x) < 5:
            raise ValueError(
                "Use of GPU cubic spline requires splines of length > 4"
            )
        if not isinstance(x, Series):
            raise TypeError(
                "Error: input independent vars must be cudf Series"
            )
        if not isinstance(y, (Series, DataFrame)):
            raise TypeError(
                "Error: input dependent vars must be cudf Series or DataFrame"
            )
        if not len(x) == len(y):
            raise TypeError(
                "Error: dependent and independent vars have different length"
            )
        if ids is None:
            self.ids = Series([0, 0]).astype("int32")
        else:
            if not isinstance(ids, Series):
                raise TypeError("cuspatial.CubicSpline requires a cudf.Series")
            if not ids.dtype == np.int32:
                raise TypeError("Error: int32 only supported at this time.")
            self.ids = ids
        self.size = size if size is not None else len(x)
        if not isinstance(self.size, int):
            raise TypeError("Error: size must be an integer")
        if not ((len(x) % self.size) == 0):
            raise ValueError(
                "Error: length of input is not a multiple of size"
            )
        if not isinstance(x, Series):
            raise TypeError("cuspatial.CubicSpline requires a cudf.Series")
        if not x.dtype == np.float32:
            raise TypeError("Error: float32 only supported at this time.")
        if not isinstance(y, Series):
            raise TypeError("cuspatial.CubicSpline requires a cudf.Series")
        if not y.dtype == np.float32:
            raise TypeError("Error: float32 only supported at this time.")
        self.x = x
        self.y = y
        if offset is None:
            self.offset = Series(
                cp.arange((len(x) / self.size) + 1) * self.size
            ).astype("int32")
        else:
            if not isinstance(offset, Series):
                raise TypeError("cuspatial.CubicSpline requires a cudf.Series")
            if not offset.dtype == np.int32:
                raise TypeError("Error: int32 only supported at this time.")
            self.offset = offset

        if self.offset.size < 15:
            warnings.warn(
                "Fitting a small number of curves on "
                "device may suffer from kernel launch overheads."
            )

        self.c = self._compute_coefficients()

    def _compute_coefficients(self):
        """
        Utility method used by __init__ once members have been initialized.
        """
        if isinstance(self.y, Series):
            return _cubic_spline_coefficients(
                self.x, self.y, self.ids, self.offset
            )
        else:
            c = {}
            for col in self.y.columns:
                c[col] = _cubic_spline_coefficients(
                    self.x, self.y, self.ids, self.offset
                )
            return c

    def __call__(self, coordinates, groups=None):
        """
        Interpolates new input values `coordinates` using the `.c` DataFrame
        or map of DataFrames.
        """
        if isinstance(self.y, Series):
            if groups is not None:
                self.groups = groups.astype("int32")
            else:
                self.groups = Series(
                    cp.repeat(cp.array(0), len(coordinates))
                ).astype("int32")
            result = _cubic_spline_fit(
                coordinates, self.groups, self.offset, self.x, self.c
            )
            return Series(result)
        else:
            result = DataFrame()
            for col in self.y.columns:
                result[col] = Series(
                    _cubic_spline_fit(self.c[col], coordinates)
                )
            return result
