# Copyright (c) 2020, NVIDIA CORPORATION.

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
    result_table = cubicspline_coefficients(x_c, y_c, ids_c, prefix_c)
    result = DataFrame._from_table(result_table)
    return result


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

    Parameters
    ----------
    t : cudf.Series
        time sample values. Must be monotonically increasing.
    y : cudf.Series
        columns to have curves fit to according to x
    ids (Optional) : cudf.Series
        ids of each spline
    size (Optional) : cudf.Series
        fixed size of each spline
    prefixes (Optional) : cudf.Series
        alternative to `size`, allows splines of varying
        length. Not yet fully supported.

    Returns
    -------
    CubicSpline object `o`. `o.c` contains the coefficients that can be
    used to compute new points along the spline fitting the original `t`
    data. `o(n)` interpolates the spline coordinates along new input
    values `n`.

    cuspatial.CubicSpline supports two usage patterns: The first is
    identical to scipy.interpolate.CubicSpline:

        curve = cuspatial.CubicSpline(t, y)
        new_points = curve(np.linspace(t.min, t.max, 50))

    This allows API parity with scipy. This isn't recommended, as scipy
    host based interpolation performance is likely to exceed GPU performance
    for a single curve.

    cuspatial massively outperforms scipy however when many
    splines are fit simultaneously. Data must be arranged in a SoA format,
    and the inclusive/exclusive prefix_sum of the separate curves must also
    be passed to the function.

        t = cudf.Series(np.repeat(np.arange(100), 1000)).astype('float32')
        y = cudf.Series(np.random.random(100*1000)).astype('float32')
        prefix_sum = cudf.Series(np.arange(1000)*100).astype('int32')
        new_samples = cudf.Series(np.repeat(np.linspace(0, 100, 1000), 1000)
            .astype('float32'))

        curve = cuspatial.CubicSpline(t, y, prefixes=prefix_sum)
        new_points = curve(new_samples, prefix_sum*10)

    """

    def __init__(self, t, y, ids=None, size=None, prefixes=None):
        """
        Computes various error preconditions on the input data, then
        calls C++/Thrust code to compute cubic splines for each set of input
        coordinates in parallel.
        """
        # error protections:
        if len(t) < 5:
            raise ValueError(
                "Use of GPU cubic spline requires splines of length > 4"
            )
        if not isinstance(t, Series):
            raise TypeError(
                "Error: input independent vars must be cudf Series"
            )
        if not isinstance(y, (Series, DataFrame)):
            raise TypeError(
                "Error: input dependent vars must be cudf Series or DataFrame"
            )
        if not len(t) == len(y):
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
        self.size = size if size is not None else len(t)
        if not isinstance(self.size, int):
            raise TypeError("Error: size must be an integer")
        if not ((len(t) % self.size) == 0):
            raise ValueError(
                "Error: length of input is not a multiple of size"
            )
        if not isinstance(t, Series):
            raise TypeError("cuspatial.CubicSpline requires a cudf.Series")
        if not t.dtype == np.float32:
            raise TypeError("Error: float32 only supported at this time.")
        if not isinstance(y, Series):
            raise TypeError("cuspatial.CubicSpline requires a cudf.Series")
        if not y.dtype == np.float32:
            raise TypeError("Error: float32 only supported at this time.")
        self.t = t
        self.y = y
        if prefixes is None:
            self.prefix = Series(
                np.arange((len(t) / self.size) + 1) * self.size
            ).astype("int32")
        else:
            if not isinstance(prefixes, Series):
                raise TypeError("cuspatial.CubicSpline requires a cudf.Series")
            if not prefixes.dtype == np.int32:
                raise TypeError("Error: int32 only supported at this time.")
            self.prefix = prefixes

        self.c = self._compute_coefficients()

    def _compute_coefficients(self):
        """
        Utility method used by __init__ once members have been initialized.
        """
        if isinstance(self.y, Series):
            return _cubic_spline_coefficients(
                self.t, self.y, self.ids, self.prefix
            )
        else:
            c = {}
            for col in self.y.columns:
                c[col] = _cubic_spline_coefficients(
                    self.t, self.y, self.ids, self.prefix
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
                self.groups = Series(np.repeat(0, len(self.t))).astype("int32")
            result = _cubic_spline_fit(
                coordinates, self.groups, self.prefix, self.t, self.c
            )
            return Series(result)
        else:
            result = DataFrame()
            for col in self.y.columns:
                result[col] = Series(
                    _cubic_spline_fit(self.c[col], coordinates)
                )
            return result
