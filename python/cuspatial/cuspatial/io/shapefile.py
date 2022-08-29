# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf import DataFrame, Series

from cuspatial._lib.shapefile_reader import (
    WindingOrder,
    read_polygon_shapefile as cpp_read_polygon_shapefile,
)


def read_polygon_shapefile(
    filename, outer_ring_order=WindingOrder.COUNTER_CLOCKWISE
):
    """
    Reads polygon geometry from an ESRI shapefile into GPU memory.

    Parameters
    ----------
    filename : str, pathlike
        ESRI Shapefile file path (usually ends in ``.shp``)
    winding_order : WindingOrder(Enum)
        COUNTER_CLOCKWISE: ESRI Format, or CLOCKWISE: Simple Feature

    Returns
    -------
    result  : tuple (cudf.Series, cudf.Series, cudf.DataFrame)
    poly_offsets   : cudf.Series(dtype=np.int32)
        Offsets of the first ring in each polygon
    ring_offsets   : cudf.Series(dtype=np.int32)
        Offsets of the first point in each ring
    points  : cudf.DataFrame
        DataFrame of all points in the shapefile
            x : cudf.Series(dtype=np.float64)
                x-components of each polygon's points
            y : cudf.Series(dtype=np.float64)
                y-components of each polygon's points
    """
    result = cpp_read_polygon_shapefile(filename, outer_ring_order)
    f_pos = Series(result[0], name="f_pos")
    r_pos = Series(result[1], name="r_pos")
    return (f_pos, r_pos, DataFrame({"x": result[2], "y": result[3]}))
