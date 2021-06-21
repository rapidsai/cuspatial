# Copyright (c) 2021 NVIDIA CORPORATION

from typing import TypeVar, Union

import numpy as np
import pandas as pd

import cudf

T = TypeVar("T", bound="GeoArrowBuffers")


class GeoArrowBuffers:
    """A GPU GeoArrowBuffers object.

    Parameters
    ----------
    data : A dict or a GeoArrowBuffers object.

    The GeoArrow format specifies a tabular data format for geometry
    information. Supported types include `Point`, `MultiPoint`, `LineString`,
    `MultiLineString`, `Polygon`, and `MultiPolygon`.  In order to store
    these coordinate types in a strictly tabular fashion, columns are
    created for Points, MultiPoints, LineStrings, and Polygons.
    MultiLines and MultiPolygons are stored in the same data structure
    as LineStrings and Polygons. GeoArrowBuffers are constructed from a dict
    of host buffers with accepted keys:

    * points_xy
    * points_z
    * multipoints_xy
    * multipoints_z
    * multipoints_offsets
    * lines_xy
    * lines_z
    * lines_offsets
    * mlines
    * polygons_xy
    * polygons_z
    * polygons_polygons
    * polygons_rings
    * mpolygons

    There are no correlations in length between any of the above columns.
    Accepted host buffer object types include python list and any type that
    implements numpy's `__array__interface__` protocol.

    GeoArrow Format

    GeoArrow format packs complex geometry types into 14 single-column Arrow
    tables. This description is included for better understanding GeoArrow
    format. Interacting with the GeoArrowBuffers is only required if you want
    to convert cudf data to GeoPandas objects without starting from GeoPandas.

    The points geometry is the simplest: N points are stored in a length 2*N
    buffer with interleaved x,y coordinates. An optional z buffer of length N
    can be used.

    The multipoints geometry is the second simplest - identical to points,
    with the addition of a multipoints_offsets buffer. The offsets buffer
    stores N+1 indexes. The first multipoint is specified by 0, which is always
    stored in offsets[0], and offsets[1], which is the length in points of
    the first multipoint geometry. Subsequent multipoints are the prefix-sum of
    the lengths of previous multipoints.

    Consider::

        buffers = GeoArrowBuffers({
            "multipoints_xy":
                [0, 0, 0, 1, 0, 2, 1, 0, 1, 1, 1, 2, 2, 0, 2, 1, 2, 2],
            "multipoints_offsets":
                [0, 6, 12, 18]
        })

    which encodes the following GeoPandas Series::

        series = geopandas.Series([
            MultiPoint((0, 0), (0, 1), (0, 2)),
            MultiPoint((1, 0), (1, 1), (1, 2)),
            MultiPoint((2, 0), (2, 1), (2, 2)),
        ])

    LineString geometry is more complicated than multipoints because the
    format allows for the use of LineStrings and MultiLineStrings in the same
    buffer, via the mlines key::

        buffers = GeoArrowBuffers({
            "lines_xy":
                [0, 0, 0, 1, 0, 2, 1, 0, 1, 1, 1, 2, 2, 0, 2, 1, 2, 2, 3, 0,
                 3, 1, 3, 2, 4, 0, 4, 1, 4, 2],
            "lines_offsets":
                [0, 6, 12, 18, 24, 30],
            "mlines":
                [1, 3]
        })

    Which encodes a GeoPandas Series::

        series = geopandas.Series([
            LineString((0, 0), (0, 1), (0, 2)),
            MultiLineString([(1, 0), (1, 1), (1, 2)],
                            [(2, 0), (2, 1), (2, 2)],
            )
            LineString((3, 0), (3, 1), (3, 2)),
            LineString((4, 0), (4, 1), (4, 2)),
        ])

    Polygon geometry includes `mpolygons` for MultiPolygons similar to the
    LineString geometry. Polygons are encoded using the same format as
    Shapefiles, with left-wound external rings and right-wound internal rings.
    An exact example of `GeoArrowBuffers` to `geopandas.Series` is left to the
    reader as an exercise. Convert any GeoPandas `Series` or `DataFrame` with
    `cuspatial.from_geopandas(geopandas_object)`.

    Notes
    -----
    Legacy cuspatial algorithms depend on separated x and y columns. Access
    them with the `.x` and `.y` properties.

    Examples
    --------
    GeoArrowBuffers accept a dict as argument. Valid keys are in the bullet
    list above. Valid values are any datatype that implements numpy's
    `__array_interface__`. Any or all of the four basic geometry types is
    supported as argument::

        buffers = GeoArrowBuffers({
            "points_xy":
                [0, 0, 0, 1, 0, 2, 1, 0, 1, 1, 1, 2, 2, 0, 2, 1, 2, 2],
            "multipoints_xy":
                [0, 0, 0, 1, 0, 2, 1, 0, 1, 1, 1, 2, 2, 0, 2, 1, 2, 2],
            "multipoints_offsets":
                [0, 6, 12, 18]
            "lines_xy":
                [0, 0, 0, 1, 0, 2, 1, 0, 1, 1, 1, 2, 2, 0, 2, 1, 2, 2, 3, 0,
                 3, 1, 3, 2, 4, 0, 4, 1, 4, 2],
            "lines_offsets":
                [0, 6, 12, 18, 24, 30],
            "mlines":
                [1, 3]
            "polygons_xy":
                [0, 0, 0, 1, 0, 2, 1, 0, 1, 1, 1, 2, 2, 0, 2, 1, 2, 2, 3, 0,
                 3, 1, 3, 2, 4, 0, 4, 1, 4, 2],
            "polygons_polygons": [0, 1, 2],
            "polygons_rings": [0, 1, 2],
            "mpolygons": [1, 3],
        })

    or another GeoArrowBuffers::

        buffers2 = GeoArrowBuffers(buffers)
    """

    def __init__(self, data: Union[dict, T], data_locale: object = cudf):
        self._points = None
        self._multipoints = None
        self._lines = None
        self._polygons = None
        if isinstance(data, dict):
            if data.get("points_xy") is not None:
                self._points = CoordinateArray(
                    data["points_xy"], data_locale=data_locale
                )
            if data.get("mpoints_xy") is not None:
                if data.get("mpoints_offsets") is None:
                    raise ValueError("mpoints must include offsets buffer")
                self._multipoints = MultiPointArray(
                    data["mpoints_xy"],
                    data["mpoints_offsets"],
                    data_locale=data_locale,
                )
            if data.get("lines_xy") is not None:
                if data.get("lines_offsets") is None:
                    raise ValueError("lines_xy must include offsets buffer")
                self._lines = LineArray(
                    data["lines_xy"],
                    data["lines_offsets"],
                    data.get("mlines"),
                    data_locale=data_locale,
                )
            if data.get("polygons_xy") is not None:
                if (
                    data.get("polygons_polygons") is None
                    or data.get("polygons_rings") is None
                ):
                    raise ValueError(
                        "polygons_xy must include polygons and" "rings buffers"
                    )
                self._polygons = PolygonArray(
                    data["polygons_xy"],
                    data["polygons_polygons"],
                    data["polygons_rings"],
                    data.get("mpolygons"),
                    data_locale=data_locale,
                )
        elif isinstance(data, GeoArrowBuffers):
            if data.points is not None:
                self._points = CoordinateArray(
                    data.points.xy, data.points.z, data_locale=data_locale
                )
            if data.multipoints is not None:
                self._multipoints = MultiPointArray(
                    data.multipoints.xy,
                    data.multipoints.offsets,
                    data.multipoints.z,
                    data_locale=data_locale,
                )
            if data.lines is not None:
                self._lines = LineArray(
                    data.lines.xy,
                    data.lines.offsets,
                    data.lines.mlines,
                    data.lines.z,
                    data_locale=data_locale,
                )
            if data.polygons is not None:
                self._polygons = PolygonArray(
                    data.polygons.xy,
                    data.polygons.polys,
                    data.polygons.rings,
                    data.polygons.mpolys,
                    data.polygons.z,
                    data_locale=data_locale,
                )
        else:
            raise TypeError(
                f"Invalid type passed to GeoArrowBuffers ctor {type(data)}"
            )

    @property
    def points(self):
        """
        A simple numeric column. x and y coordinates are interleaved such that
        even coordinates are x axis and odd coordinates are y axis.
        """
        return self._points

    @property
    def multipoints(self):
        """
        Similar to the Points column with the addition of an offsets column.
        The offsets column stores the comparable sizes and coordinates of each
        MultiPoint in the GeoArrowBuffers.
        """
        return self._multipoints

    @property
    def lines(self):
        """
        Contains the coordinates column, an offsets column, and a
        mlines column. The mlines column  is optional. The mlines column stores
        the indices of the offsets that indicate the beginning and end of each
        MultiLineString segment. The absence of an `mlines` column indicates
        there are no `MultiLineStrings` in the data source, only `LineString`s.
        """
        return self._lines

    @property
    def polygons(self):
        """
        Contains the coordinates column, a rings column specifying
        the beginning and end of every polygon, a polygons column specifying
        the beginning, or exterior, ring of each polygon and the end ring.
        All rings after the first ring are interior rings.  Finally a
        mpolygons column stores the offsets of the polygons that should be
        grouped into MultiPolygons.
        """
        return self._polygons

    def __len__(self):
        """
        The numer of unique geometries stored in this GeoArrowBuffers.
        """
        points_length = len(self._points) if self.points is not None else 0
        lines_length = len(self._lines) if self.lines is not None else 0
        multipoints_length = (
            len(self._multipoints) if self.multipoints is not None else 0
        )
        polygons_length = (
            len(self._polygons) if self.polygons is not None else 0
        )
        return (
            points_length + lines_length + multipoints_length + polygons_length
        )

    def copy(self, deep=True):
        """
        Create a copy of all of the GPU-backed data structures in this
        GeoArrowBuffers.
        """
        points = self.points.copy(deep)
        multipoints = self.multipoints.copy(deep)
        lines = self.lines.copy(deep)
        polygons = self.polygons.copy(deep)
        result = GeoArrowBuffers(
            {
                "points_xy": points.xy,
                "points_z": points.z,
                "multipoints_xy": multipoints.xy,
                "multipoints_offsets": multipoints.offsets,
                "multipoints_z": multipoints.z,
                "lines_xy": lines.xy,
                "lines_offsets": lines.offsets,
                "mlines": lines.mlines,
                "lines_z": lines.z,
                "polygons_xy": polygons.xy,
                "polygons_polygons": polygons.polys,
                "polygons_rings": polygons.rings,
                "polygons_mpolygons": polygons.mpolys,
                "polygons_z": polygons.z,
            }
        )
        return result

    def to_host(self):
        return GeoArrowBuffers(self, data_locale=pd)

    def __repr__(self):
        return (
            f"{self.points}\n"
            f"{self.multipoints}\n"
            f"{self.lines}\n"
            f"{self.polygons}\n"
        )


class CoordinateArray:
    def __init__(self, xy, z=None, data_locale=cudf):
        """
        A GeoArrow column of points. The CoordinateArray stores all of the
        points within a single data source, typically a cuspatial.GeoSeries,
        in the format specified by GeoArrow.
        """
        self.data_location = data_locale
        self.xy = xy
        self.z = z

    @property
    def data_location(self):
        """
        A `Series` supporting python library, expected to be either `cudf` or
        `pandas`. If `cudf`, GeoArrowBuffer is stored on the GPU and used in
        cuspatial operations. If `pandas`, GeoArrowBuffer is stored on Host and
        used for creating GeoPandas objects.
        """
        return self._data_location

    @data_location.setter
    def data_location(self, data_location):
        if data_location not in (cudf, pd):
            raise NotImplementedError(
                "only cudf and pandas CoordinateArrays "
                "are supported at this time"
            )
        else:
            self._data_location = data_location

    def _serialize(self, data):
        try:
            if self._data_location == pd:
                if isinstance(data, cudf.Series):
                    return data.to_pandas()
                else:
                    return self._data_location.Series(data)
            elif self._data_location == cudf:
                return self._data_location.Series(data)
            else:
                raise ValueError(
                    f"{type(self._data_location)} incorrect data"
                    f"location: pandas or cudf"
                )
        except Exception as e:
            raise TypeError(e, f"{type(data)} is not supported in cuspatial")

    @property
    def xy(self):
        """
        The coordinates of this Geometry Column in interleaved format
        [x,y,...,x,y].
        """
        return self._xy

    @xy.setter
    def xy(self, xy):
        if (len(xy) % 2) != 0:
            raise ValueError("xy must have even length")
        self._xy = self._serialize(xy)

    @property
    def z(self):
        """
        An optional third dimension for this Geometry.
        """
        return self._z

    @z.setter
    def z(self, z):
        self._z = self._serialize(z)

    def __getitem__(self, index):
        if isinstance(index, slice):
            new_index = slice(index.start * 2, index.stop * 2 + 1, index.step)
            return self.xy.iloc[new_index]
        return self.xy.iloc[(index * 2) : (index * 2) + 2]

    def __repr__(self):
        return f"xy:\n" f"{self.xy.__repr__()}\n"

    def copy(self, deep=True):
        """
        Create a copy of all points.

        Parameters
        ----------
        deep : Boolean.
            If set to False, a new object referencing the same
            set of GPU memory will be created. If True, GPU memory will be
            copied.
        """
        if self.z is not None:
            z = self.z.copy(deep)
        else:
            z = None
        result = type(self)(self.xy.copy(deep), z)
        return result

    @property
    def x(self):
        """
        Return packed x-coordinates of this GeometryArray object.
        """
        return self.xy[slice(0, None, 2)].reset_index(drop=True)

    @property
    def y(self):
        """
        Return packed y-coordinates of this GeometryArray object.
        """
        return self.xy[slice(1, None, 2)].reset_index(drop=True)

    def __len__(self):
        return len(self.xy) // 2


class PointsArray(CoordinateArray):
    """
    A GeoArrow column of points. Every pair is the `[x,y]` coordinate of the
    position/2th point in this data source. `z` can be included optionally.
    """

    def __init__(self, xy, z=None, data_locale=cudf):
        super().__init__(xy, z, data_locale=data_locale)


class OffsetArray(CoordinateArray):
    def __init__(self, xy, offsets, z=None, data_locale=cudf):
        """
        A GeoArrow column of offset geometries. This is the base class of all
        complex GeoArrow geometries. MultiLineStrings and MultiPolygons store
        extra metadata to identify individual geometry boundaries, but are also
        based on OffsetArray.
        """
        super().__init__(xy, z, data_locale=data_locale)
        self.offsets = offsets

    @property
    def offsets(self):
        """
        The offsets column of a geometry object contains the positions of
        each sub-geometry. Each pair of values in the offsets column specifies
        the beginning index of a sub-geometry in the `.xy` column and beginning
        of the subsequent sub-geometry. Contains `n+1` values where `n` is the
        number of sub-geometries. Starts always with 0.
        """
        return self._offsets

    @offsets.setter
    def offsets(self, offsets):
        self._offsets = self._serialize(offsets)

    def __getitem__(self, index):
        if isinstance(index, slice):
            rindex = index
        else:
            rindex = slice(index, index + 1, 1)
        new_slice = slice(self.offsets[rindex.start], None)
        if rindex.stop < len(self.offsets):
            new_slice = slice(new_slice.start, self.offsets[rindex.stop])
        result = self.xy[new_slice]
        return result

    def __repr__(self):
        return (
            f"{super().__repr__()}"
            f"offsets:\n"
            f"{self.offsets.__repr__()}\n"
        )

    def copy(self, deep=True):
        if self.z is not None:
            z = self.z.copy(deep)
        else:
            z = None
        result = OffsetArray(self.xy.copy(deep), self.offsets.copy(deep), z)
        return result

    def __len__(self):
        return len(self._offsets) - 1


class LineArray(OffsetArray):
    def __init__(self, xy, lines, mlines, z=None, data_locale=cudf):
        """
        A GeoArrow column of LineStrings. This format stores LineStrings and
        MultiLineStrings from a single data source (Such as a GeoSeries).
        Offset coordinates stored between pairs of mlines offsets specify
        MultiLineStrings. Offset values that do not fall within a pair of
        mlines are simple LineStrings.

        Parameters
        ---
        xy : cudf.Series
        lines : cudf.Series
        mlines : cudf.Series
        z : cudf.Series (optional)
        """
        super().__init__(xy, lines, z, data_locale=data_locale)
        self.mlines = mlines

    @property
    def mlines(self):
        """
        The mlines column of a MultiLine object contains the positions of
        each sub-geometry. Each pair of values in the mlines column specifies
        the beginning index of a sub-geometry in the `.xy` column and beginning
        of the subsequent sub-geometry. Contains `2n` values where `n` is the
        number of sub-geometries.
        """
        return self._mlines

    @mlines.setter
    def mlines(self, mlines):
        self._mlines = self._serialize(mlines)

    def __repr__(self):
        return (
            f"{super().__repr__()}" f"mlines:\n" f"{self.mlines.__repr__()}\n"
        )

    def copy(self, deep=True):
        base = super().copy(deep)
        result = LineArray(
            base.xy,
            base.offsets,
            self.mlines.copy(deep),
            base.z,
        )
        return result

    def __len__(self):
        if len(self._mlines) > 0:
            mlength = (
                self._mlines.values[
                    np.arange(
                        1, len(self._mlines), 2, like=self._mlines.values
                    )
                ]
                - self._mlines.values[
                    np.arange(
                        0, len(self._mlines), 2, like=self._mlines.values
                    )
                ]
            ).sum() - (len(self._mlines) // 2)
        else:
            mlength = 0
        return (len(self.offsets) - 1) - int(mlength)


class MultiPointArray(OffsetArray):
    def __init__(self, xy, offsets, z=None, data_locale=cudf):
        """
        A GeoArrow column of MultiPoints. These are all of the MultiPoints that
        appear in a GeoSeries or other data source. Single points are stored in
        the CoordinateArray.
        """
        super().__init__(xy, z, data_locale=data_locale)
        self.offsets = offsets

    def copy(self, deep=True):
        base = super().copy(deep)
        result = MultiPointArray(base.xy, base.offsets, base.z)
        return result


class PolygonArray(OffsetArray):
    def __init__(self, xy, polys, rings, mpolys, z=None, data_locale=cudf):
        """
        The GeoArrow column format for Polygons uses the same scheme as the
        format for LineStrings - MultiPolygons and Polygons from the same
        GeoSeries (or another data source) are stored in the same contiguous
        buffer. Rings are stored in the offsets array, exterior/interior
        polygons are stored in the `polys` array in shapefile format, and the
        `mpolys` array of pairs determines which polys are members of
        MultiPolygons.
        """
        # PolygonArray uses the offsets buffer for rings!
        super().__init__(xy, rings, z, data_locale=data_locale)
        self.polys = polys
        self.mpolys = mpolys

    @property
    def rings(self):
        """
        Each polygon in the GeoSeries is specified in the rings buffer, which
        follows the same format as `offsets` in OffsetArray.
        """
        return self.offsets

    @rings.setter
    def rings(self, rings):
        self.offsets = self._serialize(rings)

    @property
    def polys(self):
        """
        Polygons are specified according to the shapefile format. The first
        ring in any polygon is its exterior ring, all subsequent rings are
        interior rings. The `polys` column specifies which rings comprise each
        polygon in the GeoSeries.
        """
        return self._polys

    @polys.setter
    def polys(self, polys):
        self._polys = self._serialize(polys)

    @property
    def mpolys(self):
        """
        Polygons that fall within bounds constrained by each pair of values in
        this column are members of a MultiPolygon. Each pair of values
        specifies a single MultiPolygon. All MultiPolygons in the GeoSeries
        are contained here.
        """
        return self._mpolys

    @mpolys.setter
    def mpolys(self, mpolys):
        self._mpolys = self._serialize(mpolys)

    def __repr__(self):
        result = (
            f"{super().__repr__()}"
            f"polys:\n"
            f"{self.polys}\n"
            f"rings:\n"
            f"{self.rings}\n"
            f"mpolys:\n"
            f"{self.mpolys}\n"
        )
        return result

    def copy(self, deep=True):
        """
        See CoordinateArray
        """
        result = super().copy(deep)
        result.polys = self.polys.copy(deep)
        result.rings = self.rings.copy(deep)
        result.mpolys = self.mpolys.copy(deep)
        return result

    def __len__(self):
        if len(self._mpolys) > 0:
            mlength = (
                self._mpolys.values[
                    np.arange(
                        1, len(self._mpolys), 2, like=self._mpolys.values
                    )
                ]
                - self._mpolys.values[
                    np.arange(
                        0, len(self._mpolys), 2, like=self._mpolys.values
                    )
                ]
            ).sum() - (len(self._mpolys) // 2)
        else:
            mlength = 0
        return (len(self.polys) - 1) - int(mlength)
