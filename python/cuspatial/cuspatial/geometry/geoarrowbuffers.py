# Copyright (c) 2021 NVIDIA CORPORATION

from typing import TypeVar, Union

import cudf
import numpy as np

T = TypeVar("T", bound="GeoArrowBuffers")


class GeoArrowBuffers:
    """A GPU GeoArrowBuffers object.

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
    implements numpy's `__array__` protocol.

    Notes
    -----
    Legacy cuspatial algorithms depend on separated x and y columns. Access
    them with the `.x` and `.y` properties.

    Examples
    --------
    GeoArrowBuffers accept a dict as argument:

    >>> buffers = GeoArrowBuffers({

    >>> buffers = GeoArrowBuffers({
            "points_xy": [0, 1, 2, 3],
            "multipoints_xy": [0, 1, 2, 3],
            "multipoints_offsets": [0, 1, 2],
            "lines_xy": [0, 1, 2, 3],
            "lines_offsets": [0, 1, 2],
            "mlines": [1, 3],
            "polygons_xy": [0, 1, 2, 3],
            "polygons_polygonsets": [0, 1, 2],
            "polygons_rings": [0, 1, 2],
            "mpolygons": [1, 3],
        })
    """

    def __init__(self, data: Union[dict, T]):
        """
        Parameters
        ----------
        data : A dict or a GeoArrowBuffers object.
        """
        if isinstance(data, dict):
            if data.get("points_xy") is not None:
                self._points = GpuCoordinateArray(data["points_xy"])
            if data.get("mpoints_xy") is not None:
                if data.get("mpoints_offsets") is None:
                    raise ValueError("mpoints must include offsets buffer")
                self._multipoints = GpuMultiPointArray(
                    data["mpoints_xy"], data["mpoints_offsets"]
                )
            if data.get("lines_xy") is not None:
                if data.get("lines_offsets") is None:
                    raise ValueError("lines_xy must include offsets buffer")
                self._lines = GpuLineArray(
                    data["lines_xy"], data["lines_offsets"], data.get("mlines")
                )
            if data.get("polygons_xy") is not None:
                if (
                    data.get("polygons_polygons") is None
                    or data.get("polygons_rings") is None
                ):
                    raise ValueError(
                        "polygons_xy must include polygons and" "rings buffers"
                    )
                self._polygons = GpuPolygonArray(
                    data["polygons_xy"],
                    data["polygons_polygons"],
                    data["polygons_rings"],
                    data.get("mpolygons"),
                )
        elif isinstance(data, GeoArrowBuffers):
            if hasattr(data, "_points"):
                self._points = data._points.copy()
            if hasattr(data, "_multipoints"):
                self._multipoints = data._multipoints.copy()
            if hasattr(data, "_lines"):
                self._lines = data._lines.copy()
            if hasattr(data, "_polygons"):
                self._polygons = data._polygons.copy()
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
        points_length = len(self._points) if hasattr(self, "_points") else 0
        lines_length = len(self._lines) if hasattr(self, "_lines") else 0
        multipoints_length = (
            len(self._multipoints) if hasattr(self, "_multipoints") else 0
        )
        polygons_length = (
            len(self._polygons) if hasattr(self, "_polygons") else 0
        )
        return (
            points_length + lines_length + multipoints_length + polygons_length
        )

    def copy(self, deep=True):
        """
        Create a copy of all of the GPU-backed data structures in this
        GeoArrowBuffers.
        """
        result = GeoArrowBuffers(self)
        return result


class GpuCoordinateArray:
    def __init__(self, xy, z=None):
        """
        A GeoArrow column of points. The GpuCoordinateArray stores all of the
        points within a single data source, typically a cuspatial.GeoSeries,
        in the format specified by GeoArrow.
        """
        self.xy = xy
        if z is not None:
            self.z = z

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
        try:
            self._xy = cudf.Series(xy)
        except Exception as e:
            raise TypeError(e, "xy isn't a buffer type supported by `cudf`")

    @property
    def z(self):
        """
        An optional third dimension for this Geometry.
        """
        if hasattr(self, "_z"):
            return self._z
        else:
            return None

    @z.setter
    def z(self, z):
        try:
            self._z = cudf.Series(z)
        except Exception as e:
            raise TypeError(e, "z isn't a buffer type supported by `cudf`")

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
        if hasattr(self, "_z"):
            z = self.z.copy(deep)
        else:
            z = None
        result = GpuCoordinateArray(self.xy.copy(deep), z)
        return result

    @property
    def x(self):
        """
        Return packed x-coordinates of this GpuGeometryArray object.
        """
        return self.xy[slice(0, None, 2)].reset_index(drop=True)

    @property
    def y(self):
        """
        Return packed y-coordinates of this GpuGeometryArray object.
        """
        return self.xy[slice(1, None, 2)].reset_index(drop=True)

    def __len__(self):
        return len(self.xy) // 2


class GpuPointsArray(GpuCoordinateArray):
    """
    A GeoArrow column of points. Every pair is the `[x,y]` coordinate of the
    position/2th point in this data source. `z` can be included optionally.
    """

    def __init__(self, xy, z=None):
        super().__init__(xy, z)


class GpuOffsetArray(GpuCoordinateArray):
    def __init__(self, xy, offsets, z=None):
        """
        A GeoArrow column of offset geometries. This is the base class of all
        complex GeoArrow geometries. MultiLineStrings and MultiPolygons store
        extra metadata to identify individual geometry boundaries, but are also
        based on GpuOffsetArray.
        """
        super().__init__(xy, z)
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
        try:
            self._offsets = cudf.Series(offsets)
        except Exception as e:
            raise TypeError(
                e, "offsets isn't a buffer type supported by `cudf`"
            )

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
        if hasattr(self, "_z"):
            z = self.z.copy(deep)
        else:
            z = None
        result = GpuOffsetArray(self.xy.copy(deep), self.offsets.copy(deep), z)
        return result

    def __len__(self):
        return len(self._offsets) - 1


class GpuLineArray(GpuOffsetArray):
    def __init__(self, xy, lines, mlines, z=None):
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
        super().__init__(xy, lines, z)
        self.mlines = mlines

    @property
    def mlines(self):
        """
        The mlines column of a MultiLine object contains the positions of
        each sub-geometry. Each pair of values in the mlines column specifies
        the beginning index of a sub-geometry in the `.xy` column and beginning
        of the subsequent sub-geometry. Contains `n+1` values where `n` is the
        number of sub-geometries.
        """
        return self._mlines

    @mlines.setter
    def mlines(self, mlines):
        try:
            self._mlines = cudf.Series(mlines)
        except Exception as e:
            raise TypeError(
                e, "mlines isn't a buffer type supported by `cudf`"
            )

    def __repr__(self):
        return (
            f"{super().__repr__()}" f"mlines:\n" f"{self.mlines.__repr__()}\n"
        )

    def copy(self, deep=True):
        base = super().copy(deep)
        result = GpuLineArray(
            base.xy, base.offsets, self.mlines.copy(), base.z,
        )
        return result

    def __len__(self):
        mlength = np.array(
            [
                self._mlines[2 * i - 1] - self._mlines[2 * i - 2] - 1
                for i in range(1, len(self._mlines) // 2 + 1)
            ],
            dtype="int64",
        ).sum()
        return (len(self.offsets) - 1) - mlength


class GpuMultiPointArray(GpuOffsetArray):
    def __init__(self, xy, offsets, z=None):
        """
        A GeoArrow column of MultiPoints. These are all of the MultiPoints that
        appear in a GeoSeries or other data source. Single points are stored in
        the GpuCoordinateArray.
        """
        super().__init__(xy, z)
        self.offsets = offsets

    def copy(self, deep=True):
        base = super().copy(deep)
        result = GpuMultiPointArray(base.xy, base.offsets, base.z)
        return result


class GpuPolygonArray(GpuOffsetArray):
    def __init__(self, xy, polys, rings, mpolys, z=None):
        """
        The GeoArrow column format for GpuPolygons uses the same scheme as the
        format for LineStrings - MultiPolygons and Polygons from the same
        GeoSeries (or another data source) are stored in the same contiguous
        buffer. Rings are stored in the offsets array, exterior/interior
        polygons are stored in the `polys` array in shapefile format, and the
        `mpolys` array of pairs determines which polys are members of
        MultiPolygons.
        """
        # GpuPolygonArray uses the offsets buffer for rings!
        super().__init__(xy, rings, z)
        self.polys = polys
        self.mpolys = mpolys

    @property
    def rings(self):
        """
        Each polygon in the GeoSeries is specified in the rings buffer, which
        follows the same format as `offsets` in GpuOffsetArray.
        """
        return self.offsets

    @rings.setter
    def rings(self, rings):
        try:
            self.offsets = cudf.Series(rings)
        except Exception as e:
            raise TypeError(e, "rings isn't a buffer type supported by `cudf`")

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
        try:
            self._polys = cudf.Series(polys)
        except Exception as e:
            raise TypeError(
                e, "polys  isn't a buffer type supported by `cudf`"
            )

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
        try:
            self._mpolys = cudf.Series(mpolys)
        except Exception as e:
            raise TypeError(
                e, "mpolys  isn't a buffer type supported by `cudf`"
            )

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
        See GpuCoordinateArray
        """
        result = super().copy(deep)
        result.polys = self.polys.copy(deep)
        result.rings = self.rings.copy(deep)
        result.mpolys = self.mpolys.copy()
        return result

    def __len__(self):
        mlength = np.array(
            [
                self._mpolys[2 * i - 1] - self._mpolys[2 * i - 2] - 1
                for i in range(1, len(self._mpolys) // 2 + 1)
            ],
            dtype="int64",
        ).sum()
        return (len(self.polys) - 1) - mlength
