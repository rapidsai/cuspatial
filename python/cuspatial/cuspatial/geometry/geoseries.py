# Copyright (c) 2020-2021, NVIDIA CORPORATION

import numbers
import pandas as pd

from geopandas.geoseries import GeoSeries as gpGeoSeries
from shapely.geometry import (
    Point,
    MultiPoint,
    LineString,
    MultiLineString,
    Polygon,
    MultiPolygon,
)

import cudf
from cudf.core.column import ColumnBase

from cuspatial.io.geoseries_reader import GeoSeriesReader


class GeoSeries(cudf.Series):
    """
    A wrapper for Series functionality that needs to be subclassed in order
    to support our implementation of GeoColumn.
    """

    def __init__(
        self, data=None, index=None, dtype=None, name=None, nan_as_null=True
    ):
        if isinstance(data, pd.Series):
            data = gpGeoSeries(data)
        if isinstance(data, (GeoColumn, gpGeoSeries, GeoSeries, dict)):
            super().__init__(
                cudf.RangeIndex(0, len(data)), index, dtype, name, nan_as_null
            )
            if isinstance(data, GeoColumn):
                self.geocolumn = data
            else:
                self.geocolumn = GeoColumn(data)
        else:
            raise TypeError(f"Incompatible object passed to GeoSeries ctor {type(data)}")

    @property
    def geocolumn(self):
        return self._geocolumn

    @geocolumn.setter
    def geocolumn(self, value):
        if not isinstance(value, GeoColumn):
            raise TypeError
        self._geocolumn = value
        self._column = cudf.RangeIndex(0, len(value))

    @property
    def points(self):
        """
        The Points column is a simple numeric column. x and y coordinates
        can be stored either interleaved or in separate columns. If a z
        coordinate is present, it will be stored in a separate column.
        """
        return self.geocolumn.points

    @property
    def multipoints(self):
        """
        The MultiPoints column is similar to the Points column with the
        addition of an offsets column. The offsets column stores the comparable
        sizes and coordinates of each MultiPoint in the cuGeoSeries.
        """
        return self.geocolumn.multipoints

    @property
    def lines(self):
        """
        LineStrings contain the coordinates column, an offsets column, and a
        multioffsets column. The multioffsets column stores the indices of the
        offsets that indicate the beginning and end of each MultiLineString
        segment.
        """
        return self.geocolumn.lines

    @property
    def polygons(self):
        """
        Polygons contain the coordinates column, a rings column specifying
        the beginning and end of every polygon, a polygons column specifying
        the beginning, or exterior, ring of each polygon and the end ring.
        All rings after the first ring are interior rings.  Finally a
        multipolys column stores the offsets of the polygons that should be
        grouped into MultiPolygons.
        """
        return self.geocolumn.polygons
    
    @property
    def index(self):
        """
        A cudf.Index object. A mapping of row-labels.
        """
        return self._index

    @index.setter
    def index(self, index):
        self._index = index

    def to_geopandas(self):
        return self.geocolumn.to_geopandas()

    def to_pandas(self):
        """
        Treats to_pandas and to_geopandas as the same call, which improves
        compatibility with pandas.
        """
        return self.to_geopandas()

    def __getitem__(self, key):
        return self.geocolumn[self._column[key]]

    def __repr__(self):
        # TODO: Limit the the number of rows like cudf does
        return self.to_pandas().__repr__()
    
    def to_geopandas(self, index=None, nullable=False):
        """
        Returns a new GeoPandas GeoSeries object from the coordinates in
        the cuspatial GeoSeries.
        """
        if nullable is True:
            raise ValueError("cuGeoSeries doesn't support N/A yet")
        if index is None:
            index = self.index
        if isinstance(index, cudf.Index):
            index = index.to_pandas()
        output = [geom.to_shapely() for geom in self.geocolumn]
        return gpGeoSeries(output, index=index)

    def to_pandas(self, index=None, nullable=False):
        """
        Treats to_pandas and to_geopandas as the same call, which improves
        compatibility with pandas.
        """
        return self.to_geopandas(index=index, nullable=nullable)


class GeoColumn(ColumnBase):
    """
    A GPU GeoColumn object.

    The GeoArrow format specifies a tabular data format for geometry
    information. Supported types include `Point`, `MultiPoint`, `LineString`,
    `MultiLineString`, `Polygon`, and `MultiPolygon`.  In order to store
    these coordinate types in a strictly tabular fashion, columns are
    created for Points, MultiPoints, LineStrings, and Polygons.
    MultiLines and MultiPolygons are stored in the same data structure
    as LineStrings and Polygons.

    Parameters
    ----------
    data : A GeoPandas GeoSeries object, or a file path.
    name : String (optional), the name of the cudf.Series object.
    index : cudf.Index (optional), row labels for the cudf.Series object.

    Notes
    -----
    Legacy cuspatial algorithms depend on separated x and y columns. Access
    them with the `.x` and `.y` properties.
    """

    def __init__(self, *args, **kwargs):
        data = args[0]
        if isinstance(data, pd.Series):
            data = gpGeoSeries(data)
        elif isinstance(data, cudf.core.column_accessor.ColumnAccessor):
            data = data[data.name]
        elif isinstance(data, GeoSeries):
            data = data.geocolumn
        if isinstance(data, GeoColumn):
            self._data = data._data.copy()
            self._points = data.points.copy()
            self._multipoints = data.multipoints.copy()
            self._lines = data.lines.copy()
            self._polygons = data.polygons.copy()
            self.types = data.types.copy()
            self.lengths = data.lengths.copy()
        elif isinstance(data, gpGeoSeries):
            self._data = data
            self._reader = GeoSeriesReader(data)
            self._points = GpuCoordinateArray(
                self._reader.buffers[0]["points"]
            )
            self._multipoints = GpuMultiPointArray(
                self._reader.buffers[0]["multipoints"],
                self._reader.buffers[1]["multipoints"],
            )
            self._lines = GpuLineArray(
                self._reader.buffers[0]["lines"],
                self._reader.buffers[1]["lines"],
                self._reader.buffers[1]["mlines"],
            )
            self._polygons = GpuPolygonArray(
                self._reader.buffers[0]["polygons"]["coords"],
                self._reader.offsets["polygons"]["polygons"],
                self._reader.offsets["polygons"]["rings"],
                self._reader.offsets["polygons"]["mpolys"],
            )
            self._types = self._reader.buffers[2]
            self._lengths = self._reader.buffers[3]
        else:
            raise TypeError(
                f"Invalid type passed to GeoColumn ctor {type(data)}"
            )

    @property
    def types(self):
        """
        A list of string types from the set "p", "mp", "l", "ml", "poly", and
        "mpoly". These are the types of each row of the GeoSeries.
        """
        return self._types

    @types.setter
    def types(self, types):
        self._types = types

    @property
    def lengths(self):
        """
        A list of integers of the length of each Multi geometry. Each non-multi
        geometry is length 1.
        """
        return self._lengths

    @lengths.setter
    def lengths(self, lengths):
        self._lengths = lengths

    def __getitem__(self, item):
        """
        Returns gpuGeometry objects for each of the rows specified by index.
        """
        if not isinstance(item, numbers.Integral):
            raise NotImplementedError
        return self.iloc[item]

    @property
    def loc(self):
        """
        Not currently supported.
        """
        return self.GeoColumnLocIndexer(self)

    @property
    def iloc(self):
        """
        Return the i-th row of the GeoSeries.
        """
        return self.GeoColumnILocIndexer(self)

    def __len__(self):
        """
        Returns the number of unique geometries stored in this cuGeoSeries.
        """
        return int(len(self.types))

    @property
    def points(self):
        """
        The Points column is a simple numeric column. x and y coordinates
        can be stored either interleaved or in separate columns. If a z
        coordinate is present, it will be stored in a separate column.
        """
        return self._points

    @property
    def multipoints(self):
        """
        The MultiPoints column is similar to the Points column with the
        addition of an offsets column. The offsets column stores the comparable
        sizes and coordinates of each MultiPoint in the cuGeoSeries.
        """
        return self._multipoints

    @property
    def lines(self):
        """
        LineStrings contain the coordinates column, an offsets column, and a
        multioffsets column. The multioffsets column stores the indices of the
        offsets that indicate the beginning and end of each MultiLineString
        segment.
        """
        return self._lines

    @property
    def polygons(self):
        """
        Polygons contain the coordinates column, a rings column specifying
        the beginning and end of every polygon, a polygons column specifying
        the beginning, or exterior, ring of each polygon and the end ring.
        All rings after the first ring are interior rings.  Finally a
        multipolys column stores the offsets of the polygons that should be
        grouped into MultiPolygons.
        """
        return self._polygons

    def _dump(self):
        return (
            f"POINTS\n"
            f"{self.points.__repr__()}\n"
            f"MULTIPOINTS\n"
            f"{self.multipoints.__repr__()}\n"
            f"LINES\n"
            f"{self.lines.__repr__()}\n"
            f"POLYGONS\n"
            f"{self.polygons.__repr__()}\n"
        )

    def copy(self, deep=True):
        """
        Create a copy of all of the GPU-backed data structures in this
        GeoColumn.
        """
        result = GeoColumn(
            self._data,
            self.points.copy(deep),
            self.multipoints.copy(deep),
            self.lines.copy(deep),
            self.polygons.copy(deep),
            self.types,
            self.lengths,
        )
        return result

    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self):
        if self._iter_index >= len(self):
            raise StopIteration
        result = self[self._iter_index]
        self._iter_index = self._iter_index + 1
        return result

    class GeoColumnLocIndexer:
        """
        Not yet supported.
        """

        def __init__(self):
            # Todo: Easy to implement with a join.
            raise NotImplementedError

    class GeoColumnILocIndexer:
        """
        Each row of a GeoSeries is one of the six types: Point, MultiPoint,
        LineString, MultiLineString, Polygon, or MultiPolygon.
        """

        def __init__(self, sr):
            self._sr = sr

        def __getitem__(self, index):
            if not isinstance(index, slice):
                return self._getitem_int(index)
            else:
                raise NotImplementedError
                # This slice functionality is not ready yet
                # return self._getitem_slice(index)

        def _getitem_int(self, index):
            type_map = {
                "p": gpuPoint,
                "mp": gpuMultiPoint,
                "l": gpuLineString,
                "ml": gpuMultiLineString,
                "poly": gpuPolygon,
                "mpoly": gpuMultiPolygon,
            }
            return type_map[self._sr.types[index]](self._sr, index)


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
        self._xy = cudf.Series(xy)

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
        self._z = cudf.Series(z)

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
        number of sub-geometries. GpuPoints
        """
        return self._offsets

    @offsets.setter
    def offsets(self, offsets):
        self._offsets = cudf.Series(offsets)

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
        """
        See GpuCoordinateArray.
        """
        if hasattr(self, "_z"):
            z = self.z.copy(deep)
        else:
            z = None
        result = GpuOffsetArray(self.xy.copy(deep), self.offsets.copy(deep), z)
        return result


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
        self._mlines = cudf.Series(mlines)

    def __getitem__(self, index):
        result = super().__getitem__(index)
        return result

    def __repr__(self):
        return (
            f"{super().__repr__()}" f"mlines:\n" f"{self.mlines.__repr__()}\n"
        )

    def copy(self, deep=True):
        """
        See GpuCoordinateArray.
        """
        base = super().copy(deep)
        result = GpuLineArray(
            base.xy, base.offsets, self.mlines.copy(), base.z,
        )
        return result


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
        self.offsets = cudf.Series(rings)

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
        self._polys = cudf.Series(polys)

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
        self._mpolys = cudf.Series(mpolys)

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


class gpuGeometry:
    def __init__(self, source, index):
        """
        The base class of individual GPU geometries. This and its inheriting
        classes do not manage any GPU data directly - each gpuGeometry simply
        stores a reference to the GeoSeries it is stored within and the index
        of the geometry within the GeoSeries. Child gpuGeometry classes
        contain the logic necessary to serialize and convert GPU data back to
        Shapely. """
        self._source = source
        self._index = index


class gpuPoint(gpuGeometry):
    """
    See gpuGeometry.
    """

    def to_shapely(self):
        item_type = self._source.types[self._index]
        types = self._source.types[0 : self._index]
        index = 0
        for i in range(self._index):
            if types[i] == item_type:
                index = index + 1
        return Point(self._source._points[index].reset_index(drop=True))


class gpuMultiPoint(gpuGeometry):
    """
    See gpuGeometry.
    """

    def to_shapely(self):
        item_type = self._source.types[self._index]
        types = self._source.types[0 : self._index]
        item_start = 0
        for i in range(self._index):
            if types[i] == item_type:
                item_start = item_start + 1
        item_length = self._source.lengths[self._index]
        item_source = self._source._multipoints
        result = item_source[item_start]
        return MultiPoint(result.to_array().reshape(item_length, 2))


class gpuLineString(gpuGeometry):
    """
    See gpuGeometry.
    """

    def to_shapely(self):
        ml_index = self._index - 1
        preceding_line_count = 0
        preceding_ml_count = 0
        while ml_index >= 0:
            if self._source.types[ml_index] == "ml":
                preceding_ml_count = preceding_ml_count + 1
            elif (
                self._source.types[ml_index] == "l" and preceding_ml_count == 0
            ):
                preceding_line_count = preceding_line_count + 1
            ml_index = ml_index - 1
        preceding_multis = preceding_ml_count
        if preceding_multis > 0:
            multi_end = self._source._lines.mlines[preceding_multis * 2 - 1]
            item_start = multi_end + preceding_line_count
        else:
            item_start = preceding_line_count
        item_length = self._source.lengths[self._index]
        item_end = item_length + item_start
        item_source = self._source._lines
        result = item_source[item_start:item_end]
        return LineString(
            result.to_array().reshape(2 * (item_start - item_end), 2)
        )


class gpuMultiLineString(gpuGeometry):
    """
    See gpuGeometry.
    """

    def to_shapely(self):
        item_type = self._source.types[self._index]
        index = 0
        for i in range(self._index):
            if self._source.types[i] == item_type:
                index = index + 1
        line_indices = slice(
            self._source._lines.mlines[index * 2],
            self._source._lines.mlines[index * 2 + 1],
        )
        return MultiLineString(
            [
                LineString(
                    self._source._lines[i]
                    .to_array()
                    .reshape(int(len(self._source._lines[i]) / 2), 2)
                )
                for i in range(line_indices.start, line_indices.stop, 1)
            ]
        )


class gpuPolygon(gpuGeometry):
    """
    See gpuGeometry.
    """

    def to_shapely(self):
        mp_index = self._index - 1
        preceding_poly_count = 0
        preceding_mp_count = 0
        while mp_index >= 0:
            if self._source.types[mp_index] == "mpoly":
                preceding_mp_count = preceding_mp_count + 1
            elif (
                self._source.types[mp_index] == "poly"
                and preceding_mp_count == 0
            ):
                preceding_poly_count = preceding_poly_count + 1
            mp_index = mp_index - 1
        preceding_multis = preceding_mp_count
        multi_index = (
            self._source._polygons.mpolys[preceding_multis * 2 - 1]
            if preceding_multis > 0
            else 0
        )
        preceding_polys = preceding_poly_count
        ring_start = self._source._polygons.polys[
            multi_index + preceding_polys
        ]
        ring_end = self._source._polygons.polys[
            multi_index + preceding_polys + 1
        ]
        rings = self._source._polygons.rings
        exterior_slice = slice(rings[ring_start], rings[ring_start + 1])
        exterior = self._source._polygons.xy[exterior_slice]
        return Polygon(
            exterior.to_array().reshape(2 * (ring_start - ring_end), 2),
            [
                self._source._polygons.xy[interior_slice]
                .to_array()
                .reshape(
                    int((interior_slice.stop - interior_slice.start + 1) / 2),
                    2,
                )
                for interior_slice in [
                    slice(rings[interior], rings[interior + 1])
                    for interior in range(ring_end - (ring_start + 1))
                    + (ring_start + 1)
                ]
            ],
        )


class gpuMultiPolygon(gpuGeometry):
    """
    See gpuGeometry.
    """

    def to_shapely(self):
        item_type = self._source.types[self._index]
        index = 0
        for i in range(self._index):
            if self._source.types[i] == item_type:
                index = index + 1
        poly_indices = slice(
            self._source.polygons.mpolys[index * 2],
            self._source.polygons.mpolys[index * 2 + 1],
        )
        polys = []
        for i in range(poly_indices.start, poly_indices.stop):
            ring_start = self._source._polygons.polys[i]
            ring_end = self._source._polygons.polys[i + 1]
            rings = self._source._polygons.rings
            exterior_slice = slice(rings[ring_start], rings[ring_start + 1])
            exterior = self._source._polygons.xy[exterior_slice]
            polys.append(
                Polygon(
                    exterior.to_array().reshape(
                        2 * (ring_start - ring_end), 2
                    ),
                    [
                        self._source._polygons.xy[interior_slice]
                        .to_array()
                        .reshape(
                            int(
                                (
                                    interior_slice.stop
                                    - interior_slice.start
                                    + 1
                                )
                                / 2
                            ),
                            2,
                        )
                        for interior_slice in [
                            slice(rings[interior], rings[interior + 1])
                            for interior in range(ring_start + 1, ring_end)
                        ]
                    ],
                )
            )
        return MultiPolygon(polys)
