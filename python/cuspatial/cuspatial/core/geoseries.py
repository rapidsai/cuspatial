# Copyright (c) 2020-2022, NVIDIA CORPORATION

from functools import cached_property
from numbers import Integral
from typing import Optional, Tuple, TypeVar, Union

import cupy as cp
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
from geopandas.geoseries import GeoSeries as gpGeoSeries
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

import cudf
from cudf._typing import ColumnLike

import cuspatial.io.pygeoarrow as pygeoarrow
from cuspatial.core._column.geocolumn import GeoColumn
from cuspatial.core._column.geometa import Feature_Enum, GeoMeta
from cuspatial.core.binops.contains import contains_properly
from cuspatial.utils.column_utils import (
    contains_only_linestrings,
    contains_only_multipoints,
    contains_only_polygons,
)

T = TypeVar("T", bound="GeoSeries")


class GeoSeries(cudf.Series):
    """
    cuspatial.GeoSeries enables GPU-backed storage and computation of
    shapely-like objects. Our goal is to give feature parity with GeoPandas.
    At this time, only from_geopandas and to_geopandas are directly supported.
    cuspatial GIS, indexing, and trajectory functions depend on the arrays
    stored in the `GeoArrowBuffers` object, accessible with the `points`,
    `multipoints`, `lines`, and `polygons` accessors.

    Examples
    --------

    >>> from shapely.geometry import Point
        import geopandas
        import cuspatial
        cuseries = cuspatial.GeoSeries(geopandas.GeoSeries(Point(-1, 0)))
        cuseries.points.xy
        0   -1.0
        1    0.0
        dtype: float64
    """

    def __init__(
        self,
        data: Optional[
            Union[gpd.GeoSeries, Tuple, T, pd.Series, GeoColumn, list]
        ],
        index: Union[cudf.Index, pd.Index] = None,
        dtype=None,
        name=None,
        nan_as_null=True,
    ):
        # Condition data
        if data is None or isinstance(data, (pd.Series, list)):
            data = gpGeoSeries(data)
        # Create column
        if isinstance(data, GeoColumn):
            column = data
        elif isinstance(data, GeoSeries):
            column = data._column
        elif isinstance(data, gpGeoSeries):
            from cuspatial.io.geopandas_reader import GeoPandasReader

            adapter = GeoPandasReader(data)
            pandas_meta = GeoMeta(adapter.get_geopandas_meta())
            column = GeoColumn(adapter._get_geotuple(), pandas_meta)
        elif isinstance(data, Tuple):
            # This must be a Polygon Tuple returned by
            # cuspatial.read_polygon_shapefile
            # TODO: If an index is passed in, it needs to be reflected
            # in the column, because otherwise .iloc indexing will ignore
            # it.
            column = GeoColumn(
                data,
                GeoMeta(
                    {
                        "input_types": cp.repeat(
                            cp.array(
                                [Feature_Enum.POLYGON.value], dtype="int8"
                            ),
                            len(data[0]),
                        ),
                        "union_offsets": cp.arange(
                            len(data[0]), dtype="int32"
                        ),
                    }
                ),
                from_read_polygon_shapefile=True,
            )

        else:
            raise TypeError(
                f"Incompatible object passed to GeoSeries ctor {type(data)}"
            )
        # Condition index
        if isinstance(data, (gpGeoSeries, GeoSeries)):
            if index is None:
                index = data.index
        if index is None:
            index = cudf.RangeIndex(0, len(column))
        super().__init__(
            column, index, dtype=dtype, name=name, nan_as_null=nan_as_null
        )

    @property
    def type(self):
        gpu_types = cudf.Series(self._column._meta.input_types).astype("str")
        result = gpu_types.replace(
            {
                "0": "Point",
                "1": "MultiPoint",
                "2": "Linestring",
                "3": "MultiLinestring",
                "4": "Polygon",
                "5": "MultiPolygon",
            }
        )
        return result

    class GeoColumnAccessor:
        def __init__(self, list_series, meta):
            self._series = list_series
            self._col = self._series._column
            self._meta = meta
            self._type = Feature_Enum.POINT

        @property
        def x(self):
            return self.xy[::2].reset_index(drop=True)

        @property
        def y(self):
            return self.xy[1::2].reset_index(drop=True)

        @property
        def xy(self):
            return cudf.Series(
                self._get_current_features(self._type).leaves().values
            )

        def _get_current_features(self, type):
            # Resample the existing features so that the offsets returned
            # by `_offset` methods reflect previous slicing, and match
            # the values returned by .xy.
            existing_indices = self._meta.union_offsets[
                self._meta.input_types == type.value
            ]
            existing_features = self._col.take(existing_indices._column)
            return existing_features

        def point_indices(self):
            # Return a cupy.ndarray containing the index values that each
            # point belongs to.
            offsets = cp.arange(0, len(self.xy) + 1, 2)
            sizes = offsets[1:] - offsets[:-1]
            return cp.repeat(self._series.index, sizes)

    class MultiPointGeoColumnAccessor(GeoColumnAccessor):
        def __init__(self, list_series, meta):
            super().__init__(list_series, meta)
            self._type = Feature_Enum.MULTIPOINT

        @property
        def geometry_offset(self):
            return self._get_current_features(self._type).offsets.values

        def point_indices(self):
            # Return a cupy.ndarray containing the index values from the
            # MultiPoint GeoSeries that each individual point is member of.
            offsets = cp.array(self.geometry_offset)
            sizes = offsets[1:] - offsets[:-1]
            return cp.repeat(self._series.index, sizes)

    class LineStringGeoColumnAccessor(GeoColumnAccessor):
        def __init__(self, list_series, meta):
            super().__init__(list_series, meta)
            self._type = Feature_Enum.LINESTRING

        @property
        def geometry_offset(self):
            return self._get_current_features(self._type).offsets.values

        @property
        def part_offset(self):
            return self._get_current_features(
                self._type
            ).elements.offsets.values

        def point_indices(self):
            # Return a cupy.ndarray containing the index values from the
            # LineString GeoSeries that each individual point is member of.
            offsets = cp.array(self.part_offset)
            sizes = offsets[1:] - offsets[:-1]
            return cp.repeat(self._series.index, sizes)

    class PolygonGeoColumnAccessor(GeoColumnAccessor):
        def __init__(self, list_series, meta):
            super().__init__(list_series, meta)
            self._type = Feature_Enum.POLYGON

        @property
        def geometry_offset(self):
            return self._get_current_features(self._type).offsets.values

        @property
        def part_offset(self):
            return self._get_current_features(
                self._type
            ).elements.offsets.values

        @property
        def ring_offset(self):
            return self._get_current_features(
                self._type
            ).elements.elements.offsets.values

        def point_indices(self):
            # Return a cupy.ndarray containing the index values from the
            # Polygon GeoSeries that each individual point is member of.
            offsets = cp.array(self.ring_offset)
            sizes = offsets[1:] - offsets[:-1]
            return cp.repeat(self._series.index, sizes)

    @property
    def points(self):
        """
        Access the `PointsArray` of the underlying `GeoArrowBuffers`.
        """
        return self.GeoColumnAccessor(self._column.points, self._column._meta)

    @property
    def multipoints(self):
        """
        Access the `MultiPointArray` of the underlying `GeoArrowBuffers`.
        """
        return self.MultiPointGeoColumnAccessor(
            self._column.mpoints, self._column._meta
        )

    @property
    def lines(self):
        """
        Access the `LineArray` of the underlying `GeoArrowBuffers`.
        """
        return self.LineStringGeoColumnAccessor(
            self._column.lines, self._column._meta
        )

    @property
    def polygons(self):
        """
        Access the `PolygonArray` of the underlying `GeoArrowBuffers`.
        """
        return self.PolygonGeoColumnAccessor(
            self._column.polygons, self._column._meta
        )

    def __repr__(self):
        # TODO: Implement Iloc with slices so that we can use `Series.__repr__`
        return self.to_pandas().__repr__()

    class GeoSeriesLocIndexer:
        """
        Map the index to an integer Series and use that.
        """

        def __init__(self, _sr):
            self._sr = _sr

        def __getitem__(self, item):
            booltest = cudf.Series(item)
            if booltest.dtype in (bool, np.bool_):
                return self._sr.iloc[item]

            map_df = cudf.DataFrame(
                {
                    "map": self._sr.index,
                    "idx": cp.arange(len(self._sr.index))
                    if not isinstance(item, Integral)
                    else 0,
                }
            )
            index_df = cudf.DataFrame({"map": item}).reset_index()
            new_index = index_df.merge(
                map_df, how="left", sort=False
            ).sort_values("index")
            if isinstance(item, Integral):
                return self._sr.iloc[new_index["idx"][0]]
            else:
                result = self._sr.iloc[new_index["idx"]]
                result.index = item
                return result

    class GeoSeriesILocIndexer:

        """
        Each row of a GeoSeries is one of the six types: Point, MultiPoint,
        LineString, MultiLineString, Polygon, or MultiPolygon.
        """

        def __init__(self, sr):
            self._sr = sr

        @cached_property
        def _type_int_to_field(self):
            return {
                Feature_Enum.POINT: self._sr.points,
                Feature_Enum.MULTIPOINT: self._sr.mpoints,
                Feature_Enum.LINESTRING: self._sr.lines,
                Feature_Enum.POLYGON: self._sr.polygons,
            }

        def __getitem__(self, indexes):
            # Slice the types and offsets
            union_offsets = self._sr._column._meta.union_offsets.iloc[indexes]
            union_types = self._sr._column._meta.input_types.iloc[indexes]

            points = self._sr._column.points
            mpoints = self._sr._column.mpoints
            lines = self._sr._column.lines
            polygons = self._sr._column.polygons

            column = GeoColumn(
                (points, mpoints, lines, polygons),
                {
                    "input_types": union_types,
                    "union_offsets": union_offsets,
                },
            )

            if isinstance(indexes, Integral):
                return GeoSeries(column, name=self._sr.name).to_shapely()
            else:
                return GeoSeries(
                    column, index=self._sr.index[indexes], name=self._sr.name
                )

    def from_arrow(union):
        column = GeoColumn(
            (
                cudf.Series(union.child(0)),
                cudf.Series(union.child(1)),
                cudf.Series(union.child(2)),
                cudf.Series(union.child(3)),
            ),
            {
                "input_types": union.type_codes,
                "union_offsets": union.offsets,
            },
        )
        return GeoSeries(column)

    @property
    def loc(self):
        """
        Not currently supported.
        """
        return self.GeoSeriesLocIndexer(self)

    @property
    def iloc(self):
        """
        Return the i-th row of the GeoSeries.
        """
        return self.GeoSeriesILocIndexer(self)

    def to_geopandas(self, nullable=False):
        """
        Returns a new GeoPandas GeoSeries object from the coordinates in
        the cuspatial GeoSeries.
        """
        if nullable is True:
            raise ValueError("GeoSeries doesn't support <NA> yet")
        final_union_slice = self.iloc[0 : len(self._column)]
        return gpGeoSeries(
            final_union_slice.to_shapely(),
            index=self.index.to_pandas(),
            name=self.name,
        )

    def to_pandas(self):
        """
        Treats to_pandas and to_geopandas as the same call, which improves
        compatibility with pandas.
        """
        return self.to_geopandas()

    def _linestring_to_shapely(self, geom):
        if len(geom) == 1:
            result = [tuple(x) for x in geom[0]]
            return LineString(result)
        else:
            linestrings = []
            for linestring in geom:
                linestrings.append(
                    LineString([tuple(child) for child in linestring])
                )
            return MultiLineString(linestrings)

    def _polygon_to_shapely(self, geom):
        if len(geom) == 1:
            rings = []
            for ring in geom[0]:
                rings.append(tuple(tuple(point) for point in ring))
            return Polygon(rings[0], rings[1:])
        else:
            polygons = []
            for p in geom:
                rings = []
                for ring in p:
                    rings.append(tuple([tuple(point) for point in ring]))
                polygons.append(Polygon(rings[0], rings[1:]))
            return MultiPolygon(polygons)

    @cached_property
    def _arrow_to_shapely(self):
        type_map = {
            Feature_Enum.NONE: lambda x: None,
            Feature_Enum.POINT: Point,
            Feature_Enum.MULTIPOINT: MultiPoint,
            Feature_Enum.LINESTRING: self._linestring_to_shapely,
            Feature_Enum.POLYGON: self._polygon_to_shapely,
        }
        return type_map

    def to_shapely(self):
        # Copy the GPU data to host for iteration and deserialization
        result_types = self._column._meta.input_types.to_arrow()

        # Get the shapely serialization methods we'll use here.
        shapely_fns = [
            self._arrow_to_shapely[Feature_Enum(x)]
            for x in result_types.to_numpy()
        ]
        union = self.to_arrow()

        # Serialize to Shapely!
        results = []
        for (result_index, shapely_serialization_fn) in zip(
            range(0, len(self)), shapely_fns
        ):
            results.append(
                shapely_serialization_fn(union[result_index].as_py())
            )

        # Finally, a slice determines that we return a list, otherwise
        # an object.
        if len(results) == 1:
            return results[0]
        else:
            return results

    def to_arrow(self):
        """
        Convert to a GeoArrow Array.

        Returns
        -------
        result: GeoArrow Union containing GeoArrow Arrays

        Examples
        --------
        >>> from shapely.geometry import MultiLineString, LineString
        >>> cugpdf = cuspatial.from_geopandas(geopandas.GeoSeries(
            MultiLineString(
                [
                    [(1, 0), (0, 1)],
                    [(0, 0), (1, 1)]
                ]
            )))
        >>> cugpdf.to_arrow()
        <pyarrow.lib.UnionArray object at 0x7f7061c0e0a0>
        -- is_valid: all not null
        -- type_ids:   [
            2
          ]
        -- value_offsets:   [
            0
          ]
        -- child 0 type: list<item: null>
          []
        -- child 1 type: list<item: null>
          []
        -- child 2 type: list<item: list<item: list<item: double>>>
          [
            [
              [
                [
                  1,
                  0
                ],
                [
                  0,
                  1
                ]
              ],
              [
                [
                  0,
                  0
                ],
                [
                  1,
                  1
                ]
              ]
            ]
          ]
        -- child 3 type: list<item: null>
          []
        """

        points = self._column.points
        mpoints = self._column.mpoints
        lines = self._column.lines
        polygons = self._column.polygons
        arrow_points = (
            points.to_arrow().view(pygeoarrow.ArrowPointsType)
            if len(points) > 0
            else points.to_arrow()
        )
        arrow_mpoints = (
            mpoints.to_arrow().view(pygeoarrow.ArrowMultiPointsType)
            if len(mpoints) > 1
            else mpoints.to_arrow()
        )
        arrow_lines = (
            lines.to_arrow().view(pygeoarrow.ArrowLinestringsType)
            if len(lines) > 1
            else lines.to_arrow()
        )
        arrow_polygons = (
            polygons.to_arrow().view(pygeoarrow.ArrowPolygonsType)
            if len(polygons) > 1
            else polygons.to_arrow()
        )

        return pa.UnionArray.from_dense(
            self._column._meta.input_types.to_arrow(),
            self._column._meta.union_offsets.to_arrow(),
            [
                arrow_points,
                arrow_mpoints,
                arrow_lines,
                arrow_polygons,
            ],
        )

    def _align_to_index(
        self: T,
        index: ColumnLike,
        how: str = "outer",
        sort: bool = True,
        allow_non_unique: bool = False,
    ) -> T:
        """
        The values in the newly aligned columns will not change,
        only positions in the union offsets and type codes.
        """
        aligned_union_offsets = (
            self._column._meta.union_offsets._align_to_index(
                index, how, sort, allow_non_unique
            )
        ).astype("int32")
        aligned_union_offsets[
            aligned_union_offsets.isna()
        ] = Feature_Enum.NONE.value
        aligned_input_types = self._column._meta.input_types._align_to_index(
            index, how, sort, allow_non_unique
        ).astype("int8")
        aligned_input_types[
            aligned_input_types.isna()
        ] = Feature_Enum.NONE.value
        column = GeoColumn(
            (
                self._column.points,
                self._column.mpoints,
                self._column.lines,
                self._column.polygons,
            ),
            {
                "input_types": aligned_input_types,
                "union_offsets": aligned_union_offsets,
            },
        )
        return GeoSeries(column)

    def align(self, other):
        """
        Align the rows of two GeoSeries using outer join.

        `align` rearranges two GeoSeries so that their indices match.
        If one GeoSeries is longer than the other, the shorter GeoSeries
        will be increased in length and missing index values will be added,
        inserting `None` when an empty row is created.

        Alignment involves matching the length of the indices, sorting them,
        and inserting into the right GeoSeries extra index values that are
        present in the left GeoSeries.

        Parameters
        ----------
        other: GeoSeries

        Returns
        -------
        (left, right) : GeoSeries
            Pair of aligned GeoSeries

        Examples
        --------
        >>> points = gpd.GeoSeries([
            Point((-8, -8)),
            Point((-2, -2)),
        ])
        >>> point = gpd.GeoSeries(points[0])
        >>> print(points.align(point))

        (0    POINT (-8.00000 -8.00000)
        1    POINT (-2.00000 -2.00000)
        dtype: geometry, 0    POINT (-8.00000 -8.00000)
        1                         None
        dtype: geometry)

        >>> points_right = gpd.GeoSeries([
            Point((-2, -2)),
            Point((-8, -8)),
        ], index=[1,0])
        >>> print(points.align(points_right))

        (0    POINT (-8.00000 -8.00000)
        1    POINT (-2.00000 -2.00000)
        dtype: geometry, 0    POINT (-8.00000 -8.00000)
        1    POINT (-2.00000 -2.00000)
        dtype: geometry)

        >>> points_alpha = gpd.GeoSeries([
            Point((-1, 1)),
            Point((1, -1)),
        ], index=['a', 'b'])
        >>> print(points.align(points_alpha))

        (0    POINT (-8.00000 -8.00000)
        1    POINT (-2.00000 -2.00000)
        a                         None
        b                         None
        dtype: geometry, 0                        None
        1                        None
        a    POINT (-1.00000 1.00000)
        b    POINT (1.00000 -1.00000)
        dtype: geometry)

        """
        index = (
            other.index if len(other.index) >= len(self.index) else self.index
        )
        aligned_left = self._align_to_index(index)
        aligned_right = other._align_to_index(index)
        aligned_right.index = index
        aligned_right = aligned_right.sort_index()
        aligned_left.index = (
            self.index
            if len(self.index) == len(aligned_left)
            else aligned_right.index
        )
        aligned_left = aligned_left.sort_index()
        return (
            aligned_left,
            aligned_right.loc[aligned_left.index],
        )

    def _gather(
        self, gather_map, keep_index=True, nullify=False, check_bounds=True
    ):
        return self.iloc[gather_map]

    def contains_properly(self, other, align=True):
        """Compute from a GeoSeries of points and a GeoSeries of polygons which
        points are properly contained within the corresponding polygon. Polygon
        A contains Point B properly if B intersects the interior of A but not
        the boundary (or exterior).

        Parameters
        ----------
        other
            a cuspatial.GeoSeries
        align=True
            to align the indices before computing .contains or not. If the
            indices are not aligned, they will be compared based on their
            implicit row order.

        Examples
        --------

        Test if a polygon is inside another polygon:
        >>> point = cuspatial.GeoSeries(
            [Point(0.5, 0.5)],
            )
        >>> polygon = cuspatial.GeoSeries(
            [
                Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]),
            ]
        )
        >>> print(polygon.contains(point))
        0    False
        dtype: bool


        Test whether three points fall within either of two polygons
        >>> point = cuspatial.GeoSeries(
            [Point(0, 0)],
            [Point(-1, 0)],
            [Point(-2, 0)],
            [Point(0, 0)],
            [Point(-1, 0)],
            [Point(-2, 0)],
            )
        >>> polygon = cuspatial.GeoSeries(
            [
                Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]),
                Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]),
                Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]),
                Polygon([[-2, -2], [-2, 2], [2, 2], [-2, -2]]),
                Polygon([[-2, -2], [-2, 2], [2, 2], [-2, -2]]),
                Polygon([[-2, -2], [-2, 2], [2, 2], [-2, -2]]),
            ]
        )
        >>> print(polygon.contains(point))
        0    False
        1    False
        2    False
        3    False
        4    False
        5     True
        dtype: bool

        Note
        ----
        poly_ring_offsets must contain only the rings that make up the polygons
        indexed by poly_offsets. If there are rings in poly_ring_offsets that
        are not part of the polygons in poly_offsets, results are likely to be
        incorrect and behavior is undefined.
        Note
        ----
        Polygons must be closed: the first and last coordinate of each polygon
        must be the same.

        Returns
        -------
        result : cudf.Series
            A Series of boolean values indicating whether each point falls
            within the corresponding polygon in the input.
        """
        if not contains_only_polygons(self):
            raise TypeError(
                "`.contains` can only be called with polygon series."
            )

        (lhs, rhs) = self.align(other) if align else (self, other)

        # RHS conditioning:
        mode = "POINTS"
        # point in polygon
        if contains_only_linestrings(rhs):
            # condition for linestrings
            mode = "LINESTRINGS"
            geom = rhs.lines
        elif contains_only_polygons(rhs) is True:
            # polygon in polygon
            mode = "POLYGONS"
            geom = rhs.polygons
        elif contains_only_multipoints(rhs) is True:
            # mpoint in polygon
            mode = "MULTIPOINTS"
            geom = rhs.multipoints
        else:
            # no conditioning is required
            geom = rhs.points
        xy_points = geom.xy
        point_indices = geom.point_indices()
        points = GeoSeries(GeoColumn._from_points_xy(xy_points._column)).points

        # call pip on the three subtypes on the right:
        point_result = contains_properly(
            points.x,
            points.y,
            lhs.polygons.part_offset[:-1],
            lhs.polygons.ring_offset[:-1],
            lhs.polygons.x,
            lhs.polygons.y,
        )
        if (
            mode == "LINESTRINGS"
            or mode == "POLYGONS"
            or mode == "MULTIPOINTS"
        ):
            # process for completed linestrings, polygons, and multipoints.
            # Not necessary for points.
            result = cudf.DataFrame(
                {"idx": point_indices, "pip": point_result}
            )
            # if the number of points in the polygon is equal to the number of
            # points, then the requirements for `.contains_properly` are met
            # for this geometry type.
            df_result = (
                result.groupby("idx").sum().sort_index()
                == result.groupby("idx").count().sort_index()
            )
            point_result = cudf.Series(
                df_result["pip"], index=cudf.RangeIndex(0, len(df_result))
            )
            point_result.name = None
        return point_result
