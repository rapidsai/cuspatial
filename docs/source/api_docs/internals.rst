Internals
---------

This page includes information to help users understand the internal
data structure of cuspatial.

GeoArrow Format
+++++++++++++++

Geospatial data is context rich; aside from just a set of
numbers representing coordinates, they together represent certain geometry
that requires grouping. For example, given 5 points in a plane,
they could be 5 separate points, 2 line segments, a single linestring,
or a pentagon. Many geometry libraries stores the points in
arrays of geometric objects, commonly known as "Array of Structure" (AoS).
AoS is not efficient for accelerated computing on parallel devices such
as GPU. Therefore, GeoArrow format was introduced to store geodata in
densely packed format, commonly known as "Structure of Arrays" (SoA).

The GeoArrow format specifies a tabular data format for geometry
information. Supported types include `Point`, `MultiPoint`, `LineString`,
`MultiLineString`, `Polygon`, and `MultiPolygon`. In order to store
these coordinate types in a strictly tabular fashion, columns are
created for Points, MultiPoints, LineStrings, and Polygons.
MultiLines and MultiPolygons are stored in the same data structure
as LineStrings and Polygons.

GeoArrow format packs complex geometry types into 14 single-column Arrow
tables. See :func:`GeoArrowBuffers<cuspatial.GeoArrowBuffers>` docstring
for the complete list of keys for the columns.

Examples
********

The `Point` geometry is the simplest. N points are stored in a length 2*N
buffer with interleaved x,y coordinates. An optional z buffer of length N
can be used.

A `Multipoint` is a group of points, and is the second simplest GeoArrow 
geometry type. It is identical to points, with the addition of a 
`multipoints_offsets` buffer. The offsets buffer stores N+1 indices. The
first multipoint offset is specified by 0, which is always stored in 
`offsets[0]`. The second offset is stored in `offsets[1]`, and so on.
The number of points in multipoint `i` is the difference between
`offsets[i+1]` and `offsets[i]`. 


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

`LineString` geometry is more complicated than multipoints because the
format allows for the use of `LineStrings` and `MultiLineStrings` in the same
buffer, via the `mlines` key::

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
`Shapefile <https://en.wikipedia.org/wiki/Shapefile>`_ ,
with left-wound external rings and right-wound internal rings.

GeoArrow Internal APIs
**********************

.. autoclass:: cuspatial.GeoArrowBuffers
        :members:
.. autoclass:: cuspatial.geometry.geocolumn.GeoMeta
.. autoclass:: cuspatial.geometry.geocolumn.GeoColumn
        :members:
