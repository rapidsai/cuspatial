~~~~~~~~~~~~~~~~~~~~~~~
cuSpatial API Reference
~~~~~~~~~~~~~~~~~~~~~~~

cuSpatial provides functionality for fast GPU-based spatial index and join
functinos via point-in-polygon, a pathing library for trajectory identification
and reconstruction, and accelerated GIS functions like haversine distance and grid pro jection. The icing-on-the-cake of cuSpatial is that we integrate neatly
with `GeoPandas` and RAPIDS `cudf`.

This enables you to take advantage of all of the features of `GeoPandas`, plus
blisteringly fast GPU acceleration for pandas-like functionality in `cudf` and
spatial functions that perform quickly on millions and tens of millions of
geometries.

GeoPandas Interop
-----------------

Load geometry information from a `GeoPandas.GeoSeries` or `GeoPandas.GeoDataFrame`.

    >>> gpdf = geopandas.read_file('arbitrary.txt')
        cugpdf = cuspatial.from_geopandas(gpdf)

or

    >>> cugpdf = cuspatial.GeoDataFrame(gpdf)

.. currentmodule:: cuspatial

.. autoclass:: cuspatial.GeoDataFrame
        :members:
        :undoc-members:
        :show-inheritance:
.. autoclass:: cuspatial.GeoSeries
        :members:
        :undoc-members:
        :show-inheritance:
.. autoclass:: cuspatial.geometry.geocolumn.GeoColumn
        :members:
        :undoc-members:
        :show-inheritance:
.. autoclass:: cuspatial.GeoArrowBuffers
        :members:
        :undoc-members:
        :show-inheritance:


Spatial Indexing
--------
.. currentmodule:: cuspatial

.. autofunction:: cuspatial.quadtree_point_in_polygon
.. autofunction:: cuspatial.quadtree_point_to_nearest_polyline
.. autofunction:: cuspatial.point_in_polygon
.. autofunction:: cuspatial.polygon_bounding_boxes
.. autofunction:: cuspatial.polyline_bounding_boxes
.. autofunction:: cuspatial.quadtree_on_points
.. autofunction:: cuspatial.join_quadtree_and_bounding_boxes
.. autofunction:: cuspatial.points_in_spatial_window


GIS
---
.. currentmodule:: cuspatial

.. autofunction:: cuspatial.haversine_distance
.. autofunction:: cuspatial.lonlat_to_cartesian


Trajectory
----------
.. currentmodule:: cuspatial

.. autofunction:: cuspatial.derive_trajectories
.. autofunction:: cuspatial.trajectory_distances_and_speeds
.. autofunction:: cuspatial.directed_hausdorff_distance
.. autofunction:: cuspatial.trajectory_bounding_boxes
.. autoclass:: CubicSpline
.. automethod:: CubicSpline.__init__
.. automethod:: CubicSpline.__call__


IO
--
.. currentmodule:: cuspatial

.. autofunction:: cuspatial.read_polygon_shapefile
.. autofunction:: cuspatial.from_geopandas

