GeoPandas Compatibility
-----------------------

We support any geometry format supported by `GeoPandas`. Load geometry information from a `GeoPandas.GeoSeries` or `GeoPandas.GeoDataFrame`.

    >>> gpdf = geopandas.read_file('arbitrary.txt')
        cugpdf = cuspatial.from_geopandas(gpdf)

or

    >>> cugpdf = cuspatial.GeoDataFrame(gpdf)

.. currentmodule:: cuspatial

.. autoclass:: cuspatial.GeoDataFrame
        :members:
        :show-inheritance:
.. autoclass:: cuspatial.GeoSeries
        :members:
        :show-inheritance:
.. autoclass:: cuspatial.geometry.geocolumn.GeoColumn
        :members:
        :show-inheritance:
.. autoclass:: cuspatial.GeoArrowBuffers
        :members:
        :show-inheritance:


Spatial Indexing
--------

Spatial indexing functions provide blisteringly-fast on-GPU point-in-polygon
operations.

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

Two GIS functions make it easier to compute distances with geographic coordinates.

.. currentmodule:: cuspatial

.. autofunction:: cuspatial.haversine_distance
.. autofunction:: cuspatial.lonlat_to_cartesian


Trajectory
----------

Trajectory functions make it easy to identify and group trajectories from point data.

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

cuSpatial offers native GPU-accelerated shapefile reading. In addition, any host-side GeoPandas DataFrame can be copied into GPU memory for use with cuSpatial
algorithms.

.. currentmodule:: cuspatial

.. autofunction:: cuspatial.read_polygon_shapefile
.. autofunction:: cuspatial.from_geopandas

