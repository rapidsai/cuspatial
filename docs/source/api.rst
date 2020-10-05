~~~~~~~~~~~~~~~~~~~~~~~
cuSpatial API Reference
~~~~~~~~~~~~~~~~~~~~~~~


GIS
---
.. currentmodule:: cuspatial

.. autofunction:: cuspatial.directed_hausdorff_distance
.. autofunction:: cuspatial.haversine_distance
.. autofunction:: cuspatial.lonlat_to_cartesian
.. autofunction:: cuspatial.point_in_polygon
.. autofunction:: cuspatial.polygon_bounding_boxes
.. autofunction:: cuspatial.polyline_bounding_boxes


Indexing
--------
.. currentmodule:: cuspatial

.. autofunction:: cuspatial.quadtree_on_points


Interpolation
-------------
.. currentmodule:: cuspatial

.. autoclass:: CubicSpline
.. automethod:: CubicSpline.__init__
.. automethod:: CubicSpline.__call__


Spatial Querying
----------------
.. currentmodule:: cuspatial

.. autofunction:: cuspatial.points_in_spatial_window


Spatial Joining
----------------
.. currentmodule:: cuspatial

.. autofunction:: cuspatial.join_quadtree_and_bounding_boxes
.. autofunction:: cuspatial.quadtree_point_in_polygon
.. autofunction:: cuspatial.quadtree_point_to_nearest_polyline

Trajectory
----------
.. currentmodule:: cuspatial

.. autofunction:: cuspatial.derive_trajectories
.. autofunction:: cuspatial.trajectory_bounding_boxes
.. autofunction:: cuspatial.trajectory_distances_and_speeds


IO
--
.. currentmodule:: cuspatial

.. autofunction:: cuspatial.read_polygon_shapefile
