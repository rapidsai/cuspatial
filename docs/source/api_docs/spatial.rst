Spatial
-------

Functions that operate on spatial data.

.. currentmodule:: cuspatial

Spatial Indexing Functions
++++++++++++++++++++++++++
.. autofunction:: cuspatial.quadtree_on_points

Spatial Join Functions
++++++++++++++++++++++

.. autofunction:: cuspatial.point_in_polygon
.. autofunction:: cuspatial.quadtree_point_in_polygon
.. autofunction:: cuspatial.quadtree_point_to_nearest_linestring
.. autofunction:: cuspatial.join_quadtree_and_bounding_boxes

Measurement Functions
+++++++++++++++++++++

.. autofunction:: cuspatial.directed_hausdorff_distance
.. autofunction:: cuspatial.haversine_distance
.. autofunction:: cuspatial.pairwise_point_distance
.. autofunction:: cuspatial.pairwise_linestring_distance
.. autofunction:: cuspatial.pairwise_point_linestring_distance

Nearest Points Function
+++++++++++++++++++++++

.. autofunction:: cuspatial.pairwise_point_linestring_nearest_points

Bounding Boxes
++++++++++++++

.. autofunction:: cuspatial.polygon_bounding_boxes
.. autofunction:: cuspatial.linestring_bounding_boxes

Projection Functions
++++++++++++++++++++

.. autofunction:: cuspatial.sinusoidal_projection

Spatial Filtering Functions
+++++++++++++++++++++++++++

.. autofunction:: cuspatial.points_in_spatial_window
