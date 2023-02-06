IO
--

cuSpatial offers limited shapefile reading (polygons only), but this is deprecated and will be
removed in a future release.

Any host-side GeoPandas DataFrame can be copied into GPU memory for use with cuSpatial algorithms.

.. currentmodule:: cuspatial

.. autofunction:: cuspatial.read_polygon_shapefile
.. autofunction:: cuspatial.from_geopandas
