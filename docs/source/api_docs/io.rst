IO
--

cuSpatial offers native GPU-accelerated shapefile reading. In addition, any host-side GeoPandas DataFrame can be copied into GPU memory for use with cuSpatial
algorithms.

.. currentmodule:: cuspatial

.. autofunction:: cuspatial.read_polygon_shapefile
.. autofunction:: cuspatial.from_geopandas
