Welcome to cuSpatial's documentation!
=====================================

cuSpatial provides functionality for fast GPU-based spatial index and join
functions via point-in-polygon, a pathing library for trajectory identification
and reconstruction, and accelerated GIS functions like haversine distance and
grid projection. The icing-on-the-cake of cuSpatial is that we integrate neatly
with `GeoPandas` and RAPIDS `cudf`.

This enables you to take advantage of all of the features of `GeoPandas`, plus
blisteringly fast GPU acceleration for pandas-like functionality in `cudf` and
spatial functions that perform quickly on millions and tens of millions of
geometries.

GeoArrow
--------

cuSpatial proposes a new GeoArrow format from the fruit of discussions
with the GeoPandas team. GeoArrow is a packed columnar data format
for the six fundamental geometry types:
Point, MultiPoint, Lines, MultiLines, Polygons, and MultiPolygons.
MultiGeometry is a possibility that may be implemented in the future.
GeoArrow uses packed coordinate and offset columns to define objects,
which enables very fast copies between CPU, GPU, and NIC.

Any data source that is loaded into cuSpatial via :func:`cuspatial.from_geopandas`
can then take advantage of `cudf`'s GPU-accelerated Arrow I/O routines.

Read more about GeoArrow format in :ref:`GeoArrow Format`.

cuSpatial API Reference
-----------------------

.. toctree::
   :maxdepth: 2

   api_docs/gis.rst
   api_docs/spatial_indexing.rst
   api_docs/trajectory.rst
   api_docs/geopandas_compatibility.rst
   api_docs/io.rst
   api_docs/internals.rst
 

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
