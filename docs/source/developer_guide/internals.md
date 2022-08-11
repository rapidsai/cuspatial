# Internals

This page includes information to help users understand the internal
data structure of cuspatial.

GeoArrow Format
+++++++++++++++

cuspatial uses an implementation of the [GeoArrow Format](
https://github.com/geopandas/geo-arrow-spec              ) for high-  
performance calculations on the GPU. In general, features are collected  
into homogeneous buffers based on feature type. GeoArrow specifies a data  
structure for Points, MultiPoints, LineStrings, and Polygons. Algorithms
such as spatial join via point-in-polygon take as arguments a buffer of
points and a buffer of polygons.

GeoArrow uses offset buffers to encode where each feature begins and ends.  
Our algorithms will usually require that these offset buffers be passed  
along with the coordinate values. LineStrings have two extra buffers: a  
lines buffer, identifying which linestrings are multilinestrings, and an  
offsets buffer that identifies where each linestring begins and ends.

Polygons have an extra buffer, the rings buffer, that identifies which  
groups of coordinates are the exterior and interior rings of the polygon.  

## IO

Two primary methods exist for getting data into cuspatial to utilize our  
algorithms. 

[`cuspatial.read_polygon_shapefile`]() can read Shapefiles
full of single polygons via C++ with high performance. 

[`cuspatial.from_geopandas`]() utilizes a host geopandas DataFrame to  
prepare the data first, then copies the data into GeoArrow format on  
the GPU.

GeoArrow Internal APIs
**********************

.. autoclass:: cuspatial.GeoArrowBuffers
        :members:
.. autoclass:: cuspatial.geometry.geocolumn.GeoMeta
.. autoclass:: cuspatial.geometry.geocolumn.GeoColumn
        :members:
