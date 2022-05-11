GeoPandas Compatibility
-----------------------

cuSpatial supports any geometry format supported by `GeoPandas`. Load geometry information from a `GeoPandas.GeoSeries` or `GeoPandas.GeoDataFrame`.

    >>> gpdf = geopandas.read_file('arbitrary.txt')
        cugpdf = cuspatial.from_geopandas(gpdf)

or

    >>> cugpdf = cuspatial.GeoDataFrame(gpdf)

.. currentmodule:: cuspatial

.. autoclass:: cuspatial.GeoDataFrame
        :members:
.. autoclass:: cuspatial.GeoSeries
        :members:
