GeoPandas Compatibility
-----------------------

cuSpatial supports any geometry format supported by `GeoPandas`. Load geometry information from a `GeoPandas.GeoSeries` or `GeoPandas.GeoDataFrame`.

    >>> host_dataframe = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
        cugpdf = cuspatial.from_geopandas(host_dataframe)

or

    >>> cugpdf = cuspatial.GeoDataFrame(gpdf)

.. currentmodule:: cuspatial

.. autoclass:: cuspatial.GeoDataFrame
        :members:
.. autoclass:: cuspatial.GeoSeries
        :members:
