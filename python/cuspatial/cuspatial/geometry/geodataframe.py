# Copyright (c) 2020-2021, NVIDIA CORPORATION

from geopandas import GeoDataFrame as gpGeoDataFrame

import cudf

from cuspatial.geometry.geoarrowbuffers import GeoArrowBuffers
from cuspatial.geometry.geocolumn import GeoColumn, GeoMeta
from cuspatial.geometry.geoseries import GeoSeries
from cuspatial.geometry.geoutil import is_geometry_type
from cuspatial.io.geopandas_adapter import GeoPandasAdapter


class GeoDataFrame(cudf.DataFrame):
    """
    A GPU GeoDataFrame object.
    """

    def __init__(self, data: gpGeoDataFrame = None):
        """
        Constructs a GPU GeoDataFrame from a GeoPandas dataframe.

        Parameters
        ----------
        data : A geopandas.geodataframe.GeoDataFrame object
        """
        super().__init__()
        if isinstance(data, gpGeoDataFrame):
            self.index = data.index
            for col in data.columns:
                if is_geometry_type(data[col]):
                    adapter = GeoPandasAdapter(data[col])
                    buffers = GeoArrowBuffers(
                        adapter.get_geoarrow_host_buffers()
                    )
                    pandas_meta = GeoMeta(adapter.get_geopandas_meta())
                    column = GeoColumn(buffers, pandas_meta)
                    self._data[col] = column
                else:
                    self[col] = data[col]
        elif data is None:
            pass
        else:
            raise ValueError("Invalid type passed to GeoDataFrame ctor")

    @property
    def _constructor(self):
        return GeoDataFrame

    @property
    def _constructor_sliced(self):
        return _GeoSeriesUtility

    def to_pandas(self, nullable=False):
        """
        Calls `self.to_geopandas`, converting GeoSeries columns into GeoPandas
        columns and cudf.Series columns into pandas.Series columns, and
        returning a pandas.DataFrame.

        Parameters
        ----------
        nullable: matches the cudf `to_pandas` signature, not yet supported.
        """
        return self.to_geopandas(nullable=nullable)

    def to_geopandas(self, nullable=False):
        """
        Returns a new GeoPandas GeoDataFrame object from the coordinates in
        the cuspatial GeoDataFrame.

        Parameters
        ----------
        nullable: matches the cudf `to_pandas` signature, not yet supported.
        """
        if nullable is True:
            raise ValueError("cuGeoDataFrame doesn't support N/A yet")
        result = gpGeoDataFrame(
            dict([(col, self[col].to_pandas()) for col in self.columns]),
            index=self.index.to_pandas(),
        )
        return result

    def __repr__(self):
        return self.to_pandas().__repr__() + "\n" + "(GPU)" + "\n"

    def groupby(self, *args, **kwargs):
        result = super().groupby(*args, **kwargs)
        for col in self.columns:
            if is_geometry_type(self[col]):
                result.obj = result.obj.drop(col, axis=1)
        return result


class _GeoSeriesUtility:
    def _from_data(self, new_data, name=None, index=False):
        new_column = new_data.columns[0]
        if is_geometry_type(new_column):
            return GeoSeries(new_column, name=name, index=index)
        else:
            return cudf.Series(new_column, name=name, index=index)
