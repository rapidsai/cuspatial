# Copyright (c) 2020-2021, NVIDIA CORPORATION

from geopandas import GeoDataFrame as gpGeoDataFrame
from geopandas.geoseries import is_geometry_type as gp_is_geometry_type

import cudf

from cuspatial.geometry.geoseries import GeoSeries, GeoColumn


def is_geometry_type(obj):
    """
    Returns `True` if the column is a `GeoPandas` or `cuspatial.GeoSeries`
    """
    if gp_is_geometry_type(obj):
        return True
    if isinstance(obj, (GeoSeries, GeoColumn)):
        return True
    return False


class GeoDataFrame(cudf.DataFrame):
    """
    A GPU GeoDataFrame object.
    """

    def __init__(self, data=None, index=None, columns=None, dtype=None):
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
                    self._data[col] = GeoColumn(data[col])
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

    """
    def __setitem__(self, arg, value):
        if isinstance(value, GeoSeries):
            self._data[arg] = value
        else:
            super().__setitem__(arg, value)
    """

    def to_pandas(self, index=None, nullable=False):
        """
        Calls `self.to_geopandas`, converting GeoSeries columns into GeoPandas
        columns and cudf.Series columns into pandas.Series columns, and
        returning a pandas.DataFrame.

        Parameters
        ----------
        index : a `cudf.Index` or `pandas.Index`.
        """
        return self.to_geopandas(index=index, nullable=nullable)

    def to_geopandas(self, index=None, nullable=False):
        """
        Returns a new GeoPandas GeoDataFrame object from the coordinates in
        the cuspatial GeoDataFrame.

        Parameters
        ----------
        index : a `cudf.Index` or `pandas.Index`.
        """
        if nullable is True:
            raise ValueError("cuGeoDataFrame doesn't support N/A yet")
        if index is None:
            index = self.index.to_pandas()
        result = gpGeoDataFrame(
            dict([(col, self[col].to_pandas()) for col in self.columns])
        )
        result = result.set_index(index)
        result.columns = self.columns
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
