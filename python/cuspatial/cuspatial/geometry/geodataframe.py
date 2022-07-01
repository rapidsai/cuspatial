# Copyright (c) 2020-2022, NVIDIA CORPORATION

from geopandas import GeoDataFrame as gpGeoDataFrame

import cudf
import pyarrow as pa

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
        data : A geopandas.GeoDataFrame object
        """
        super().__init__()
        if isinstance(data, gpGeoDataFrame):
            self.index = data.index
            for col in data.columns:
                if is_geometry_type(data[col]):
                    adapter = GeoPandasAdapter(data[col])
                    buffers = GeoArrowBuffers(
                        adapter.get_geoarrow_union(), data_locale=pa
                    )
                    pandas_meta = GeoMeta(adapter.get_geopandas_meta())
                    column = GeoColumn(buffers, pandas_meta)
                    self._data[col] = column
                else:
                    self._data[col] = data[col]
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
                result.obj.drop(col, axis=1, inplace=True)
        return result

    def _copy_type_metadata(self, other, include_index: bool = True):
        """
        Copy type metadata from each column of `other` to the corresponding
        column of `self`.
        See `ColumnBase._with_type_metadata` for more information.
        """
        for name, col, other_col in zip(
            self._data.keys(), self._data.values(), other._data.values()
        ):
            # libcudf APIs lose all information about GeoColumns, operating
            # solely on the underlying base data. Therefore, our only recourse
            # is to recreate a new GeoColumn with the same underlying data.
            # Since there's no easy way to create a GeoColumn from a
            # NumericalColumn, we're forced to do so manually.
            if isinstance(other_col, GeoColumn):
                col = GeoColumn(
                    other_col._geo, other_col._meta, cudf.Index(col)
                )

            self._data.set_by_label(
                name, col._with_type_metadata(other_col.dtype), validate=False
            )

        if include_index:
            if self._index is not None and other._index is not None:
                self._index._copy_type_metadata(other._index)
                # When other._index is a CategoricalIndex, there is
                if isinstance(
                    other._index, cudf.core.index.CategoricalIndex
                ) and not isinstance(
                    self._index, cudf.core.index.CategoricalIndex
                ):
                    self._index = cudf.Index(self._index._column)

        return self


class _GeoSeriesUtility:
    @classmethod
    def _from_data(cls, new_data, name=None, index=False):
        new_column = new_data.columns[0]
        if is_geometry_type(new_column):
            return GeoSeries(new_column, name=name, index=index)
        else:
            return cudf.Series(new_column, name=name, index=index)
