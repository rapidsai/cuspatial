# Copyright (c) 2020-2022, NVIDIA CORPORATION
from typing import Dict, Tuple, TypeVar, Union

import pandas as pd
from geopandas import GeoDataFrame as gpGeoDataFrame
from geopandas.geoseries import is_geometry_type as gp_is_geometry_type

import cudf

from cuspatial.core._column.geocolumn import GeoColumn, GeoMeta
from cuspatial.core.geoseries import GeoSeries
from cuspatial.io.geopandas_reader import GeoPandasReader

T = TypeVar("T", bound="GeoDataFrame")


class GeoDataFrame(cudf.DataFrame):
    """
    A GPU GeoDataFrame object.
    """

    def __init__(self, data: Union[Dict, gpGeoDataFrame] = None):
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
                    adapter = GeoPandasReader(data[col])
                    pandas_meta = GeoMeta(adapter.get_geopandas_meta())
                    column = GeoColumn(adapter._get_geotuple(), pandas_meta)
                    self._data[col] = column
                else:
                    self._data[col] = data[col]
        elif isinstance(data, dict):
            super()._init_from_dict_like(data)
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

    def _copy_type_metadata(
        self, other, include_index: bool = True, *, override_dtypes=None
    ):
        """
        Copy type metadata from each column of `other` to the corresponding
        column of `self`.
        See `ColumnBase._with_type_metadata` for more information.
        """

        type_copied = super()._copy_type_metadata(
            other, include_index=include_index, override_dtypes=override_dtypes
        )
        for name, col, other_col in zip(
            type_copied._data.keys(),
            type_copied._data.values(),
            other._data.values(),
        ):
            # A GeoColumn is currently implemented as a NumericalColumn with
            # several child columns to hold the geometry data. The native
            # return loop of cudf cython code can only reconstruct the
            # geocolumn as a NumericalColumn. We can't reconstruct the full
            # GeoColumn at the column level via _with_type_metadata, because
            # there is neither a well defined geometry type in cuspatial,
            # nor we can reconstruct the entire geocolumn data with pure
            # geometry type. So we need an extra pass here to copy the geometry
            # data into the type reconstructed dataframe.
            if isinstance(other_col, GeoColumn):
                col = GeoColumn(
                    (
                        other_col.points,
                        other_col.mpoints,
                        other_col.lines,
                        other_col.polygons,
                    ),
                    other_col._meta,
                    cudf.Index(col),
                )
                type_copied._data.set_by_label(name, col, validate=False)

        return type_copied

    def _split_out_geometry_columns(self) -> Tuple:
        """
        Break the geometry columns and non-geometry columns into
        separate dataframes and return them separated.
        """
        columns_mask = pd.Series(self.columns)
        geocolumn_mask = pd.Series(
            [isinstance(self[col], GeoSeries) for col in self.columns],
            dtype="bool",
        )
        geo_columns = self[columns_mask[geocolumn_mask]]
        # Send the rest of the columns to `cudf` to slice.
        data_columns = cudf.DataFrame(
            self[columns_mask[~geocolumn_mask].values]
        )
        return (geo_columns, data_columns)

    def _recombine_columns(self, geo_columns, data_columns):
        """
        Combine a GeoDataFrame of only geometry columns with a DataFrame
        of non-geometry columns in the same order as the columns in `self`
        """
        columns_mask = pd.Series(self.columns)
        geocolumn_mask = pd.Series(
            [isinstance(self[col], GeoSeries) for col in self.columns]
        )
        return {
            name: (geo_columns[name] if mask else data_columns[name])
            for name, mask in zip(columns_mask.values, geocolumn_mask.values)
        }

    def _slice(self: T, arg: slice) -> T:
        """
        Overload the _slice functionality from cudf's frame members.
        """
        geo_columns, data_columns = self._split_out_geometry_columns()
        sliced_geo_columns = GeoDataFrame(
            {name: geo_columns[name].iloc[arg] for name in geo_columns.columns}
        )
        sliced_data_columns = data_columns._slice(arg)
        result = self._recombine_columns(
            sliced_geo_columns, sliced_data_columns
        )
        return self.__class__(result)


class _GeoSeriesUtility:
    @classmethod
    def _from_data(cls, new_data, name=None, index=False):
        new_column = new_data.columns[0]
        if is_geometry_type(new_column):
            return GeoSeries(new_column, name=name, index=index)
        else:
            return cudf.Series(new_column, name=name, index=index)


def is_geometry_type(obj):
    """
    Returns `True` if the column is a `GeoPandas` or `cuspatial.GeoSeries`
    """
    if isinstance(obj, (GeoSeries, GeoColumn)):
        return True
    if gp_is_geometry_type(obj):
        return True
    return False
