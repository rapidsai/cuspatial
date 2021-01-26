# Copyright (c) 2020-2021, NVIDIA CORPORATION

from geopandas import GeoDataFrame as gpGeoDataFrame
from geopandas.geoseries import is_geometry_type

import cudf

from cuspatial.geometry.geoseries import GeoSeries


class GeoDataFrame(cudf.DataFrame):
    def __init__(self, data):
        super().__init__()
        """
        A GPU GeoDataFrame object.
        """
        if isinstance(data, gpGeoDataFrame):
            for col in data.columns:
                if is_geometry_type(data[col]):
                    self[col] = GeoSeries(data[col])
                else:
                    self[col] = data[col]
            self.index = data.index
        else:
            raise ValueError("Invalid type passed to GeoDataFrame()")

    def to_pandas(self, index=None, nullable=False):
        return self.to_geopandas(index=index, nullable=nullable)

    def to_geopandas(self, index=None, nullable=False):
        """
        Returns a new GeoPandas GeoDataFrame object from the coordinates in
        the cuspatial GeoDataFrame.
        """
        if nullable is True:
            raise ValueError("cuGeoDataFrame doesn't support N/A yet")
        if index is None:
            index = self.index.to_array()
        result = gpGeoDataFrame()
        for col in self.columns:
            if is_geometry_type(self[col]):
                geoSeries = []
                for i in self[col]:
                    geoSeries.append(self[col].to_shapely())
                result[col] = geoSeries
            else:
                result[col] = self[col].to_pandas()
        result.index = index
        return result

    def __repr__(self):
        return (
            self.to_pandas().__repr__()
            + "\n"
            + "(GPU)"
            + "\n"
        )

    def de_interleave(self):
        pass
        # de_interleave all Series
