# Copyright (c) 2020-2022 NVIDIA CORPORATION.

import pyarrow as pa

from geopandas import GeoSeries as gpGeoSeries
from cuspatial.geometry import pygeoarrow


class GeoPandasAdapter:
    buffers = None
    source = None

    def __init__(self, geoseries: gpGeoSeries):
        """
        GeoPandasAdapter copies a GeoPandas GeoSeries object iteratively into
        a set of arrays: points, multipoints, lines, and polygons.

        Parameters
        ----------
        geoseries : A GeoPandas GeoSeries
        """
        self.buffers = pygeoarrow.from_geopandas(geoseries)
        self.source = geoseries

    def get_geoarrow_host_buffers(self) -> dict:
        """
        Returns a set of host buffers containing the geopandas object converted
        to GeoArrow format.
        """
        points_xy = self.buffers.field(0).values
        mpoints_xy = self.buffers.field(1).values
        mpoints_offsets = self.buffers.field(1).offsets
        lines_xy = self.buffers.field(2).values.values
        lines_offsets = self.buffers.field(2).offsets
        mlines = lines_offsets
        polygons_xy = self.buffers.field(3).values.values.values
        mpolygons = self.buffers.field(3).offsets
        polygons_polygons = self.buffers.field(3).values.offsets
        polygons_rings = self.buffers.field(3).values.values.offsets
        return {
            "points_xy": points_xy,
            "mpoints_xy": mpoints_xy,
            "mpoints_offsets": mpoints_offsets,
            "lines_xy": lines_xy,
            "lines_offsets": lines_offsets,
            "mlines": mlines,
            "polygons_xy": polygons_xy,
            "polygons_polygons": polygons_polygons,
            "polygons_rings": polygons_rings,
            "mpolygons": mpolygons,
        }

    def get_geoarrow_union(self) -> pa.UnionArray:
        return self.buffers

    def get_geopandas_meta(self) -> dict:
        """
        Returns the metadata that was created converting the GeoSeries into
        GeoArrow format. The metadata essentially contains the object order
        in the GeoSeries format. GeoArrow doesn't support custom orderings,
        every GeoArrow data store contains points, multipoints, lines, and
        polygons in an arbitrary order.
        """
        buffers = self.buffers
        return {
            "input_types": buffers.type_codes,
            "input_lengths": buffers.offsets,
            "inputs": self.source,
        }
