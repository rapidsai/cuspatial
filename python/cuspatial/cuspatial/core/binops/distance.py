# Copyright (c) 2023, NVIDIA CORPORATION.

from itertools import product

from cuspatial._lib.distance import (
    pairwise_point_distance ,
    pairwise_point_linestring_distance ,
    pairwise_point_polygon_distance ,
    pairwise_linestring_distance ,
    pairwise_linestring_polygon_distance ,
    pairwise_polygon_distance ,
)

from cuspatial.core._column.geometa import Feature_Enum as F
from cuspatial.core._column.geocolumn import GeoColumn
from cuspatial.core.dispatch.geocolumn_binop_dispatch import GeoColumnBinopDispatch


class DistanceDispatch(GeoColumnBinopDispatch):
    def __init__(self, lhs: GeoColumn, rhs: GeoColumn):
        super().__init__(lhs, rhs)
        self.dispatch_dict = {
            (F.POINT, F.POINT): (pairwise_point_distance, False),
            (F.POINT, F.MULTIPOINT): (pairwise_point_distance, False),
            (F.POINT, F.LINESTRING): (pairwise_point_linestring_distance, False),
            (F.POINT, F.POLYGON): (pairwise_point_polygon_distance, False),

            (F.MULTIPOINT, F.POINT): (pairwise_point_distance, False),
            (F.MULTIPOINT, F.MULTIPOINT): (pairwise_point_distance, False),
            (F.MULTIPOINT, F.LINESTRING): (pairwise_point_linestring_distance, False),
            (F.MULTIPOINT, F.POLYGON): (pairwise_point_polygon_distance, False),

            (F.LINESTRING, F.POINT): (pairwise_point_linestring_distance, True),
            (F.LINESTRING, F.MULTIPOINT): (pairwise_point_linestring_distance, True),
            (F.LINESTRING, F.LINESTRING): (pairwise_linestring_distance, False),
            (F.LINESTRING, F.POLYGON): (pairwise_linestring_polygon_distance, False),

            (F.POLYGON, F.POINT): (pairwise_point_polygon_distance, True),
            (F.POLYGON, F.MULTIPOINT): (pairwise_point_polygon_distance, True),
            (F.POLYGON, F.LINESTRING): (pairwise_linestring_polygon_distance, True),
            (F.POLYGON, F.POLYGON): (pairwise_polygon_distance, False),
        }

        none = [F.NONE]
        other = [F.POINT, F.MULTIPOINT, F.LINESTRING, F.POLYGON]
        for ltype, rtype in [*product(none, other), *product(other, none)]:
            self.dispatch_dict[(ltype, rtype)] = ("Impossible", None)
