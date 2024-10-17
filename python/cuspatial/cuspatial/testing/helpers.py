# Copyright (c) 2024, NVIDIA CORPORATION.
from itertools import chain

from shapely import get_coordinates


def geometry_to_coords(geom, geom_types):
    points_list = geom[geom.apply(lambda x: isinstance(x, geom_types))]
    # flatten multigeometries, then geometries, then coordinates
    points = list(chain(points_list.apply(get_coordinates)))
    coords_list = list(chain(*points))
    xy = list(chain(*coords_list))
    x = xy[::2]
    y = xy[1::2]
    return xy, x, y
