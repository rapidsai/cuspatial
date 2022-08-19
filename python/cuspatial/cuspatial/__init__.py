from ._version import get_versions
from .core import interpolate
from .core.spatial import (
    directed_hausdorff_distance,
    haversine_distance,
    join_quadtree_and_bounding_boxes,
    lonlat_to_cartesian,
    pairwise_linestring_distance,
    polygon_bounding_boxes,
    polyline_bounding_boxes,
    point_in_polygon,
    points_in_spatial_window,
    quadtree_on_points,
    quadtree_point_in_polygon,
    quadtree_point_to_nearest_polyline,
)
from .core.interpolate import CubicSpline
from .core.trajectory import (
    derive_trajectories,
    trajectory_bounding_boxes,
    trajectory_distances_and_speeds,
)
from .geometry.geoseries import GeoSeries
from .geometry.geodataframe import GeoDataFrame
from .io.shapefile import read_polygon_shapefile
from .io.geopandas import from_geopandas

__version__ = get_versions()["version"]
del get_versions
