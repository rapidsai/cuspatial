from ._version import get_versions
from .core import interpolate
from .core.geodataframe import GeoDataFrame
from .core.geoseries import GeoSeries
from .core.interpolate import CubicSpline
from .core.spatial import (
    directed_hausdorff_distance,
    haversine_distance,
    join_quadtree_and_bounding_boxes,
    linestring_bounding_boxes,
    pairwise_linestring_distance,
    pairwise_point_distance,
    pairwise_point_linestring_distance,
    pairwise_point_linestring_nearest_points,
    point_in_polygon,
    points_in_spatial_window,
    polygon_bounding_boxes,
    quadtree_on_points,
    quadtree_point_in_polygon,
    quadtree_point_to_nearest_linestring,
    sinusoidal_projection,
)
from .core.trajectory import (
    derive_trajectories,
    trajectory_bounding_boxes,
    trajectory_distances_and_speeds,
)
from .io.geopandas import from_geopandas
from .io.shapefile import read_polygon_shapefile

__version__ = get_versions()["version"]
del get_versions
