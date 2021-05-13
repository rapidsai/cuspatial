from ._version import get_versions
from .core import interpolate
from .core.gis import (
    directed_hausdorff_distance,
    haversine_distance,
    lonlat_to_cartesian,
    point_in_polygon,
    polygon_bounding_boxes,
    polyline_bounding_boxes,
)
from .core.indexing import quadtree_on_points
from .core.interpolate import CubicSpline
from .core.spatial_join import (
    join_quadtree_and_bounding_boxes,
    quadtree_point_in_polygon,
    quadtree_point_to_nearest_polyline,
)
from .core.spatial_window import points_in_spatial_window
from .core.trajectory import (
    derive_trajectories,
    trajectory_bounding_boxes,
    trajectory_distances_and_speeds,
)
from .geometry.geoseries import GeoSeries
from .geometry.geodataframe import GeoDataFrame
from .geometry.geoarrowbuffers import GeoArrowBuffers
from .io.shapefile import read_polygon_shapefile
from .io.geopandas import from_geopandas

__version__ = get_versions()["version"]
del get_versions
