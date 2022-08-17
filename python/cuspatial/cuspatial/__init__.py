from ._version import get_versions
from .core import interpolate
from .core.spatial import *
from .core.interpolate import CubicSpline
from .core.trajectory import *
from .geometry.geoseries import GeoSeries
from .geometry.geodataframe import GeoDataFrame
from .io.shapefile import read_polygon_shapefile
from .io.geopandas import from_geopandas

__version__ = get_versions()["version"]
del get_versions
