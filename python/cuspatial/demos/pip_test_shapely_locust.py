"""
PIP demo directly using shapely, more efficient than using python gdal/ogr
directly polygons are created only once and stored for reuse

To run the demo, first install python gdal and pyshp by `conda install -c
conda-forge gdal pyshp` under cudf_dev environment
"""

import numpy as np
import shapefile
from shapely.geometry import Point, Polygon

data_dir = "/home/jianting/cuspatial/data/"

plyreader = shapefile.Reader(data_dir + "its_4326_roi.shp")
polygon = plyreader.shapes()
plys = []
for shape in polygon:
    plys.append(Polygon(shape.points))

pnt_x = np.array(
    [-90.666418409895840, -90.665136925928721, -90.671840534675397],
    dtype=np.float64,
)
pnt_y = np.array(
    [42.492199401857071, 42.492104092138952, 42.490649501411141],
    dtype=np.float64,
)

for i in range(3):
    pt = Point(pnt_x[i], pnt_y[i])
    res = ""
    for j in range(len(plys)):
        pip = plys[len(plys) - 1 - j].contains(pt)
        if pip:
            res += "1"
        else:
            res += "0"
    print(res)
