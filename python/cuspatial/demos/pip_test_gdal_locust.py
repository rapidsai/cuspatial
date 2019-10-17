"""
pip demo directly using gdal/ogr for python; not for performance comparisons.
To run the demo, first install python-gdal by `conda install -c conda-forge
gdal` under cudf_dev environment
"""

import numpy as np
from osgeo import ogr

data_dir = "/home/jianting/cuspatial/data/"
shapefile = data_dir + "its_4326_roi.shp"
driver = ogr.GetDriverByName("ESRI Shapefile")
spatialReference = ogr.osr.SpatialReference()
spatialReference.SetWellKnownGeogCS("WGS84")
pt = ogr.Geometry(ogr.wkbPoint)
pt.AssignSpatialReference(spatialReference)
pnt_x = np.array(
    [-90.666418409895840, -90.665136925928721, -90.671840534675397],
    dtype=np.float64,
)
pnt_y = np.array(
    [42.492199401857071, 42.492104092138952, 42.490649501411141],
    dtype=np.float64,
)

for i in range(3):
    pt.SetPoint(0, pnt_x[i], pnt_y[i])
    res = ""

    """
 features can not be saved for later reuse
 known issue: https://trac.osgeo.org/gdal/wiki/PythonGotchas
 """

    dataSource = driver.Open(shapefile, 0)
    layer = dataSource.GetLayer()
    for f in layer:
        pip = pt.Within(f.geometry())
        if pip:
            res += "1"
        else:
            res += "0"
    print(res)
