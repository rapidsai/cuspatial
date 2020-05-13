"""
verify the correctness of GPU-based implementation by comparing with shapely
python package GPU C++ kernel time 0.966ms, GPU C++ libcuspatial end-to-end
time 1.104ms, GPU python cuspaital end-to-end time 1.270ms shapely python
end-to-end time 127659.4, 100,519X speedup (127659.4/1.27)
"""

import time

import shapefile
from shapely.geometry import Point, Polygon

import cuspatial

data_dir = "/home/jianting/cuspatial/data/"
plyreader = shapefile.Reader(data_dir + "its_4326_roi.shp")
polygon = plyreader.shapes()
plys = []
for shape in polygon:
    plys.append(Polygon(shape.points))

pnt_lon, pnt_lat = cuspatial.read_points_lonlat(data_dir + "locust.location")
fpos, rpos, plyx, plyy = cuspatial.read_polygon(data_dir + "itsroi.ply")

start = time.time()
bm = cuspatial.point_in_polygon(pnt_lon, pnt_lat, fpos, rpos, plyx, plyy)
end = time.time()
print("Python GPU Time in ms (end-to-end)={}".format((end - start) * 1000))

bma = bm.data.to_array()
pntx = pnt_lon.data.to_array()
pnty = pnt_lat.data.to_array()

start = time.time()
mis_match = 0
for i in range(pnt_lon.data.size):
    pt = Point(pntx[i], pnty[i])
    res = 0
    for j in range(len(plys)):
        pip = plys[len(plys) - 1 - j].contains(pt)
        if pip:
            res |= 0x01 << (len(plys) - 1 - j)
    if res != bma[i]:
        mis_match = mis_match + 1

end = time.time()
print(end - start)
print(
    "python(shapely) CPU Time in ms (end-to-end)={}".format(
        (end - start) * 1000
    )
)

print("CPU and GPU results mismatch={}".format(mis_match))
