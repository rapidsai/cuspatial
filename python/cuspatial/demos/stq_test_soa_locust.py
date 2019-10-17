"""
GPU-based spatial window query demo using 1.3 million points read from file
and (x1,x2,y1,y2)=[-180,180,-90,90] as the query window num should be the same
as x.data.size, both are 1338671
"""

import cuspatial

data_dir = "./data/"
data = cuspatial.read_points_lonlat(data_dir + "locust.location")

points_inside = cuspatial.window_points(
    -180, -90, 180, 90, data["lon"], data["lat"]
)
print(points_inside.shape[0])
assert points_inside.shape[0] == data.shape[0]
