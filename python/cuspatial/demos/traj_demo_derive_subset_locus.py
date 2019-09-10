"""
demo of chaining three APIs: derive_trajectories+subset_trajectory(by ID)
+hausdorff_distance also serves as an example to integrate cudf and cuspatial
"""

import cuspatial

data_dir = "./data/"
lonlats = cuspatial.read_points_lonlat(data_dir + "locust.location")
ids = cuspatial.read_uint(data_dir + "locust.objectid")
ts = cuspatial.read_its_timestamps(data_dir + "locust.time")

num_traj, trajectories = cuspatial.derive(
    lonlats["lon"], lonlats["lat"], ids, ts
)
df = trajectories.query("length>=256")
query_ids = df["trajectory_id"]
query_cnts = df["length"]
new_trajs = cuspatial.subset_trajectory_id(
    query_ids, lonlats["lon"], lonlats["lat"], ids, ts
)
new_lon = new_trajs["x"]
new_lat = new_trajs["y"]
num_traj = df.count()[0]
dist = cuspatial.directed_hausdorff_distance(new_lon, new_lat, query_cnts)
cuspatial_dist0 = dist.data.to_array().reshape((num_traj, num_traj))
print(cuspatial_dist0)
