import time
import cudf
from cudf.dataframe import columnops
import cuspatial.bindings.spatial as gis

start = time.time()
#data dowloaded from https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2009-01.csv
df = cudf.read_csv("/home/jianting/hardbd19/data/nyctaxi/yellow_tripdata_2009-01.csv")
end = time.time()
print("data ingesting time (from SSD) in ms={}".format((end - start)*1000)) 
df.head().to_pandas().columns

start = time.time()
x1=columnops.as_column(df['Start_Lon'])
y1=columnops.as_column(df['Start_Lat'])
x2=columnops.as_column(df['End_Lon'])
y2=columnops.as_column(df['End_Lat'])
end = time.time()
print("data frame to gdf column conversion time in ms={}".format((end - start)*1000)) 

start = time.time()
h_dist=gis.cpp_haversine_distance(x1,y1,x2,y1)
end = time.time()
print("python computing distance time in ms={}".format((end - start)*1000)) 
#h_dist.data.to_array()