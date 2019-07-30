"""
verify the correctness of GPU-based implementation by comparing with shapely python package
GPU C++ kernel time 0.966ms, GPU C++ libcuspatial end-to-end time 1.104ms, GPU python cuspaital end-to-end time 1.270ms
shapely python end-to-end time 127659.4, 100,519X speedup (127659.4/1.27)
"""

import numpy as np
import time
import cuspatial.bindings.spatial as gis
import cuspatial.bindings.soa_readers as readers
from cudf.dataframe import columnops
from shapely.geometry import Polygon, Point
from shapely.geometry import shape
import shapefile

data_dir="/home/jianting/cuspatial/data/"
plyreader = shapefile.Reader(data_dir+"its_4326_roi.shp")
polygon = plyreader.shapes() 
plys=[]
for shape in polygon:
 plys.append(Polygon(shape.points)) 

pnt_x,pnt_y=readers.cpp_read_pnt_soa(data_dir+"locust.location");
fpos,rpos,plyx,plyy=readers.cpp_read_ply_soa(data_dir+"itsroi.ply")

start = time.time()
bm=gis.cpp_pip2_bm(pnt_x,pnt_y,fpos,rpos,plyx,plyy)
end = time.time()
print("Python GPU Time in ms (end-to-end)={}".format((end - start)*1000)) 

bma=bm.data.to_array()
pntx=pnt_x.data.to_array()
pnty=pnt_y.data.to_array()

start = time.time()
mis_match=0
for i in range(pnt_x.data.size):
 pt=Point(pntx[i],pnty[i])
 res=0
 for j in range(len(plys)):
  pip= plys[len(plys)-1-j].contains(pt)
  if(pip):
   res|=(0x01<<(len(plys)-1-j))
 if (res!=bma[i]):
  mis_match=mis_match+1

end = time.time()
print(end - start)  
print("python(shapely) CPU Time in ms (end-to-end)={}".format((end - start)*1000)) 

print("CPU and GPU results mismatch={}".format(mis_match))
