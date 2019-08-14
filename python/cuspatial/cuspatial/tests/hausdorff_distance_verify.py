"""
Note: make sure cudf_dev conda environment is activated
"""

import numpy as np
import time
import sys
import pickle

from scipy.spatial.distance import directed_hausdorff
import cuspatial.bindings.spatial as gis
import cuspatial.bindings.soa_readers as readers

data_dir="/home/jianting/trajcode/"
data_set="locust256"

#scipy_res='scipyres.mat'
#cuspatial_res='cuspatialres.mat'
#if(len(sys.argv)>=2):
# scipy_res=sys.argv[1]
#if(len(sys.argv)>=3):
# cuspatial_res=sys.argv[2]

if(len(sys.argv)>=2):
 data_set=sys.argv[1]

#reading poing xy coordinate data (relative to a camera origin)
pnt_x,pnt_y=readers.cpp_read_pnt_xy_soa(data_dir+data_set+".coor");
#reading numbers of points in trajectories
cnt=readers.cpp_read_uint_soa(data_dir+data_set+".objcnt")
#reading object(vehicle) id
id=readers.cpp_read_uint_soa(data_dir+data_set+".objectid")

num_traj=cnt.data.size
dist0=gis.cpp_directed_hausdorff(pnt_x,pnt_y,cnt)
cuspatial_dist0=dist0.data.to_array().reshape((num_traj,num_traj))

start = time.time()
dist=gis.cpp_directed_hausdorff(pnt_x,pnt_y,cnt)
print("dis.size={} num_traj*num_traj={}".format(dist.data.size,num_traj*num_traj))
end = time.time()
print(end - start)  
print("python Directed Hausdorff distance GPU end-to-end time in ms (end-to-end)={}".format((end - start)*1000)) 

start = time.time()
cuspatial_dist=dist.data.to_array().reshape((num_traj,num_traj))
print("num_traj={}".format(num_traj))
print("cuspatial_dist[0[1]={}".format(cuspatial_dist[0][1]))

#with open(cuspatial_res, 'wb') as f:
#        pickle.dump(cuspatial_dist, f)

mis_match=0
for i in range(num_traj):
 for j in range(num_traj):
  if(abs(cuspatial_dist0[i][j]-cuspatial_dist[i][j])>0.00001):
   mis_match=mis_match+1
print('mis_match between two rounds ={}'.format(mis_match))


x=pnt_x.data.to_array()
y=pnt_y.data.to_array()
n=cnt.data.to_array()
end = time.time()  
print("data conversion time={}".format((end - start)*1000)) 

start = time.time()
trajs=[]
c=0
for i in range(num_traj):
 traj=np.zeros((n[i],2),dtype=np.float64)
 for j in range(n[i]):
  traj[j][0]=x[c+j]
  traj[j][1]=y[c+j]
 trajs.append(traj.reshape(-1,2))
 c=c+n[i]
#print('c={}'.format(c))
end=time.time()
print("CPU traj prep time={}".format((end - start)*1000)) 
#print("trajs[0]")
#print(trajs[0])

mis_match=0
d=np.zeros((num_traj,num_traj), dtype=np.float64)
for i in range(num_traj):
 if(i%100==99):
  print("i={}".format(i))
 for j in range(num_traj):
  dij=directed_hausdorff(trajs[i],trajs[j])
  d[i][j]=dij[0]
  if(abs(d[i][j]-cuspatial_dist[i][j])>0.00001):
   print('{} {} {} {}'.format(i,j,d[i][j],cuspatial_dist[i][j]))
   mis_match=mis_match+1
print('mis_match={}'.format(mis_match))  
end = time.time()  
print("python Directed Hausdorff distance cpu end-to-end time in ms (end-to-end)={}".format((end - start)*1000)) 

#for val in d[0]:
# print('{}'.format(val))
#print

#with open(scipy_res, 'wb') as f:
#        pickle.dump(d, f)
        