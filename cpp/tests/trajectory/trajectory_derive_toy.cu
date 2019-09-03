/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <time.h>
#include <sys/time.h>
#include <vector>
#include <string>
#include <iostream>

#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <utilities/error_utils.hpp>
#include <cuspatial/types.hpp> 
#include <cuspatial/trajectory.hpp> 
#include <utility/utility.hpp>
#include <utility/trajectory_thrust.cuh>

#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/cudf_test_utils.cuh>
#include <tests/utilities/cudf_test_fixtures.h>

using namespace cuspatial;

struct TrajectoryDeriveToy : public GdfTest 
{
};   
   
TEST_F(TrajectoryDeriveToy, trajectoryderivetest)
{
  //three sorted trajectories with 5,4,3 points, respectively
  std::cout<<"in TrajectoryDeriveToy"<<std::endl;
  double point_x[]={1.0,2.0,3.0,5.0,7.0,1.0,2.0,3.0,6.0,0.0,3.0,6.0};
  double point_y[]={0.0,1.0,2.0,3.0,1.0,3.0,5.0,6.0,5.0,4.0,7.0,4.0};
  uint32_t point_id[]={0,0,0,0,0,1,1,1,1,2,2,2};
  
  //handling timestamps - use millsecond field only for now 
  int point_ms[]={1,2,3,4,5,1,2,3,4,1,2,3};
  size_t num_point=sizeof(point_x)/sizeof(double);
  std::cout<<"num_point="<<num_point<<std::endl;
  
  std::vector<its_timestamp> point_ts;
  for(size_t i=0;i<num_point;i++)
  {
  	its_timestamp ts;
  	memset(&ts,0,sizeof(its_timestamp));
  	ts.ms=point_ms[i];	
  	point_ts.push_back(ts);
  }
  
  //randomize points 
  uint32_t seq[]={11,9,4,5,2,7,10,1,3,8,0,6};
  std::vector<double> rand_x;
  std::vector<double> rand_y;
  std::vector<uint32_t> rand_id;
  std::vector<its_timestamp> rand_ts;
  

  for(size_t i=0;i<num_point;i++)
  {
  	rand_x.push_back(point_x[seq[i]]);
  	rand_y.push_back(point_y[seq[i]]);
  	rand_id.push_back(point_id[seq[i]]);
  	rand_ts.push_back(point_ts[seq[i]]);
  }
  //std::copy(rand_ts.begin(),rand_ts.end(),std::ostream_iterator<its_timestamp>(std::cout, " "));std::cout<<std::endl;
  
  cudf::test::column_wrapper<double> point_x_wrapp{rand_x};
  cudf::test::column_wrapper<double> point_y_wrapp{rand_y};
  cudf::test::column_wrapper<uint32_t> point_id_wrapp{rand_id};
  cudf::test::column_wrapper<its_timestamp> point_ts_wrapp{rand_ts};

  gdf_column traj_id,traj_len,traj_offset;
  memset(&traj_id,0,sizeof(gdf_column));
  memset(&traj_len,0,sizeof(gdf_column));
  memset(&traj_offset,0,sizeof(gdf_column));
  gdf_column pnt_x=*(point_x_wrapp.get());
  gdf_column pnt_y=*(point_y_wrapp.get());
  gdf_column pnt_id=*(point_id_wrapp.get());
  gdf_column pnt_ts=*(point_ts_wrapp.get()); 
 
  std::cout<<"calling cuspatial::derive_trajectory"<<std::endl;
  uint32_t num_traj=cuspatial::derive_trajectories(pnt_x,pnt_y,pnt_id,pnt_ts,traj_id,traj_len,traj_offset);
  
  std::cout<<"point data after sorting"<<std::endl; 
  thrust::device_ptr<double> pnt_x_ptr=thrust::device_pointer_cast(static_cast<double*>(pnt_x.data));
  thrust::device_ptr<double> pnt_y_ptr=thrust::device_pointer_cast(static_cast<double*>(pnt_y.data));
  thrust::device_ptr<uint32_t> pnt_id_ptr=thrust::device_pointer_cast(static_cast<uint32_t*>(pnt_id.data));
  thrust::device_ptr<its_timestamp> pnt_ts_ptr=thrust::device_pointer_cast(static_cast<its_timestamp*>(pnt_ts.data));
 
  int num_print = (pnt_x.size<20)?pnt_x.size:20;  
  std::cout<<"x"<<std::endl;
  thrust::copy(pnt_x_ptr,pnt_x_ptr+num_print,std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl;  
  std::cout<<"y"<<std::endl;
  thrust::copy(pnt_y_ptr,pnt_y_ptr+num_print,std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl;  
  std::cout<<"id"<<std::endl;
  thrust::copy(pnt_id_ptr,pnt_id_ptr+num_print,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;  
  std::cout<<"timestamp"<<std::endl;
  thrust::copy(pnt_ts_ptr,pnt_ts_ptr+num_print,std::ostream_iterator<its_timestamp>(std::cout, " "));std::cout<<std::endl;
  
  std::cout<<"derived trajectories"<<std::endl;
  num_print = (num_traj<10)?num_traj:10;  
  thrust::device_ptr<uint32_t> traj_id_ptr=thrust::device_pointer_cast(static_cast<uint32_t*>(traj_id.data));
  thrust::device_ptr<uint32_t> traj_len_ptr=thrust::device_pointer_cast(static_cast<uint32_t*>(traj_len.data));
  thrust::device_ptr<uint32_t> traj_offset_ptr=thrust::device_pointer_cast(static_cast<uint32_t*>(traj_offset.data));
  std::cout<<"ids of trajectories"<<std::endl;
  thrust::copy(traj_id_ptr,traj_id_ptr+num_print,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl; 
  std::cout<<"#of points of trajectories"<<std::endl;
  thrust::copy(traj_len_ptr,traj_len_ptr+num_print,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl; 
  std::cout<<"poisition indices on sorted point x/y array of trajectories"<<std::endl; 
  thrust::copy(traj_offset_ptr,traj_offset_ptr+num_print,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;  
}
