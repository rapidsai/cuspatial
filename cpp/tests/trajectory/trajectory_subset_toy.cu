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

struct TrajectorySubsetToy : public GdfTest 
{
};   
   
TEST_F(TrajectorySubsetToy, trajectorysubsettest)
{
  //three sorted trajectories with 5,4,3 points, respectively
  std::cout<<"in TrajectorySubsetToy"<<std::endl;
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
  std::copy(rand_ts.begin(),rand_ts.end(),std::ostream_iterator<its_timestamp>(std::cout, " "));std::cout<<std::endl;
  
  
  cudf::test::column_wrapper<double> point_x_wrapp{rand_x};
  cudf::test::column_wrapper<double> point_y_wrapp{rand_y};
  cudf::test::column_wrapper<uint32_t> point_id_wrapp{rand_id};
  cudf::test::column_wrapper<its_timestamp> point_ts_wrapp{rand_ts};
  cudf::test::column_wrapper<uint32_t> ids_wrapp{1,2};
  
  gdf_column out_x,out_y,out_id,out_ts;
  memset(&out_x,0,sizeof(gdf_column));
  memset(&out_y,0,sizeof(gdf_column));
  memset(&out_id,0,sizeof(gdf_column));
  memset(&out_ts,0,sizeof(gdf_column));
  
  std::cout<<"calling cuspatial::subset_trajectory_id"<<std::endl;
  uint32_t num_hit=cuspatial::subset_trajectory_id(*(ids_wrapp.get()),
  	*(point_x_wrapp.get()), *(point_y_wrapp.get()),*(point_id_wrapp.get()),*(point_ts_wrapp.get()),
  	out_x,out_y,out_id,out_ts);
  
  std::cout<<"filtered trajectory point data"<<std::endl;
  int num_print = (num_hit<10)?num_hit:10;
  thrust::device_ptr<double> out_x_ptr=thrust::device_pointer_cast(static_cast<double*>(out_x.data));
  thrust::device_ptr<double> out_y_ptr=thrust::device_pointer_cast(static_cast<double*>(out_y.data));
  thrust::device_ptr<uint32_t> out_id_ptr=thrust::device_pointer_cast(static_cast<uint32_t*>(out_id.data));
  thrust::device_ptr<its_timestamp> out_ts_ptr=thrust::device_pointer_cast(static_cast<its_timestamp*>(out_ts.data));
  
  std::cout<<"x"<<std::endl;
  thrust::copy(out_x_ptr,out_x_ptr+num_print,std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl;  
  std::cout<<"y"<<std::endl;
  thrust::copy(out_y_ptr,out_y_ptr+num_print,std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl;  
  std::cout<<"id"<<std::endl;
  thrust::copy(out_id_ptr,out_id_ptr+num_print,std::ostream_iterator<uint32_t>(std::cout, " "));std::cout<<std::endl;  
  std::cout<<"timestamp"<<std::endl;
  thrust::copy(out_ts_ptr,out_ts_ptr+num_print,std::ostream_iterator<its_timestamp>(std::cout, " "));std::cout<<std::endl;
}
