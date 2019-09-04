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
#include <cuspatial/query.hpp>
#include <utility/utility.hpp>

#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/cudf_test_utils.cuh>
#include <tests/utilities/cudf_test_fixtures.h>

struct TrajectoryDeriveToy : public GdfTest 
{
};   
   
TEST_F(TrajectoryDeriveToy, trajectoryderivetest)
{
  //three sorted trajectories with 5,4,3 points, respectively
  std::cout<<"in TrajectoryDeriveToy"<<std::endl;
  //assuming x/y are in the unit of killometers (km); 
  //computed distance and speed are in the units of meters and m/s, respectively
  double point_x[]={1.0,2.0,3.0,5.0,7.0,1.0,2.0,3.0,6.0,0.0,3.0,6.0};
  double point_y[]={0.0,1.0,2.0,3.0,1.0,3.0,5.0,6.0,5.0,4.0,7.0,4.0};
  
  int num_point=sizeof(point_x)/sizeof(double);
  std::vector<double> point_x_vec(point_x,point_x+num_point);
  std::vector<double> point_y_vec(point_y,point_y+num_point);
  std::cout<<"points before query:"<<std::endl;
  std::cout<<"x:"<<std::endl;
  std::copy(point_x_vec.begin(),point_x_vec.end(),std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl; 
  std::cout<<"y:"<<std::endl;
  std::copy(point_y_vec.begin(),point_y_vec.end(),std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl; 
  
  cudf::test::column_wrapper<double> point_x_wrapp{point_x_vec};
  cudf::test::column_wrapper<double> point_y_wrapp{point_y_vec};
  
  
  gdf_scalar x1; 
  x1.data.fp64=1.5;
  x1.dtype=GDF_FLOAT64;
  x1.is_valid=true;
  gdf_scalar y1;
  y1.data.fp64=1.5;
  y1.dtype=GDF_FLOAT64;
  y1.is_valid=true;
  gdf_scalar x2;
  x2.data.fp64=5.5;
  x2.dtype=GDF_FLOAT64;
  x2.is_valid=true;
  gdf_scalar y2;
  y2.data.fp64=5.5;
  y2.dtype=GDF_FLOAT64;
  y2.is_valid=true;
  
  std::cout<<"calling cuspatial::spatial_window_points"<<std::endl;
  std::pair<gdf_column,gdf_column> res_pair=cuspatial::spatial_window_points(
	x1,y1,x2,y2,*(point_x_wrapp.get()),*(point_y_wrapp.get()));
  	
  std::cout<<"points within query window(" <<x1.data.fp64 <<"," <<y1.data.fp64 <<"," << x2.data.fp64 <<"," <<y2.data.fp64 <<")"<<std::endl;
  thrust::device_ptr<double> out_x_ptr= thrust::device_pointer_cast(static_cast<double*>(res_pair.first.data));
  thrust::device_ptr<double> out_y_ptr= thrust::device_pointer_cast(static_cast<double*>(res_pair.second.data));
  int num_print=res_pair.first.size;
  std::cout<<"x:"<<std::endl;
  thrust::copy(out_x_ptr,out_x_ptr+num_print,std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl; 
  std::cout<<"y:"<<std::endl;
  thrust::copy(out_y_ptr,out_y_ptr+num_print,std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl; 
  
}
