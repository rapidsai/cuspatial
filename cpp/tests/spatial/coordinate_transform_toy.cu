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
#include <utilities/legacy/error_utils.hpp>
#include <cuspatial/coordinate_transform.hpp>
#include <utility/utility.hpp>

#include <tests/utilities/legacy/column_wrapper.cuh>
#include <tests/utilities/legacy/cudf_test_utils.cuh>
#include <tests/utilities/legacy/cudf_test_fixtures.h>

struct CoordinateTransToy : public GdfTest 
{
};   
   
TEST_F(CoordinateTransToy, coordinatetranstest)
{

  gdf_scalar x0; 
  x0.data.fp64=-90.66511046;
  x0.dtype=GDF_FLOAT64;
  x0.is_valid=true;
  gdf_scalar y0;
  y0.data.fp64=42.49197018;
  y0.dtype=GDF_FLOAT64;
  y0.is_valid=true;
  
  double point_lon[]={-90.664973,-90.665393,-90.664976,-90.664537};
  double point_lat[]={42.493894,42.491520,42.491420,42.493823};
  
  int num_point=sizeof(point_lon)/sizeof(double);
  std::vector<double> point_lon_vec(point_lon,point_lon+num_point);
  std::vector<double> point_lat_vec(point_lat,point_lat+num_point);
  std::cout<<"using camera origin ("<<x0.data.fp64<<","<<y0.data.fp64<<")"<<std::endl;
  std::cout<<"points before query:"<<std::endl;
  std::cout<<"lon:"<<std::endl;
  std::copy(point_lon_vec.begin(),point_lon_vec.end(),std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl; 
  std::cout<<"lat:"<<std::endl;
  std::copy(point_lat_vec.begin(),point_lat_vec.end(),std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl; 
  
  cudf::test::column_wrapper<double> point_lon_wrapp{point_lon_vec};
  cudf::test::column_wrapper<double> point_lat_wrapp{point_lat_vec};
  
  std::cout<<"calling cuspatial::spatial_window_points"<<std::endl;
  std::pair<gdf_column,gdf_column> res_pair=cuspatial::lonlat_to_coord(
	x0,y0,*(point_lon_wrapp.get()),*(point_lat_wrapp.get()));
  	
  thrust::device_ptr<double> out_x_ptr= thrust::device_pointer_cast(static_cast<double*>(res_pair.first.data));
  thrust::device_ptr<double> out_y_ptr= thrust::device_pointer_cast(static_cast<double*>(res_pair.second.data));
  int num_print=res_pair.first.size;
  std::cout<<"x:"<<std::endl;
  thrust::copy(out_x_ptr,out_x_ptr+num_print,std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl; 
  std::cout<<"y:"<<std::endl;
  thrust::copy(out_y_ptr,out_y_ptr+num_print,std::ostream_iterator<double>(std::cout, " "));std::cout<<std::endl; 
  
}
