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

#include <gtest/gtest.h>
#include <cuspatial/haversine.hpp> 
#include <utilities/legacy/error_utils.hpp>

#include <tests/utilities/legacy/cudf_test_utils.cuh>
#include <tests/utilities/legacy/column_wrapper.cuh>
#include <tests/utilities/legacy/cudf_test_fixtures.h>

struct HaversineToy : public GdfTest 
{
};   
   
TEST_F(HaversineToy, haversinetest)
{
  const uint32_t num_point=3;
  double h_point_x[]={ -74.0060, 2.3522,151.2093};
  double h_point_y[]={40.7128,48.8566,-33.8688};
  const char *cities[]={"New York","Paris","Sydney"};
  double *h_pair_x1=new double[num_point*num_point];
  double *h_pair_y1=new double[num_point*num_point];
  double *h_pair_x2=new double[num_point*num_point];
  double *h_pair_y2=new double[num_point*num_point];
  
  CUDF_EXPECTS(h_pair_x1!=nullptr&&h_pair_y1!=nullptr&&h_pair_x2!=nullptr&&h_pair_y2!=nullptr,
  	"invalid point pair x/y arrays");
  
  for(size_t i=0; i<num_point;i++)
      for(size_t j=0; j<num_point;j++)
      {
          h_pair_x1[i*num_point+j]=h_point_x[i];
          h_pair_y1[i*num_point+j]=h_point_y[i];
          h_pair_x2[i*num_point+j]=h_point_x[j];
          h_pair_y2[i*num_point+j]=h_point_y[j];
      }
  	      
  cudf::test::column_wrapper<double> point_x1_wrapp{std::vector<double>(h_pair_x1,h_pair_x1+num_point*num_point)};
  cudf::test::column_wrapper<double> point_y1_wrapp{std::vector<double>(h_pair_y1,h_pair_y1+num_point*num_point)};
  cudf::test::column_wrapper<double> point_x2_wrapp{std::vector<double>(h_pair_x2,h_pair_x2+num_point*num_point)};
  cudf::test::column_wrapper<double> point_y2_wrapp{std::vector<double>(h_pair_y2,h_pair_y2+num_point*num_point)};
  
  gdf_column dist=cuspatial::haversine_distance(
  	*(point_x1_wrapp.get()), *(point_y1_wrapp.get()),*(point_x2_wrapp.get()),*(point_y2_wrapp.get()));
  double *h_dist=new double[dist.size];
  CUDA_TRY(cudaMemcpy(h_dist, dist.data, dist.size*sizeof(double), cudaMemcpyDeviceToHost));
  
  CUDF_EXPECTS(fabs(h_dist[0])<1e-10&&fabs(h_dist[4])<1e-10&&fabs(h_dist[8])<1e-10,
  	"distance between the same points should be close to 0"); 
  
  std::cout<<"dist("<<cities[0]<<","<<cities[1]<<")="<<h_dist[1]<<std::endl;
  std::cout<<"dist("<<cities[0]<<","<<cities[2]<<")="<<h_dist[2]<<std::endl;
  std::cout<<"dist("<<cities[1]<<","<<cities[0]<<")="<<h_dist[3]<<std::endl;
  std::cout<<"dist("<<cities[1]<<","<<cities[2]<<")="<<h_dist[5]<<std::endl;
  std::cout<<"dist("<<cities[2]<<","<<cities[0]<<")="<<h_dist[6]<<std::endl;
  std::cout<<"dist("<<cities[2]<<","<<cities[1]<<")="<<h_dist[7]<<std::endl;
    
  delete[] h_pair_x1;
  delete[] h_pair_y1;
  delete[] h_pair_x2;
  delete[] h_pair_y2;
  delete[] h_dist;
}
