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
#include <utilities/error_utils.hpp>
#include <cuspatial/hausdorff.hpp> 

#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/cudf_test_utils.cuh>
#include <tests/utilities/cudf_test_fixtures.h>

using namespace cuspatial;

struct HausdorffToy : public GdfTest 
{
};   
   
TEST_F(HausdorffToy, hausdorfftest)
{
  std::cout<<"in HausdorffToy"<<std::endl;
  cudf::test::column_wrapper<double> point_x_wrapp{0,-8,6};
  cudf::test::column_wrapper<double> point_y_wrapp{0,-8,6};
  cudf::test::column_wrapper<uint32_t> cnt_wrapp{1,2};
  gdf_column dist=cuspatial::directed_hausdorff_distance(
  	*(point_x_wrapp.get()), *(point_y_wrapp.get()),*(cnt_wrapp.get()));
  double *h_dist=new double[dist.size];
  CUDA_TRY(cudaMemcpy(h_dist, dist.data, dist.size*sizeof(double), cudaMemcpyDeviceToHost));
  CUDF_EXPECTS(h_dist[0]==0&&h_dist[3]==0,"distance between the same trajectoriy pair should be 0"); 
  std::cout<<"dist(0,1)="<<h_dist[1]<<std::endl;
  std::cout<<"dist(1,0)="<<h_dist[2]<<std::endl;
  delete[] h_dist;
}
