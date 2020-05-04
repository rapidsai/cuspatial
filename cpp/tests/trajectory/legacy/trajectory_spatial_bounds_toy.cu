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

#include <sys/time.h>
#include <time.h>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <cuspatial/legacy/trajectory.hpp>
#include <cuspatial/types.hpp>
#include <utilities/legacy/error_utils.hpp>
#include <utility/trajectory_thrust.cuh>
#include <utility/utility.hpp>

#include <tests/utilities/legacy/cudf_test_fixtures.h>
#include <tests/utilities/legacy/column_wrapper.cuh>
#include <tests/utilities/legacy/cudf_test_utils.cuh>

struct TrajectoryDeriveToy : public GdfTest {
};

TEST_F(TrajectoryDeriveToy, trajectoryderivetest)
{
  // three sorted trajectories with 5,4,3 points, respectively
  std::cout << "in TrajectoryDeriveToy" << std::endl;
  // assuming x/y are in the unit of killometers (km);
  // computed distance and speed are in the units of meters and m/s, respectively
  double point_x[]       = {1.0, 2.0, 3.0, 5.0, 7.0, 1.0, 2.0, 3.0, 6.0, 0.0, 3.0, 6.0};
  double point_y[]       = {0.0, 1.0, 2.0, 3.0, 1.0, 3.0, 5.0, 6.0, 5.0, 4.0, 7.0, 4.0};
  uint32_t traj_len[]    = {5, 4, 3};
  uint32_t traj_offset[] = {5, 9, 12};
  size_t num_point       = sizeof(point_x) / sizeof(double);
  size_t num_traj        = sizeof(traj_len) / sizeof(uint32_t);

  cudf::test::column_wrapper<double> point_x_wrapp{
    std::vector<double>(point_x, point_x + num_point)};
  cudf::test::column_wrapper<double> point_y_wrapp{
    std::vector<double>(point_y, point_y + num_point)};
  cudf::test::column_wrapper<uint32_t> traj_len_wrapp{
    std::vector<uint32_t>(traj_len, traj_len + num_traj)};
  cudf::test::column_wrapper<uint32_t> traj_pos_wrapp{
    std::vector<uint32_t>(traj_offset, traj_offset + num_traj)};

  gdf_column x1, x2, y1, y2;
  memset(&x1, 0, sizeof(gdf_column));
  memset(&x2, 0, sizeof(gdf_column));
  memset(&y1, 0, sizeof(gdf_column));
  memset(&y2, 0, sizeof(gdf_column));

  std::cout << "calling cuspatial::trajectory_spatial_bounds" << std::endl;
  cuspatial::trajectory_spatial_bounds(*(point_x_wrapp.get()),
                                       *(point_y_wrapp.get()),
                                       *(traj_len_wrapp.get()),
                                       *(traj_pos_wrapp.get()),
                                       x1,
                                       y1,
                                       x2,
                                       y2);

  std::cout << "computed bounding boxes (x1,y1,x2,y2)" << std::endl;
  int num_print                     = (num_traj < 10) ? num_traj : 10;
  thrust::device_ptr<double> x1_ptr = thrust::device_pointer_cast(static_cast<double*>(x1.data));
  thrust::device_ptr<double> y1_ptr = thrust::device_pointer_cast(static_cast<double*>(y1.data));
  thrust::device_ptr<double> x2_ptr = thrust::device_pointer_cast(static_cast<double*>(x2.data));
  thrust::device_ptr<double> y2_ptr = thrust::device_pointer_cast(static_cast<double*>(y2.data));

  std::cout << "x1:" << std::endl;
  thrust::copy(x1_ptr, x1_ptr + num_print, std::ostream_iterator<double>(std::cout, " "));
  std::cout << std::endl;
  std::cout << "y1:" << std::endl;
  thrust::copy(y1_ptr, y1_ptr + num_print, std::ostream_iterator<double>(std::cout, " "));
  std::cout << std::endl;
  std::cout << "x2:" << std::endl;
  thrust::copy(x2_ptr, x2_ptr + num_print, std::ostream_iterator<double>(std::cout, " "));
  std::cout << std::endl;
  std::cout << "x2:" << std::endl;
  thrust::copy(y2_ptr, y2_ptr + num_print, std::ostream_iterator<double>(std::cout, " "));
  std::cout << std::endl;
}
