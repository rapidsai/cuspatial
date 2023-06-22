/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cuproj/projection_factories.hpp>

#include <cuspatial/geometry/vec_2d.hpp>

#include <cuspatial_test/vector_equality.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tabulate.h>

#include <proj.h>

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>

template <typename T>
struct ProjectionTest : public ::testing::Test {};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(ProjectionTest, TestTypes);

template <typename T>
using coordinate = typename cuspatial::vec_2d<T>;

template <typename T>
void run_test(thrust::host_vector<PJ_COORD> const& input_coords,
              thrust::host_vector<PJ_COORD> const& expected_coords,
              cuproj::direction dir,
              T tolerance = T{0})  // 0 for 1-ulp comparison
{
  std::vector<coordinate<T>> input(input_coords.size());
  std::vector<coordinate<T>> expected(expected_coords.size());
  std::transform(input_coords.begin(), input_coords.end(), input.begin(), [](auto const& c) {
    return coordinate<T>{static_cast<T>(c.xy.x), static_cast<T>(c.xy.y)};
  });
  std::transform(
    expected_coords.begin(), expected_coords.end(), expected.begin(), [](auto const& c) {
      return coordinate<T>{static_cast<T>(c.xy.x), static_cast<T>(c.xy.y)};
    });

  thrust::device_vector<coordinate<T>> d_in = input;
  thrust::device_vector<coordinate<T>> d_out(d_in.size());
  thrust::device_vector<coordinate<T>> d_expected = expected;

  auto utm_proj = make_utm_projection<coordinate<T>>(56, cuproj::hemisphere::SOUTH);

  utm_proj.transform(d_in.begin(), d_in.end(), d_out.begin(), dir);

#ifdef DEBUG
  std::cout << "expected " << std::setprecision(20) << expected_coords[0].xy.x << " "
            << expected_coords[0].xy.y << std::endl;
  coordinate<T> c_out = d_out[0];
  std::cout << "Device: " << std::setprecision(20) << c_out.x << " " << c_out.y << std::endl;
#endif

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_expected, d_out, tolerance);
}

template <typename T, typename HostVector>
void run_forward_and_inverse(HostVector const& input, T tolerance = T{0})
{
  thrust::host_vector<PJ_COORD> input_coords{input.size()};
  std::transform(input.begin(), input.end(), input_coords.begin(), [](coordinate<T> const& c) {
    return PJ_COORD{c.x, c.y, 0, 0};
  });
  thrust::host_vector<PJ_COORD> expected_coords(input_coords);

  // transform using PROJ
  {
    PJ_CONTEXT* C = proj_context_create();
    PJ* P         = proj_create_crs_to_crs(C, "EPSG:4326", "EPSG:32756", NULL);
    proj_trans_array(P, PJ_FWD, expected_coords.size(), expected_coords.data());

    proj_destroy(P);
    proj_context_destroy(C);
  }

  run_test(input_coords, expected_coords, cuproj::direction::FORWARD, tolerance);
}

TYPED_TEST(ProjectionTest, Test_forward_one)
{
  using T = TypeParam;
  std::vector<coordinate<T>> input{{-28.667003, 153.090959}};
  // We can expect nanometer accuracy with double precision. The precision ratio of
  // double to single precision is 2^53 / 2^24 == 2^29 ~= 10^9, then we should
  // expect meter (10^9 nanometer) accuracy with single precision.
  T tolerance = std::is_same_v<T, float> ? T{1.0} : T{1e-9};
  run_forward_and_inverse<T>(input, tolerance);
}

template <typename T>
struct grid_generator {
  coordinate<T> min_corner{};
  coordinate<T> max_corner{};
  coordinate<T> spacing{};
  int num_points_x{};
  int num_points_y{};

  grid_generator(coordinate<T> min_corner,
                 coordinate<T> max_corner,
                 int num_points_x,
                 int num_points_y)
    : min_corner(min_corner),
      max_corner(max_corner),
      num_points_x(num_points_x),
      num_points_y(num_points_y)
  {
    spacing = coordinate<T>{(max_corner.x - min_corner.x) / num_points_x,
                            (max_corner.y - min_corner.y) / num_points_y};
  }

  __device__ coordinate<T> operator()(int i) const
  {
    return min_corner +
           coordinate<T>{(i % num_points_x) * spacing.x, (i / num_points_x) * spacing.y};
  }
};

TYPED_TEST(ProjectionTest, test_many)
{
  using T = TypeParam;
  // generate (lat, lon) points on a grid between -60 and 60 degrees longitude and
  // -40 and 80 degrees latitude
  int num_points_x         = 100;
  int num_points_y         = 100;
  coordinate<T> min_corner = {-26.5, -152.5};
  coordinate<T> max_corner = {-25.5, -153.5};

  auto gen = grid_generator<T>(min_corner, max_corner, num_points_x, num_points_y);

  thrust::device_vector<coordinate<T>> input(num_points_x * num_points_y);

  thrust::tabulate(rmm::exec_policy(), input.begin(), input.end(), gen);

  thrust::host_vector<coordinate<T>> h_input = input;

  // We can expect nanometer accuracy with double precision. The precision ratio of
  // double to single precision is 2^53 / 2^24 == 2^29 ~= 10^9, then we should
  // expect meter (10^9 nanometer) accuracy with single precision.
  // For large arrays seem to need to relax the tolerance a bit (5nm and 10nm respectively)
  T tolerance = std::is_same_v<T, float> ? T{10.0} : T{5e-9};
  run_forward_and_inverse<T>(h_input, tolerance);
}
