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

#include "cuproj/error.hpp"
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
void run_cuproj_test(thrust::host_vector<PJ_COORD> const& input_coords,
                     thrust::host_vector<PJ_COORD> const& expected_coords,
                     cuproj::projection<coordinate<T>> const& proj,
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

  proj.transform(d_in.begin(), d_in.end(), d_out.begin(), dir);

#ifdef DEBUG
  std::cout << "expected " << std::setprecision(20) << expected_coords[0].xy.x << " "
            << expected_coords[0].xy.y << std::endl;
  coordinate<T> c_out = d_out[0];
  std::cout << "Device: " << std::setprecision(20) << c_out.x << " " << c_out.y << std::endl;
#endif

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_expected, d_out, tolerance);
}

void run_proj_test(thrust::host_vector<PJ_COORD> const& input_coords,
                   thrust::host_vector<PJ_COORD>& expected_coords,
                   char const* epsg_src,
                   char const* epsg_dst)
{
  PJ_CONTEXT* C = proj_context_create();
  PJ* P         = proj_create_crs_to_crs(C, epsg_src, epsg_dst, nullptr);
  proj_trans_array(P, PJ_FWD, expected_coords.size(), expected_coords.data());

  proj_destroy(P);
  proj_context_destroy(C);
}

template <typename T, typename HostVector, bool inverted = false>
void run_forward_and_inverse(HostVector const& input, T tolerance = T{0})
{
  // note there are two notions of direction here. The direction of the construction of the
  // projection is determined by the order of the epsg strings. The direction of the transform is
  // determined by the direction argument to the transform method. This test runs both directions
  // for a single projection, with the order of construction determined by the inverted template
  // parameter. This is needed because a user may construct either a UTM->WGS84 or WGS84->UTM
  // projection, and we want to test both directions for each.

  thrust::host_vector<PJ_COORD> input_coords{input.size()};
  std::transform(input.begin(), input.end(), input_coords.begin(), [](coordinate<T> const& c) {
    return PJ_COORD{c.x, c.y, 0, 0};
  });
  thrust::host_vector<PJ_COORD> expected_coords(input_coords);

  char const* epsg_src = "EPSG:4326";
  char const* epsg_dst = "EPSG:32756";

  if constexpr (inverted) {}

  auto run = [&]() {
    run_proj_test(input_coords, expected_coords, epsg_src, epsg_dst);

    auto proj = cuproj::make_projection<coordinate<T>>(epsg_src, epsg_dst);

    run_cuproj_test(input_coords, expected_coords, proj, cuproj::direction::FORWARD, tolerance);
    run_cuproj_test(expected_coords, input_coords, proj, cuproj::direction::INVERSE, tolerance);
  };

  // forward construction
  run();
  // invert construction
  std::swap(input_coords, expected_coords);
  std::swap(epsg_src, epsg_dst);
  run();
}

TYPED_TEST(ProjectionTest, make_projection_valid_epsg)
{
  using T = TypeParam;
  cuproj::make_projection<coordinate<T>>("EPSG:4326", "EPSG:32756");
  cuproj::make_projection<coordinate<T>>("EPSG:32756", "EPSG:4326");
  cuproj::make_projection<coordinate<T>>("EPSG:4326", "EPSG:32601");
  cuproj::make_projection<coordinate<T>>("EPSG:32601", "EPSG:4326");
}

TYPED_TEST(ProjectionTest, invalid_epsg)
{
  using T = TypeParam;
  EXPECT_THROW(cuproj::make_projection<coordinate<T>>("EPSG:4326", "EPSG:756"),
               cuproj::logic_error);
  EXPECT_THROW(cuproj::make_projection<coordinate<T>>("EPSG:32756", "EPSG:4326"),
               cuproj::logic_error);
  EXPECT_THROW(cuproj::make_projection<coordinate<T>>("EPSG:4326", "UTM:32756"),
               cuproj::logic_error);
}

TYPED_TEST(ProjectionTest, one)
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

TYPED_TEST(ProjectionTest, many)
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
