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

#include <cuproj/ellipsoid.hpp>
#include <cuproj/error.hpp>
#include <cuproj/transform.cuh>

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

TYPED_TEST(ProjectionTest, Test_forward_one)
{
  PJ_CONTEXT* C;
  PJ* P;

  C = proj_context_create();

  P = proj_create_crs_to_crs(C, "EPSG:4326", "EPSG:32756", NULL);

  double ellps_a{};
  double ellps_b{};
  int is_semi_minor_computed{};
  double ellps_inv_flattening{};
  auto wgs84   = proj_create(C, "EPSG:4326");
  PJ* pj_ellps = proj_get_ellipsoid(C, wgs84);
  proj_ellipsoid_get_parameters(
    C, pj_ellps, &ellps_a, &ellps_b, &is_semi_minor_computed, &ellps_inv_flattening);
  proj_destroy(pj_ellps);

  using T = TypeParam;

  std::vector<PJ_COORD> input_coords{{-28.667003, 153.090959, 0, 0}};
  std::vector<PJ_COORD> expected_coords(input_coords);

  proj_trans_array(P, PJ_FWD, expected_coords.size(), expected_coords.data());

  /* Clean up */
  proj_destroy(P);
  proj_context_destroy(C);  // may be omitted in the single threaded case

  // semimajor and inverse flattening
  cuproj::ellipsoid<T> ellps{static_cast<T>(ellps_a), static_cast<T>(ellps_inv_flattening)};

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

  cuproj::projection<T> tmerc_proj{ellps, 56, 0, 0};

  cuproj::transform(
    tmerc_proj, d_in.begin(), d_in.end(), d_out.begin(), cuproj::direction::DIR_FWD);

#ifdef DEBUG
  std::cout << "expected " << std::setprecision(20) << expected_coords[0].xy.x << " "
            << expected_coords[0].xy.y << std::endl;
  coordinate<T> c_out = d_out[0];
  std::cout << "Device: " << std::setprecision(20) << c_out.x << " " << c_out.y << std::endl;
#endif

  // We can expect nanometer accuracy with double precision. The precision ratio of
  // double to single precision is 2^53 / 2^24 == 2^29 ~= 10^9, then we should
  // expect meter (10^9 nanometer) accuracy with single precision.
  if constexpr (std::is_same_v<T, float>) {
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_expected, d_out, T{1.0});  // within 1 meter
  } else {
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_expected, d_out);  // just use normal 1-ulp comparison
  }
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

TYPED_TEST(ProjectionTest, Test_forward_many)
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

  // create PROJ context
  PJ_CONTEXT* C = proj_context_create();

  double ellps_a{};
  double ellps_b{};
  int is_semi_minor_computed{};
  double ellps_inv_flattening{};
  auto wgs84   = proj_create(C, "EPSG:4326");
  PJ* pj_ellps = proj_get_ellipsoid(C, wgs84);
  proj_ellipsoid_get_parameters(
    C, pj_ellps, &ellps_a, &ellps_b, &is_semi_minor_computed, &ellps_inv_flattening);
  proj_destroy(pj_ellps);

  // semimajor and inverse flattening
  cuproj::ellipsoid<T> ellps{static_cast<T>(ellps_a), static_cast<T>(ellps_inv_flattening)};

  // create a projection object
  cuproj::projection<T> tmerc_proj{ellps, 56};
  // create a vector of output points
  thrust::device_vector<coordinate<T>> output(input.size());
  // transform the input points to output points
  cuproj::transform(
    tmerc_proj, input.begin(), input.end(), output.begin(), cuproj::direction::DIR_FWD);

  using T = TypeParam;

  thrust::host_vector<PJ_COORD> input_coords{input.size()};
  std::transform(input.begin(), input.end(), input_coords.begin(), [](coordinate<T> const& c) {
    return PJ_COORD{c.x, c.y, 0, 0};
  });
  thrust::host_vector<PJ_COORD> expected_coords(input_coords);

  PJ* P = proj_create_crs_to_crs(C, "EPSG:4326", "EPSG:32756", NULL);

  proj_trans_array(P, PJ_FWD, expected_coords.size(), expected_coords.data());

  proj_destroy(P);
  proj_context_destroy(C);

  std::vector<coordinate<T>> expected(expected_coords.size());
  std::transform(
    expected_coords.begin(), expected_coords.end(), expected.begin(), [](auto const& c) {
      return coordinate<T>{static_cast<T>(c.xy.x), static_cast<T>(c.xy.y)};
    });

  thrust::device_vector<coordinate<T>> d_expected = expected;

#ifdef DEBUG
  std::cout << "expected " << std::setprecision(20) << expected_coords[0].xy.x << " "
            << expected_coords[0].xy.y << std::endl;
  coordinate<T> c_out = output[0];
  std::cout << "Device: " << std::setprecision(20) << c_out.x << " " << c_out.y << std::endl;
#endif

  // Assumption: we can expect 5 nanometer (5e-9m) accuracy with double precision. The precision
  // ratio of double to single precision is 2^53 / 2^24 == 2^29 ~= 10^9, so we should expect 5 meter
  // (5e9 nanometer) accuracy with single precision. However we are seeing 10 meter accuracy
  // relative to PROJ with single precision (which uses double precision internally)

  // TODO: can we use double precision for key parts of the algorithm for accuracy while
  // using single precision for the rest for performance?
  if constexpr (std::is_same_v<T, float>) {
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_expected, output, T{10.0});  // within 10m
  } else {
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_expected, output, T{5e-9});  // within 5nm
  }
}
