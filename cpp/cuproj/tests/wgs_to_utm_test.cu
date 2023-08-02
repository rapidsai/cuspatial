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

#include <cuproj_test/convert_coordinates.hpp>
#include <cuproj_test/coordinate_generator.cuh>

#include <cuproj/error.hpp>
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
#include <type_traits>

template <typename T>
struct ProjectionTest : public ::testing::Test {};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(ProjectionTest, TestTypes);

template <typename T>
using coordinate = typename cuspatial::vec_2d<T>;

// run a test using the cuproj library
template <typename T>
void run_cuproj_test(thrust::host_vector<coordinate<T>> const& input,
                     thrust::host_vector<coordinate<T>> const& expected,
                     cuproj::projection<coordinate<T>> const& proj,
                     cuproj::direction dir,
                     T tolerance = T{0})  // 0 for 1-ulp comparison
{
  thrust::device_vector<coordinate<T>> d_in = input;
  thrust::device_vector<coordinate<T>> d_out(d_in.size());

  proj.transform(d_in.begin(), d_in.end(), d_out.begin(), dir);

#ifndef NDEBUG
  std::cout << "expected " << std::setprecision(20) << expected[0].x << " " << expected[0].y
            << std::endl;
  coordinate<T> c_out = d_out[0];
  std::cout << "Device: " << std::setprecision(20) << c_out.x << " " << c_out.y << std::endl;
#endif

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected, d_out, tolerance);
}

// run a test using the proj library for comparison
void run_proj_test(thrust::host_vector<PJ_COORD>& coords,
                   char const* epsg_src,
                   char const* epsg_dst)
{
  PJ_CONTEXT* C = proj_context_create();
  PJ* P         = proj_create_crs_to_crs(C, epsg_src, epsg_dst, nullptr);
  proj_trans_array(P, PJ_FWD, coords.size(), coords.data());

  proj_destroy(P);
  proj_context_destroy(C);
}

// Run a test using the cuproj library in both directions, comparing to the proj library
template <typename T, typename DeviceVector, bool inverted = false>
void run_forward_and_inverse(DeviceVector const& input,
                             T tolerance                 = T{0},
                             std::string const& utm_epsg = "EPSG:32756")
{
  // note there are two notions of direction here. The direction of the construction of the
  // projection is determined by the order of the epsg strings. The direction of the transform is
  // determined by the direction argument to the transform method. This test runs both directions
  // for a single projection, with the order of construction determined by the inverted template
  // parameter. This is needed because a user may construct either a UTM->WGS84 or WGS84->UTM
  // projection, and we want to test both directions for each.
  thrust::host_vector<coordinate<T>> h_input(input.begin(), input.end());
  thrust::host_vector<PJ_COORD> pj_input{input.size()};
  cuproj_test::convert_coordinates(h_input, pj_input);
  thrust::host_vector<PJ_COORD> pj_expected(pj_input);

  char const* epsg_src = "EPSG:4326";
  char const* epsg_dst = utm_epsg.c_str();

  if constexpr (inverted) {}

  auto run = [&]() {
    run_proj_test(pj_expected, epsg_src, epsg_dst);

    thrust::host_vector<coordinate<T>> h_expected{pj_expected.size()};
    cuproj_test::convert_coordinates(pj_expected, h_expected);

    auto proj = cuproj::make_projection<coordinate<T>>(epsg_src, epsg_dst);

    run_cuproj_test(h_input, h_expected, proj, cuproj::direction::FORWARD, tolerance);
    run_cuproj_test(h_expected, h_input, proj, cuproj::direction::INVERSE, tolerance);
  };

  // forward construction
  run();
  // invert construction
  pj_input = pj_expected;
  cuproj_test::convert_coordinates(pj_input, h_input);
  std::swap(epsg_src, epsg_dst);
  run();
}

// Just test construction of the projection from supported EPSG codes
TYPED_TEST(ProjectionTest, make_projection_valid_epsg)
{
  using T = TypeParam;
  cuproj::make_projection<coordinate<T>>("EPSG:4326", "EPSG:32756");
  cuproj::make_projection<coordinate<T>>("EPSG:32756", "EPSG:4326");
  cuproj::make_projection<coordinate<T>>("EPSG:4326", "EPSG:32601");
  cuproj::make_projection<coordinate<T>>("EPSG:32601", "EPSG:4326");
}

// Test that construction of the projection from unsupported EPSG codes throws
// expected exceptions
TYPED_TEST(ProjectionTest, invalid_epsg)
{
  using T = TypeParam;
  EXPECT_THROW(cuproj::make_projection<coordinate<T>>("EPSG:4326", "EPSG:756"),
               cuproj::logic_error);
  EXPECT_THROW(cuproj::make_projection<coordinate<T>>("EPSG:4326", "UTM:32756"),
               cuproj::logic_error);
}

// Test on a single coordinate
TYPED_TEST(ProjectionTest, one)
{
  using T = TypeParam;

  coordinate<T> sydney{-33.865143, 151.209900};  // Sydney, NSW, Australia
  std::vector<coordinate<T>> input{sydney};
  // We can expect nanometer accuracy with double precision. The precision ratio of
  // double to single precision is 2^53 / 2^24 == 2^29 ~= 10^9, then we should
  // expect meter (10^9 nanometer) accuracy with single precision.
  T tolerance = std::is_same_v<T, float> ? T{1.0} : T{1e-9};
  run_forward_and_inverse<T>(input, tolerance, "EPSG:32756");
}

// Test on a grid of coordinates
template <typename T>
void test_grid(coordinate<T> const& min_corner,
               coordinate<T> max_corner,
               int num_points_xy,
               std::string const& utm_epsg)
{
  auto input = cuproj_test::make_grid_array<coordinate<T>, rmm::device_vector<coordinate<T>>>(
    min_corner, max_corner, num_points_xy, num_points_xy);

  thrust::host_vector<coordinate<T>> h_input(input);

  // We can expect nanometer accuracy with double precision. The precision ratio of
  // double to single precision is 2^53 / 2^24 == 2^29 ~= 10^9, then we should
  // expect meter (10^9 nanometer) accuracy with single precision.
  // For large arrays seem to need to relax the tolerance a bit to match PROJ results.
  // 1um for double and 10m for float seems like reasonable accuracy while not allowing excessive
  // variance from PROJ results.
  T tolerance = std::is_same_v<T, double> ? T{1e-6} : T{10};
  run_forward_and_inverse<T>(h_input, tolerance);
}

TYPED_TEST(ProjectionTest, many)
{
  int num_points_xy = 100;

  // Test with grids of coordinates covering various locations on the globe
  // Sydney Harbour
  {
    coordinate<TypeParam> min_corner{-33.9, 151.2};
    coordinate<TypeParam> max_corner{-33.7, 151.3};
    std::string epsg = "EPSG:32756";
    test_grid<TypeParam>(min_corner, max_corner, num_points_xy, epsg);
  }

  // London, UK
  {
    coordinate<TypeParam> min_corner{51.0, -1.0};
    coordinate<TypeParam> max_corner{52.0, 1.0};
    std::string epsg = "EPSG:32630";
    test_grid<TypeParam>(min_corner, max_corner, num_points_xy, epsg);
  }

  // Svalbard
  {
    coordinate<TypeParam> min_corner{77.0, 15.0};
    coordinate<TypeParam> max_corner{79.0, 20.0};
    std::string epsg = "EPSG:32633";
    test_grid<TypeParam>(min_corner, max_corner, num_points_xy, epsg);
  }

  // Ushuaia, Argentina
  {
    coordinate<TypeParam> min_corner{-55.0, -70.0};
    coordinate<TypeParam> max_corner{-53.0, -65.0};
    std::string epsg = "EPSG:32719";
    test_grid<TypeParam>(min_corner, max_corner, num_points_xy, epsg);
  }

  // McMurdo Station, Antarctica
  {
    coordinate<TypeParam> min_corner{-78.0, 165.0};
    coordinate<TypeParam> max_corner{-77.0, 170.0};
    std::string epsg = "EPSG:32706";
    test_grid<TypeParam>(min_corner, max_corner, num_points_xy, epsg);
  }

  // Singapore
  {
    coordinate<TypeParam> min_corner{1.0, 103.0};
    coordinate<TypeParam> max_corner{2.0, 104.0};
    std::string epsg = "EPSG:32648";
    test_grid<TypeParam>(min_corner, max_corner, num_points_xy, epsg);
  }
}
