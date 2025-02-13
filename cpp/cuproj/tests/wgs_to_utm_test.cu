/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <cuspatial_test/vector_equality.hpp>

#include <cuproj/error.hpp>
#include <cuproj/projection_factories.cuh>
#include <cuproj/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tabulate.h>

#include <gtest/gtest.h>
#include <proj.h>

#include <cmath>
#include <iostream>
#include <type_traits>

template <typename T>
struct ProjectionTest : public ::testing::Test {};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(ProjectionTest, TestTypes);

template <typename T>
using coordinate = typename cuspatial::vec_2d<T>;

enum class transform_call_type { HOST, DEVICE };

template <typename Coordinate, typename T = typename Coordinate::value_type>
__global__ void transform_kernel(cuproj::device_projection<Coordinate> const d_proj,
                                 Coordinate const* in,
                                 Coordinate* out,
                                 size_t n)
{
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    out[i] = d_proj.transform(in[i]);
  }
}

// run a test using the cuproj library
template <typename T>
void run_cuproj_test(thrust::host_vector<coordinate<T>> const& input,
                     thrust::host_vector<coordinate<T>> const& expected,
                     cuproj::projection<coordinate<T>> const& proj,
                     cuproj::direction dir,
                     T tolerance                   = T{0},  // 0 for 1-ulp comparison
                     transform_call_type call_type = transform_call_type::HOST)
{
  thrust::device_vector<coordinate<T>> d_in = input;
  thrust::device_vector<coordinate<T>> d_out(d_in.size());

  if (call_type == transform_call_type::DEVICE) {
    // Demonstrates how to call transform coordinates in device code
    // Note that this ultimately executes the same code as projection::transform(),
    // but in a device kernel. Both methods transform one coordinate per CUDA thread.
    // This gives more flexibility to write custom CUDA kernels that use the projection.
    auto d_proj            = proj.get_device_projection(dir);
    std::size_t block_size = 256;
    std::size_t grid_size  = (d_in.size() + block_size - 1) / block_size;
    transform_kernel<coordinate<T>>
      <<<grid_size, block_size>>>(d_proj, d_in.data().get(), d_out.data().get(), d_in.size());
    cudaDeviceSynchronize();
  } else {
    proj.transform(d_in.begin(), d_in.end(), d_out.begin(), dir);
  }

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
template <typename T, typename DeviceVector>
void run_forward_and_inverse(DeviceVector const& input,
                             T tolerance                   = T{0},
                             std::string const& utm_epsg   = "EPSG:32756",
                             transform_call_type call_type = transform_call_type::HOST)
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

  auto run = [&]() {
    run_proj_test(pj_expected, epsg_src, epsg_dst);

    thrust::host_vector<coordinate<T>> h_expected{pj_expected.size()};
    cuproj_test::convert_coordinates(pj_expected, h_expected);

    auto proj = cuproj::make_projection<coordinate<T>>(epsg_src, epsg_dst);

    run_cuproj_test<T>(
      h_input, h_expected, *proj, cuproj::direction::FORWARD, tolerance, call_type);
    run_cuproj_test<T>(
      h_expected, h_input, *proj, cuproj::direction::INVERSE, tolerance, call_type);

    delete proj;
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
  {  // auth:code strings
    cuproj::make_projection<coordinate<T>>("EPSG:4326", "EPSG:32756");
    cuproj::make_projection<coordinate<T>>("EPSG:32756", "EPSG:4326");
    cuproj::make_projection<coordinate<T>>("EPSG:4326", "EPSG:32601");
    cuproj::make_projection<coordinate<T>>("EPSG:32601", "EPSG:4326");
  }
  {  // int epsg codes
    cuproj::make_projection<coordinate<T>>(4326, 32756);
    cuproj::make_projection<coordinate<T>>(32756, 4326);
    cuproj::make_projection<coordinate<T>>(4326, 32601);
    cuproj::make_projection<coordinate<T>>(32601, 4326);
  }
}

// Test that construction of the projection from unsupported EPSG codes throws
// expected exceptions
TYPED_TEST(ProjectionTest, invalid_epsg)
{
  using T = TypeParam;
  {  // auth:code strings
    EXPECT_THROW(cuproj::make_projection<coordinate<T>>("EPSG:4326", "EPSG:756"),
                 cuproj::logic_error);
    EXPECT_THROW(cuproj::make_projection<coordinate<T>>("EPSG:4326", "UTM:32756"),
                 cuproj::logic_error);
    EXPECT_THROW(cuproj::make_projection<coordinate<T>>("EPSG:32611", "EPSG:32756"),
                 cuproj::logic_error);
  }
  {  // int codes
    EXPECT_THROW(cuproj::make_projection<coordinate<T>>(4326, 756), cuproj::logic_error);
    EXPECT_THROW(cuproj::make_projection<coordinate<T>>(32611, 32756), cuproj::logic_error);
  }
}

template <typename T>
void test_one(transform_call_type call_type = transform_call_type::HOST)
{
  coordinate<T> sydney{-33.858700, 151.214000};  // Sydney, NSW, Australia
  std::vector<coordinate<T>> input{sydney};
  // We can expect nanometer accuracy with double precision. The precision ratio of
  // double to single precision is 2^53 / 2^24 == 2^29 ~= 10^9, then we should
  // expect meter (10^9 nanometer) accuracy with single precision.
  T tolerance = std::is_same_v<T, double> ? T{1e-9} : T{1.0};
  run_forward_and_inverse<T>(input, tolerance, "EPSG:32756", call_type);
}

// Test on a single coordinate
TYPED_TEST(ProjectionTest, host_one)
{
  using T = TypeParam;

  test_one<T>(transform_call_type::HOST);
}

// Test on a single coordinate
TYPED_TEST(ProjectionTest, device_one)
{
  using T = TypeParam;

  test_one<T>(transform_call_type::DEVICE);
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

template <typename T>
void test_grids(int num_points_xy, transform_call_type call_type = transform_call_type::HOST)
{
  // Test with grids of coordinates covering various locations on the globe
  // Sydney Harbour
  {
    coordinate<T> min_corner{-33.9, 151.2};
    coordinate<T> max_corner{-33.7, 151.3};
    std::string epsg = "EPSG:32756";
    test_grid<T>(min_corner, max_corner, num_points_xy, epsg);
  }

  // London, UK
  {
    coordinate<T> min_corner{51.0, -1.0};
    coordinate<T> max_corner{52.0, 1.0};
    std::string epsg = "EPSG:32630";
    test_grid<T>(min_corner, max_corner, num_points_xy, epsg);
  }

  // Svalbard
  {
    coordinate<T> min_corner{77.0, 15.0};
    coordinate<T> max_corner{79.0, 20.0};
    std::string epsg = "EPSG:32633";
    test_grid<T>(min_corner, max_corner, num_points_xy, epsg);
  }

  // Ushuaia, Argentina
  {
    coordinate<T> min_corner{-55.0, -70.0};
    coordinate<T> max_corner{-53.0, -65.0};
    std::string epsg = "EPSG:32719";
    test_grid<T>(min_corner, max_corner, num_points_xy, epsg);
  }

  // McMurdo Station, Antarctica
  {
    coordinate<T> min_corner{-78.0, 165.0};
    coordinate<T> max_corner{-77.0, 170.0};
    std::string epsg = "EPSG:32706";
    test_grid<T>(min_corner, max_corner, num_points_xy, epsg);
  }

  // Singapore
  {
    coordinate<T> min_corner{1.0, 103.0};
    coordinate<T> max_corner{2.0, 104.0};
    std::string epsg = "EPSG:32648";
    test_grid<T>(min_corner, max_corner, num_points_xy, epsg);
  }
}

TYPED_TEST(ProjectionTest, host_many)
{
  int num_points_xy = 100;

  test_grids<TypeParam>(num_points_xy, transform_call_type::HOST);
}

TYPED_TEST(ProjectionTest, device_many)
{
  int num_points_xy = 100;

  test_grids<TypeParam>(num_points_xy, transform_call_type::DEVICE);
}

// Test the code in the readme
TYPED_TEST(ProjectionTest, readme_example)
{
  using T = TypeParam;

  // Make a projection to convert WGS84 (lat, lon) coordinates to UTM zone 56S (x, y) coordinates
  auto* proj = cuproj::make_projection<cuproj::vec_2d<T>>("EPSG:4326", "EPSG:32756");

  cuproj::vec_2d<T> sydney{-33.858700, 151.214000};  // Sydney, NSW, Australia
  thrust::device_vector<cuproj::vec_2d<T>> d_in{1, sydney};
  thrust::device_vector<cuproj::vec_2d<T>> d_out(d_in.size());

  // Convert the coordinates. Works the same with a vector of many coordinates.
  proj->transform(d_in.begin(), d_in.end(), d_out.begin(), cuproj::direction::FORWARD);
}

using device_projection = cuproj::device_projection<cuproj::vec_2d<float>>;

__global__ void example_kernel(device_projection const d_proj,
                               cuproj::vec_2d<float> const* in,
                               cuproj::vec_2d<float>* out,
                               size_t n)
{
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    out[i] = d_proj.transform(in[i]);
  }
}

TEST(ProjectionTest, device_readme_example)
{
  using coordinate = cuproj::vec_2d<float>;

  // Make a projection to convert WGS84 (lat, lon) coordinates to
  // UTM zone 56S (x, y) coordinates
  auto proj = cuproj::make_projection<coordinate>("EPSG:4326", "EPSG:32756");

  // Sydney, NSW, Australia
  coordinate sydney{-33.858700, 151.214000};
  thrust::device_vector<coordinate> d_in{1, sydney};
  thrust::device_vector<coordinate> d_out(d_in.size());

  auto d_proj            = proj->get_device_projection(cuproj::direction::FORWARD);
  std::size_t block_size = 256;
  std::size_t grid_size  = (d_in.size() + block_size - 1) / block_size;
  example_kernel<<<grid_size, block_size>>>(
    d_proj, d_in.data().get(), d_out.data().get(), d_in.size());
  cudaDeviceSynchronize();
}
