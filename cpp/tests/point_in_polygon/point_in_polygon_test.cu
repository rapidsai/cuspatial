/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cuspatial_test/base_fixture.hpp>
#include <cuspatial_test/vector_equality.hpp>
#include <cuspatial_test/vector_factories.cuh>

#include <cuspatial/error.hpp>
#include <cuspatial/geometry/vec_2d.hpp>
#include <cuspatial/iterator_factory.cuh>
#include <cuspatial/point_in_polygon.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>

#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <gtest/gtest.h>

using namespace cuspatial;
using namespace cuspatial::test;

template <typename T>
struct PointInPolygonTest : public BaseFixture {
 public:
  void run_test(std::initializer_list<vec_2d<T>> points,
                std::initializer_list<int> polygon_offsets,
                std::initializer_list<int> ring_offsets,
                std::initializer_list<vec_2d<T>> polygon_points,
                std::initializer_list<int32_t> expected)
  {
    auto d_points          = make_device_vector<vec_2d<T>>(points);
    auto d_polygon_offsets = make_device_vector<int>(polygon_offsets);
    auto d_ring_offsets    = make_device_vector<int>(ring_offsets);
    auto d_polygon_points  = make_device_vector<vec_2d<T>>(polygon_points);

    auto mpoints = make_multipoint_range(
      d_points.size(), thrust::make_counting_iterator(0), d_points.size(), d_points.begin());
    auto mpolys = make_multipolygon_range(polygon_offsets.size() - 1,
                                          thrust::make_counting_iterator(0),
                                          d_polygon_offsets.size() - 1,
                                          d_polygon_offsets.begin(),
                                          d_ring_offsets.size() - 1,
                                          d_ring_offsets.begin(),
                                          d_polygon_points.size(),
                                          d_polygon_points.begin());

    auto d_expected = make_device_vector(expected);

    auto got = rmm::device_uvector<int32_t>(points.size(), stream());

    auto ret = point_in_polygon(mpoints, mpolys, got.begin(), stream());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_expected, got);
    EXPECT_EQ(ret, got.end());
  }

  void run_spherical_test(std::initializer_list<vec_3d<T>> points,
                          std::initializer_list<int> polygon_offsets,
                          std::initializer_list<int> ring_offsets,
                          std::initializer_list<vec_3d<T>> polygon_points,
                          std::initializer_list<int32_t> expected)
  {
    auto d_points          = make_device_vector<vec_3d<T>>(points);
    auto d_polygon_offsets = make_device_vector<int>(polygon_offsets);
    auto d_ring_offsets    = make_device_vector<int>(ring_offsets);
    auto d_polygon_points  = make_device_vector<vec_3d<T>>(polygon_points);

    auto mpoints = make_multipoint_range(
      d_points.size(), thrust::make_counting_iterator(0), d_points.size(), d_points.begin());
    auto mpolys = make_multipolygon_range(polygon_offsets.size() - 1,
                                          thrust::make_counting_iterator(0),
                                          d_polygon_offsets.size() - 1,
                                          d_polygon_offsets.begin(),
                                          d_ring_offsets.size() - 1,
                                          d_ring_offsets.begin(),
                                          d_polygon_points.size(),
                                          d_polygon_points.begin());

    auto d_expected = make_device_vector(expected);

    auto got = rmm::device_uvector<int32_t>(points.size(), stream());

    auto ret = point_in_polygon(mpoints, mpolys, got.begin(), stream());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_expected, got);
    EXPECT_EQ(ret, got.end());
  }
};

TYPED_TEST_CASE(PointInPolygonTest, FloatingPointTypes);

TYPED_TEST(PointInPolygonTest, OnePolygonOneRing)
{
  CUSPATIAL_RUN_TEST(this->run_test,
                     {{-2.0, 0.0},
                      {2.0, 0.0},
                      {0.0, -2.0},
                      {0.0, 2.0},
                      {-0.5, 0.0},
                      {0.5, 0.0},
                      {0.0, -0.5},
                      {0.0, 0.5}},
                     {0, 1},
                     {0, 5},
                     {{-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}, {-1.0, -1.0}},
                     {false, false, false, false, true, true, true, true});
}

TYPED_TEST(PointInPolygonTest, OnePolygonOneRingSpherical)
{
  CUSPATIAL_RUN_TEST(this->run_spherical_test,
                     {{-2503.357, -4660.203, 3551.245},
                      {-2503.357, -4660.203, 3551.245},
                      {-2686.757, -4312.736, 3842.237},
                      {-2684.959, -4312.568, 3843.673},
                      {519.181, -5283.34, 3523.313}},
                     {0, 1},
                     {0, 5},
                     {{-2681.925, -4311.158, 3847.346},   // San Jose
                      {-2695.156, -4299.131, 3851.527},   // MTV
                      {-2691.386, -4313.414, 3838.26},    // Los Gatos
                      {-2673.88, -4319.257, 3843.883},    // East San Jose
                      {-2681.925, -4311.158, 3847.346}},  // San Jose
                     {false, false, true, true, false});
}

// cuspatial expects closed rings, however algorithms may work OK with unclosed rings
// in the future if we change to an algorithm that requires closed rings we may change or remove
// this test.
// Note that we don't introspect the values of offset arrays to validate closedness. So this test
// uses a polygon ring with 4 vertices so it doesn't fail polygon validation.
TYPED_TEST(PointInPolygonTest, OnePolygonOneRingUnclosed)
{
  CUSPATIAL_RUN_TEST(this->run_test,
                     {{-2.0, 0.0},
                      {2.0, 0.0},
                      {0.0, -2.0},
                      {0.0, 2.0},
                      {-0.5, 0.0},
                      {0.5, 0.0},
                      {0.0, -0.5},
                      {0.0, 0.5}},
                     {0, 1},
                     {0, 4},
                     {{-1.0, -1.0}, {1.0, -1.0}, {1.0, 0.0}, {1.0, 1.0}},
                     {false, false, false, false, false, true, true, false});
}

TYPED_TEST(PointInPolygonTest, OnePolygonOneRingUnclosedSpherical)
{
  CUSPATIAL_RUN_TEST(this->run_spherical_test,
                     {{-2503.357, -4660.203, 3551.245},
                      {-2503.357, -4660.203, 3551.245},
                      {-2686.757, -4312.736, 3842.237},
                      {-2684.959, -4312.568, 3843.673},
                      {519.181, -5283.34, 3523.313}},
                     {0, 1},
                     {0, 4},
                     {{-2681.925, -4311.158, 3847.346},  // San Jose
                      {-2695.156, -4299.131, 3851.527},  // MTV
                      {-2691.386, -4313.414, 3838.26},   // Los Gatos
                      {-2673.88, -4319.257, 3843.883}},  // East San Jose
                     {false, false, true, true, false});
}

TYPED_TEST(PointInPolygonTest, TwoPolygonsOneRingEach)
{
  CUSPATIAL_RUN_TEST(this->run_test,

                     {{-2.0, 0.0},
                      {2.0, 0.0},
                      {0.0, -2.0},
                      {0.0, 2.0},
                      {-0.5, 0.0},
                      {0.5, 0.0},
                      {0.0, -0.5},
                      {0.0, 0.5}},

                     {0, 1, 2},
                     {0, 5, 10},
                     {{-1.0, -1.0},
                      {-1.0, 1.0},
                      {1.0, 1.0},
                      {1.0, -1.0},
                      {-1.0, -1.0},
                      {0.0, 1.0},
                      {1.0, 0.0},
                      {0.0, -1.0},
                      {-1.0, 0.0},
                      {0.0, 1.0}},

                     {0b00, 0b00, 0b00, 0b00, 0b11, 0b11, 0b11, 0b11});
}

TYPED_TEST(PointInPolygonTest, TwoPolygonsOneRingEachSpherical)
{
  CUSPATIAL_RUN_TEST(this->run_spherical_test,
                     {{-2503.357, -4660.203, 3551.245},
                      {-2503.357, -4660.203, 3551.245},
                      {-2686.757, -4312.736, 3842.237},
                      {-2684.959, -4312.568, 3843.673},
                      {519.181, -5283.34, 3523.313}},
                     {0, 1, 2},
                     {0, 5, 10},
                     {{-2681.925, -4311.158, 3847.346},  // San Jose
                      {-2695.156, -4299.131, 3851.527},  // MTV
                      {-2691.386, -4313.414, 3838.26},   // Los Gatos
                      {-2673.88, -4319.257, 3843.883},   // East San Jose
                      {-2681.925, -4311.158, 3847.346},  // San Jose
                      {-2691.386, -4313.414, 3838.26},   // Los Gatos
                      {-2673.88, -4319.257, 3843.883},   // East San Jose
                      {-2681.925, -4311.158, 3847.346},  // San Jose
                      {-2695.156, -4299.131, 3851.527},  // MTV
                      {-2691.386, -4313.414, 3838.26}},  // Los Gatos
                     {0b00, 0b00, 0b11, 0b11, 0b00});
}

TYPED_TEST(PointInPolygonTest, OnePolygonTwoRings)
{
  CUSPATIAL_RUN_TEST(this->run_test,
                     {{0.0, 0.0}, {-0.4, 0.0}, {-0.6, 0.0}, {0.0, 0.4}, {0.0, -0.6}},
                     {0, 2},
                     {0, 5, 10},
                     {{-1.0, -1.0},
                      {1.0, -1.0},
                      {1.0, 1.0},
                      {-1.0, 1.0},
                      {-1.0, -1.0},
                      {-0.5, -0.5},
                      {-0.5, 0.5},
                      {0.5, 0.5},
                      {0.5, -0.5},
                      {-0.5, -0.5}},

                     {0b0, 0b0, 0b1, 0b0, 0b1});
}

TYPED_TEST(PointInPolygonTest, OnePolygonTwoRingsSpherical)
{
  CUSPATIAL_RUN_TEST(this->run_spherical_test,
                     {{-2503.357, -4660.203, 3551.245},
                      {-2503.357, -4660.203, 3551.245},
                      {-2686.757, -4312.736, 3842.237},
                      {-2684.959, -4312.568, 3843.673},
                      {519.181, -5283.34, 3523.313}},
                     {0, 2},
                     {0, 5, 10},
                     {
                       {-2867.9, -3750.684, 4273.764},
                       {-3346.365, -4376.427, 3203.156},
                       {-2010.426, -5129.275, 3203.156},
                       {-1722.974, -4395.889, 4273.764},
                       {-2867.9, -3750.684, 4273.764},
                       {-2681.925, -4311.158, 3847.346},  // San Jose
                       {-2673.88, -4319.257, 3843.883},   // East San Jose
                       {-2691.386, -4313.414, 3838.26},   // Los Gatos
                       {-2695.156, -4299.131, 3851.527},  // MTV
                       {-2681.925, -4311.158, 3847.346},  // San Jose
                     },
                     {0b1, 0b1, 0b0, 0b0, 0b0});
}

TYPED_TEST(PointInPolygonTest, EdgesOfSquare)
{
  CUSPATIAL_RUN_TEST(
    this->run_test,
    {{0.0, 0.0}},
    {0, 1, 2, 3, 4},
    {0, 5, 10, 15, 20},

    // 0: rect on min x side
    // 1: rect on max x side
    // 2: rect on min y side
    // 3: rect on max y side
    {{-1.0, -1.0}, {0.0, -1.0}, {0.0, 1.0},  {-1.0, 1.0},  {-1.0, -1.0}, {0.0, -1.0}, {1.0, -1.0},
     {1.0, 1.0},   {0.0, 1.0},  {0.0, -1.0}, {-1.0, -1.0}, {-1.0, 0.0},  {1.0, 0.0},  {1.0, -1.0},
     {-1.0, 1.0},  {-1.0, 0.0}, {-1.0, 1.0}, {1.0, 1.0},   {1.0, 0.0},   {-1.0, 0.0}},

    {0b0000});
}

TYPED_TEST(PointInPolygonTest, CornersOfSquare)
{
  CUSPATIAL_RUN_TEST(
    this->run_test,
    {{0.0, 0.0}},
    {0, 1, 2, 3, 4},
    {0, 5, 10, 15, 20},

    // 0: min x min y corner
    // 1: min x max y corner
    // 2: max x min y corner
    // 3: max x max y corner
    {{-1.0, -1.0}, {-1.0, 0.0}, {0.0, 0.0},  {0.0, -1.0}, {-1.0, -1.0}, {-1.0, 0.0}, {-1.0, 1.0},
     {0.0, 1.0},   {-1.0, 0.0}, {-1.0, 0.0}, {0.0, -1.0}, {0.0, 0.0},   {1.0, 0.0},  {1.0, -1.0},
     {0.0, -1.0},  {0.0, 0.0},  {0.0, 1.0},  {1.0, 1.0},  {1.0, 0.0},   {0.0, 0.0}},

    {0b0000});
}

TYPED_TEST(PointInPolygonTest, OnePolygonOneRingDifferentHemisphereSpherical)
{
  CUSPATIAL_RUN_TEST(this->run_spherical_test,
                     {{0, 1.0, 0},
                      {0.5773502, -0.5773502, -0.5773502},
                      {-0.5773502, -0.5773502, -0.5773502},
                      {-0.5773502, -0.5773502, 0.5773502}},
                     {0, 1},
                     {0, 5},
                     {{-0.5773502, 0.5773502, 0.5773502},
                      {0.5773502, 0.5773502, 0.5773502},
                      {0.5773502, 0.5773502, -0.5773502},
                      {-0.5773502, 0.5773502, -0.5773502},
                      {-0.5773502, 0.5773502, 0.5773502}},
                     {true, false, false, false});
}

struct OffsetIteratorFunctor {
  std::size_t __device__ operator()(std::size_t idx) { return idx * 5; }
};

template <typename T>
struct PolyPointIteratorFunctorA {
  T __device__ operator()(std::size_t idx)
  {
    switch (idx % 5) {
      case 0:
      case 1: return -1.0;
      case 2:
      case 3: return 1.0;
      case 4:
      default: return -1.0;
    }
  }
};

template <typename T>
struct PolyPointIteratorFunctorB {
  T __device__ operator()(std::size_t idx)
  {
    switch (idx % 5) {
      case 0: return -1.0;
      case 1:
      case 2: return 1.0;
      case 3:
      case 4:
      default: return -1.0;
    }
  }
};

TYPED_TEST(PointInPolygonTest, 31PolygonSupport)
{
  using T = TypeParam;

  auto constexpr num_polys       = 31;
  auto constexpr num_poly_points = num_polys * 5;

  auto test_point   = make_device_vector<vec_2d<T>>({{0.0, 0.0}, {2.0, 0.0}});
  auto offsets_iter = thrust::make_counting_iterator<std::size_t>(0);
  auto poly_ring_offsets_iter =
    thrust::make_transform_iterator(offsets_iter, OffsetIteratorFunctor{});
  auto poly_point_xs_iter =
    thrust::make_transform_iterator(offsets_iter, PolyPointIteratorFunctorA<T>{});
  auto poly_point_ys_iter =
    thrust::make_transform_iterator(offsets_iter, PolyPointIteratorFunctorB<T>{});
  auto poly_point_iter = make_vec_2d_iterator(poly_point_xs_iter, poly_point_ys_iter);

  auto points_range = make_multipoint_range(
    test_point.size(), thrust::make_counting_iterator(0), test_point.size(), test_point.begin());

  auto polygons_range = make_multipolygon_range(num_polys,
                                                thrust::make_counting_iterator(0),
                                                num_polys,
                                                offsets_iter,
                                                num_polys,
                                                poly_ring_offsets_iter,
                                                num_poly_points,
                                                poly_point_iter);

  auto expected =
    std::vector<int32_t>({0b1111111111111111111111111111111, 0b0000000000000000000000000000000});
  auto got = rmm::device_vector<int32_t>(test_point.size());

  auto ret = point_in_polygon(points_range, polygons_range, got.begin());

  EXPECT_EQ(got, expected);
  EXPECT_EQ(ret, got.end());
}

TYPED_TEST(PointInPolygonTest, SelfClosingLoopLeftEdgeMissing)
{
  CUSPATIAL_RUN_TEST(this->run_test,

                     {{-2.0, 0.0}, {0.0, 0.0}, {2.0, 0.0}},
                     {0, 1},
                     {0, 4},
                     // "left" edge missing
                     {{-1, 1}, {1, 1}, {1, -1}, {-1, -1}},
                     {0b0, 0b1, 0b0});
}

TYPED_TEST(PointInPolygonTest, SelfClosingLoopRightEdgeMissing)
{
  using T = TypeParam;
  CUSPATIAL_RUN_TEST(this->run_test,
                     {{-2.0, 0.0}, {0.0, 0.0}, {2.0, 0.0}},
                     {0, 1},
                     {0, 4},
                     // "right" edge missing
                     {{1, -1}, {-1, -1}, {-1, 1}, {1, 1}},
                     {0b0, 0b1, 0b0});
}

TYPED_TEST(PointInPolygonTest, ContainsButCollinearWithBoundary)
{
  using T = TypeParam;
  CUSPATIAL_RUN_TEST(
    this->run_test,
    {{0.5, 0.5}},
    {0, 1},
    {0, 9},
    {{0, 0}, {0, 1}, {1, 1}, {1, 0.5}, {1.5, 0.5}, {1.5, 1}, {2, 1}, {2, 0}, {0, 0}},
    {0b1});
}
