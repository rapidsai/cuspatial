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

template <typename T>
struct PointInPolygonTest : public ::testing::Test {
 public:
  rmm::device_vector<vec_2d<T>> make_device_points(std::initializer_list<vec_2d<T>> pts)
  {
    return rmm::device_vector<vec_2d<T>>(pts.begin(), pts.end());
  }

  rmm::device_vector<std::size_t> make_device_offsets(std::initializer_list<std::size_t> pts)
  {
    return rmm::device_vector<std::size_t>(pts.begin(), pts.end());
  }
};

// float and double are logically the same but would require separate tests due to precision.
using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(PointInPolygonTest, TestTypes);

TYPED_TEST(PointInPolygonTest, OnePolygonOneRing)
{
  auto test_point        = this->make_device_points({{-2.0, 0.0},
                                                     {2.0, 0.0},
                                                     {0.0, -2.0},
                                                     {0.0, 2.0},
                                                     {-0.5, 0.0},
                                                     {0.5, 0.0},
                                                     {0.0, -0.5},
                                                     {0.0, 0.5}});
  auto poly_offsets      = this->make_device_offsets({0, 1});
  auto poly_ring_offsets = this->make_device_offsets({0, 5});
  auto poly_point =
    this->make_device_points({{-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}, {-1.0, -1.0}});

  auto got      = rmm::device_vector<int32_t>(test_point.size());
  auto expected = std::vector<int32_t>{false, false, false, false, true, true, true, true};

  auto ret = point_in_polygon(test_point.begin(),
                              test_point.end(),
                              poly_offsets.begin(),
                              poly_offsets.end(),
                              poly_ring_offsets.begin(),
                              poly_ring_offsets.end(),
                              poly_point.begin(),
                              poly_point.end(),
                              got.begin());

  EXPECT_EQ(got, expected);
  EXPECT_EQ(ret, got.end());
}

// cuspatial expects closed rings, however algorithms may work OK with unclosed rings
// in the future if we change to an algorithm that requires closed rings we may change or remove
// this test.
// Note that we don't introspect the values of offset arrays to validate closedness. So this test
// uses a polygon ring with 4 vertices so it doesn't fail polygon validation.
TYPED_TEST(PointInPolygonTest, OnePolygonOneRingUnclosed)
{
  auto test_point        = this->make_device_points({{-2.0, 0.0},
                                                     {2.0, 0.0},
                                                     {0.0, -2.0},
                                                     {0.0, 2.0},
                                                     {-0.5, 0.0},
                                                     {0.5, 0.0},
                                                     {0.0, -0.5},
                                                     {0.0, 0.5}});
  auto poly_offsets      = this->make_device_offsets({0, 1});
  auto poly_ring_offsets = this->make_device_offsets({0, 4});
  auto poly_point = this->make_device_points({{-1.0, -1.0}, {1.0, -1.0}, {1.0, 0.0}, {1.0, 1.0}});

  auto got      = rmm::device_vector<int32_t>(test_point.size());
  auto expected = std::vector<int32_t>{false, false, false, false, false, true, true, false};

  auto ret = point_in_polygon(test_point.begin(),
                              test_point.end(),
                              poly_offsets.begin(),
                              poly_offsets.end(),
                              poly_ring_offsets.begin(),
                              poly_ring_offsets.end(),
                              poly_point.begin(),
                              poly_point.end(),
                              got.begin());

  EXPECT_EQ(got, expected);
  EXPECT_EQ(ret, got.end());
}

TYPED_TEST(PointInPolygonTest, TwoPolygonsOneRingEach)
{
  auto test_point = this->make_device_points({{-2.0, 0.0},
                                              {2.0, 0.0},
                                              {0.0, -2.0},
                                              {0.0, 2.0},
                                              {-0.5, 0.0},
                                              {0.5, 0.0},
                                              {0.0, -0.5},
                                              {0.0, 0.5}});

  auto poly_offsets      = this->make_device_offsets({0, 1, 2});
  auto poly_ring_offsets = this->make_device_offsets({0, 5, 10});
  auto poly_point        = this->make_device_points({{-1.0, -1.0},
                                                     {-1.0, 1.0},
                                                     {1.0, 1.0},
                                                     {1.0, -1.0},
                                                     {-1.0, -1.0},
                                                     {0.0, 1.0},
                                                     {1.0, 0.0},
                                                     {0.0, -1.0},
                                                     {-1.0, 0.0},
                                                     {0.0, 1.0}});

  auto got      = rmm::device_vector<int32_t>(test_point.size());
  auto expected = std::vector<int32_t>({0b00, 0b00, 0b00, 0b00, 0b11, 0b11, 0b11, 0b11});

  auto ret = point_in_polygon(test_point.begin(),
                              test_point.end(),
                              poly_offsets.begin(),
                              poly_offsets.end(),
                              poly_ring_offsets.begin(),
                              poly_ring_offsets.end(),
                              poly_point.begin(),
                              poly_point.end(),
                              got.begin());

  EXPECT_EQ(got, expected);
  EXPECT_EQ(ret, got.end());
}

TYPED_TEST(PointInPolygonTest, OnePolygonTwoRings)
{
  auto test_point =
    this->make_device_points({{0.0, 0.0}, {-0.4, 0.0}, {-0.6, 0.0}, {0.0, 0.4}, {0.0, -0.6}});
  auto poly_offsets      = this->make_device_offsets({0, 2});
  auto poly_ring_offsets = this->make_device_offsets({0, 5, 10});
  auto poly_point        = this->make_device_points({{-1.0, -1.0},
                                                     {1.0, -1.0},
                                                     {1.0, 1.0},
                                                     {-1.0, 1.0},
                                                     {-1.0, -1.0},
                                                     {-0.5, -0.5},
                                                     {-0.5, 0.5},
                                                     {0.5, 0.5},
                                                     {0.5, -0.5},
                                                     {-0.5, -0.5}});

  auto got      = rmm::device_vector<int32_t>(test_point.size());
  auto expected = std::vector<int32_t>{0b0, 0b0, 0b1, 0b0, 0b1};

  auto ret = point_in_polygon(test_point.begin(),
                              test_point.end(),
                              poly_offsets.begin(),
                              poly_offsets.end(),
                              poly_ring_offsets.begin(),
                              poly_ring_offsets.end(),
                              poly_point.begin(),
                              poly_point.end(),
                              got.begin());

  EXPECT_EQ(got, expected);
  EXPECT_EQ(ret, got.end());
}

TYPED_TEST(PointInPolygonTest, EdgesOfSquare)
{
  auto test_point        = this->make_device_points({{0.0, 0.0}});
  auto poly_offsets      = this->make_device_offsets({0, 1, 2, 3, 4});
  auto poly_ring_offsets = this->make_device_offsets({0, 5, 10, 15, 20});

  // 0: rect on min x side
  // 1: rect on max x side
  // 2: rect on min y side
  // 3: rect on max y side
  auto poly_point = this->make_device_points(
    {{-1.0, -1.0}, {0.0, -1.0}, {0.0, 1.0},  {-1.0, 1.0},  {-1.0, -1.0}, {0.0, -1.0}, {1.0, -1.0},
     {1.0, 1.0},   {0.0, 1.0},  {0.0, -1.0}, {-1.0, -1.0}, {-1.0, 0.0},  {1.0, 0.0},  {1.0, -1.0},
     {-1.0, 1.0},  {-1.0, 0.0}, {-1.0, 1.0}, {1.0, 1.0},   {1.0, 0.0},   {-1.0, 0.0}});

  auto expected = std::vector<int32_t>{0b0000};
  auto got      = rmm::device_vector<int32_t>(test_point.size());

  auto ret = point_in_polygon(test_point.begin(),
                              test_point.end(),
                              poly_offsets.begin(),
                              poly_offsets.end(),
                              poly_ring_offsets.begin(),
                              poly_ring_offsets.end(),
                              poly_point.begin(),
                              poly_point.end(),
                              got.begin());

  EXPECT_EQ(got, expected);
  EXPECT_EQ(ret, got.end());
}

TYPED_TEST(PointInPolygonTest, CornersOfSquare)
{
  auto test_point        = this->make_device_points({{0.0, 0.0}});
  auto poly_offsets      = this->make_device_offsets({0, 1, 2, 3, 4});
  auto poly_ring_offsets = this->make_device_offsets({0, 5, 10, 15, 20});

  // 0: min x min y corner
  // 1: min x max y corner
  // 2: max x min y corner
  // 3: max x max y corner
  auto poly_point = this->make_device_points(
    {{-1.0, -1.0}, {-1.0, 0.0}, {0.0, 0.0},  {0.0, -1.0}, {-1.0, -1.0}, {-1.0, 0.0}, {-1.0, 1.0},
     {0.0, 1.0},   {-1.0, 0.0}, {-1.0, 0.0}, {0.0, -1.0}, {0.0, 0.0},   {1.0, 0.0},  {1.0, -1.0},
     {0.0, -1.0},  {0.0, 0.0},  {0.0, 1.0},  {1.0, 1.0},  {1.0, 0.0},   {0.0, 0.0}});

  auto expected = std::vector<int32_t>{0b0000};
  auto got      = rmm::device_vector<int32_t>(test_point.size());

  auto ret = point_in_polygon(test_point.begin(),
                              test_point.end(),
                              poly_offsets.begin(),
                              poly_offsets.end(),
                              poly_ring_offsets.begin(),
                              poly_ring_offsets.end(),
                              poly_point.begin(),
                              poly_point.end(),
                              got.begin());

  EXPECT_EQ(got, expected);
  EXPECT_EQ(ret, got.end());
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

  auto test_point   = this->make_device_points({{0.0, 0.0}, {2.0, 0.0}});
  auto offsets_iter = thrust::make_counting_iterator<std::size_t>(0);
  auto poly_ring_offsets_iter =
    thrust::make_transform_iterator(offsets_iter, OffsetIteratorFunctor{});
  auto poly_point_xs_iter =
    thrust::make_transform_iterator(offsets_iter, PolyPointIteratorFunctorA<T>{});
  auto poly_point_ys_iter =
    thrust::make_transform_iterator(offsets_iter, PolyPointIteratorFunctorB<T>{});
  auto poly_point_iter = make_vec_2d_iterator(poly_point_xs_iter, poly_point_ys_iter);

  auto expected =
    std::vector<int32_t>({0b1111111111111111111111111111111, 0b0000000000000000000000000000000});
  auto got = rmm::device_vector<int32_t>(test_point.size());

  auto ret = point_in_polygon(test_point.begin(),
                              test_point.end(),
                              offsets_iter,
                              offsets_iter + num_polys + 1,
                              poly_ring_offsets_iter,
                              poly_ring_offsets_iter + num_polys + 1,
                              poly_point_iter,
                              poly_point_iter + num_poly_points,
                              got.begin());

  EXPECT_EQ(got, expected);
  EXPECT_EQ(ret, got.end());
}

struct PointInPolygonErrorTest : public PointInPolygonTest<double> {};

TYPED_TEST(PointInPolygonTest, SelfClosingLoopLeftEdgeMissing)
{
  using T                = TypeParam;
  auto test_point        = this->make_device_points({{-2.0, 0.0}, {0.0, 0.0}, {2.0, 0.0}});
  auto poly_offsets      = this->make_device_offsets({0, 1});
  auto poly_ring_offsets = this->make_device_offsets({0, 4});
  // "left" edge missing
  auto poly_point = this->make_device_points({{-1, 1}, {1, 1}, {1, -1}, {-1, -1}});
  auto expected   = std::vector<int32_t>{0b0, 0b1, 0b0};
  auto got        = rmm::device_vector<int32_t>(test_point.size());

  auto ret = point_in_polygon(test_point.begin(),
                              test_point.end(),
                              poly_offsets.begin(),
                              poly_offsets.end(),
                              poly_ring_offsets.begin(),
                              poly_ring_offsets.end(),
                              poly_point.begin(),
                              poly_point.end(),
                              got.begin());

  EXPECT_EQ(expected, got);
  EXPECT_EQ(got.end(), ret);
}

TYPED_TEST(PointInPolygonTest, SelfClosingLoopRightEdgeMissing)
{
  using T                = TypeParam;
  auto test_point        = this->make_device_points({{-2.0, 0.0}, {0.0, 0.0}, {2.0, 0.0}});
  auto poly_offsets      = this->make_device_offsets({0, 1});
  auto poly_ring_offsets = this->make_device_offsets({0, 4});
  // "right" edge missing
  auto poly_point = this->make_device_points({{1, -1}, {-1, -1}, {-1, 1}, {1, 1}});
  auto expected   = std::vector<int32_t>{0b0, 0b1, 0b0};
  auto got        = rmm::device_vector<int32_t>(test_point.size());

  auto ret = point_in_polygon(test_point.begin(),
                              test_point.end(),
                              poly_offsets.begin(),
                              poly_offsets.end(),
                              poly_ring_offsets.begin(),
                              poly_ring_offsets.end(),
                              poly_point.begin(),
                              poly_point.end(),
                              got.begin());

  EXPECT_EQ(expected, got);
  EXPECT_EQ(got.end(), ret);
}

TYPED_TEST(PointInPolygonTest, ContainsButCollinearWithBoundary)
{
  using T = TypeParam;

  auto point   = cuspatial::test::make_multipoints_array<T>({{{0.5, 0.5}}});
  auto polygon = cuspatial::test::make_multipolygon_array<T>(
    {0, 1},
    {0, 1},
    {0, 9},
    {{0, 0}, {0, 1}, {1, 1}, {1, 0.5}, {1.5, 0.5}, {1.5, 1}, {2, 1}, {2, 0}, {0, 0}});

  auto point_range   = point.range();
  auto polygon_range = polygon.range();

  auto res = rmm::device_uvector<int32_t>(1, rmm::cuda_stream_default);

  cuspatial::point_in_polygon(point_range.point_begin(),
                              point_range.point_end(),
                              polygon_range.part_offset_begin(),
                              polygon_range.part_offset_end(),
                              polygon_range.ring_offset_begin(),
                              polygon_range.ring_offset_end(),
                              polygon_range.point_begin(),
                              polygon_range.point_end(),
                              res.begin());

  auto expect = cuspatial::test::make_device_vector<int32_t>({0b1});

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(res, expect);
}
