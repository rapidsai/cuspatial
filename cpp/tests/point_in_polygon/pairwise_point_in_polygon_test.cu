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

#include <rmm/device_vector.hpp>

#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <initializer_list>

using namespace cuspatial;
using namespace cuspatial::test;

template <typename T>
struct PairwisePointInPolygonTest : public BaseFixture {
  void run_test(std::initializer_list<vec_2d<T>> points,
                std::initializer_list<int> polygon_offsets,
                std::initializer_list<int> ring_offsets,
                std::initializer_list<vec_2d<T>> polygon_points,
                std::initializer_list<uint8_t> expected)
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

    auto got = rmm::device_uvector<uint8_t>(points.size(), stream());

    auto ret = pairwise_point_in_polygon(mpoints, mpolys, got.begin(), stream());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(d_expected, got);
    EXPECT_EQ(ret, got.end());
  }
};

// float and double are logically the same but would require separate tests due to precision.
TYPED_TEST_CASE(PairwisePointInPolygonTest, FloatingPointTypes);

TYPED_TEST(PairwisePointInPolygonTest, OnePolygonOneRing)
{
  using T         = TypeParam;
  auto point_list = std::vector<vec_2d<T>>{{-2.0, 0.0},
                                           {2.0, 0.0},
                                           {0.0, -2.0},
                                           {0.0, 2.0},
                                           {-0.5, 0.0},
                                           {0.5, 0.0},
                                           {0.0, -0.5},
                                           {0.0, 0.5}};

  auto poly_offsets      = make_device_vector({0, 1});
  auto poly_ring_offsets = make_device_vector({0, 5});
  auto poly_point        = make_device_vector<vec_2d<T>>(
    {{-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}, {-1.0, -1.0}});

  auto polygon_range = make_multipolygon_range(poly_offsets.size() - 1,
                                               thrust::make_counting_iterator(0),
                                               poly_offsets.size() - 1,
                                               poly_offsets.begin(),
                                               poly_ring_offsets.size() - 1,
                                               poly_ring_offsets.begin(),
                                               poly_point.size(),
                                               poly_point.begin());

  auto got      = rmm::device_vector<uint8_t>(1);
  auto expected = thrust::host_vector{{false, false, false, false, true, true, true, true}};

  for (size_t i = 0; i < point_list.size(); ++i) {
    auto p           = point_list[i];
    auto d_point     = make_device_vector<vec_2d<T>>({{p.x, p.y}});
    auto point_range = make_multipoint_range(
      d_point.size(), thrust::make_counting_iterator(0), d_point.size(), d_point.begin());

    auto ret = pairwise_point_in_polygon(point_range, polygon_range, got.begin(), this->stream());
    EXPECT_EQ(got, std::vector<uint8_t>({expected[i]}));
    EXPECT_EQ(ret, got.end());
  }
}

TYPED_TEST(PairwisePointInPolygonTest, TwoPolygonsOneRingEach)
{
  using T         = TypeParam;
  auto point_list = std::vector<vec_2d<T>>{{-2.0, 0.0},
                                           {2.0, 0.0},
                                           {0.0, -2.0},
                                           {0.0, 2.0},
                                           {-0.5, 0.0},
                                           {0.5, 0.0},
                                           {0.0, -0.5},
                                           {0.0, 0.5}};

  auto poly_offsets      = make_device_vector({0, 1, 2});
  auto poly_ring_offsets = make_device_vector({0, 5, 10});
  auto poly_point        = make_device_vector<vec_2d<T>>({{-1.0, -1.0},
                                                          {-1.0, 1.0},
                                                          {1.0, 1.0},
                                                          {1.0, -1.0},
                                                          {-1.0, -1.0},
                                                          {0.0, 1.0},
                                                          {1.0, 0.0},
                                                          {0.0, -1.0},
                                                          {-1.0, 0.0},
                                                          {0.0, 1.0}});

  auto polygon_range = make_multipolygon_range(poly_offsets.size() - 1,
                                               thrust::make_counting_iterator(0),
                                               poly_offsets.size() - 1,
                                               poly_offsets.begin(),
                                               poly_ring_offsets.size() - 1,
                                               poly_ring_offsets.begin(),
                                               poly_point.size(),
                                               poly_point.begin());

  auto got      = rmm::device_vector<uint8_t>(2);
  auto expected = std::vector<uint8_t>({false, false, false, false, true, true, true, true});

  for (size_t i = 0; i < point_list.size() / 2; i = i + 2) {
    auto points = make_device_vector<vec_2d<T>>(
      {{point_list[i].x, point_list[i].y}, {point_list[i + 1].x, point_list[i + 1].y}});
    auto points_range = make_multipoint_range(
      points.size(), thrust::make_counting_iterator(0), points.size(), points.begin());

    auto ret = pairwise_point_in_polygon(points_range, polygon_range, got.begin(), this->stream());

    EXPECT_EQ(got, std::vector<uint8_t>({expected[i], expected[i + 1]}));
    EXPECT_EQ(ret, got.end());
  }
}

TYPED_TEST(PairwisePointInPolygonTest, OnePolygonTwoRings)
{
  using T = TypeParam;
  auto point_list =
    std::vector<std::vector<T>>{{0.0, 0.0}, {-0.4, 0.0}, {-0.6, 0.0}, {0.0, 0.4}, {0.0, -0.6}};

  auto poly_offsets      = make_device_vector({0, 2});
  auto num_polys         = poly_offsets.size() - 1;
  auto poly_ring_offsets = make_device_vector({0, 5, 10});
  auto poly_point        = make_device_vector<vec_2d<T>>({{-1.0, -1.0},
                                                          {1.0, -1.0},
                                                          {1.0, 1.0},
                                                          {-1.0, 1.0},
                                                          {-1.0, -1.0},
                                                          {-0.5, -0.5},
                                                          {-0.5, 0.5},
                                                          {0.5, 0.5},
                                                          {0.5, -0.5},
                                                          {-0.5, -0.5}});

  auto polygon_range = make_multipolygon_range(num_polys,
                                               thrust::make_counting_iterator(0),
                                               num_polys,
                                               poly_offsets.begin(),
                                               poly_ring_offsets.size() - 1,
                                               poly_ring_offsets.begin(),
                                               poly_point.size(),
                                               poly_point.begin());

  auto expected = std::vector<uint8_t>{0b0, 0b0, 0b1, 0b0, 0b1};

  for (size_t i = 0; i < point_list.size(); ++i) {
    auto got = rmm::device_vector<uint8_t>(1);

    auto point        = make_device_vector<vec_2d<T>>({{point_list[i][0], point_list[i][1]}});
    auto points_range = make_multipoint_range(
      point.size(), thrust::make_counting_iterator(0), point.size(), point.begin());

    auto ret = pairwise_point_in_polygon(points_range, polygon_range, got.begin(), this->stream());

    EXPECT_EQ(got, std::vector<uint8_t>{expected[i]});
    EXPECT_EQ(ret, got.end());
  }
}

TYPED_TEST(PairwisePointInPolygonTest, EdgesOfSquare)
{
  // 0: rect on min x side
  // 1: rect on max x side
  // 2: rect on min y side
  // 3: rect on max y side
  CUSPATIAL_RUN_TEST(
    this->run_test,
    {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
    {0, 1, 2, 3, 4},
    {0, 5, 10, 15, 20},
    {{-1.0, -1.0}, {0.0, -1.0}, {0.0, 1.0},  {-1.0, 1.0},  {-1.0, -1.0}, {0.0, -1.0}, {1.0, -1.0},
     {1.0, 1.0},   {0.0, 1.0},  {0.0, -1.0}, {-1.0, -1.0}, {-1.0, 0.0},  {1.0, 0.0},  {1.0, -1.0},
     {-1.0, 1.0},  {-1.0, 0.0}, {-1.0, 1.0}, {1.0, 1.0},   {1.0, 0.0},   {-1.0, 0.0}},
    {0b0, 0b0, 0b0, 0b0});
}

TYPED_TEST(PairwisePointInPolygonTest, CornersOfSquare)
{
  // 0: min x min y corner
  // 1: min x max y corner
  // 2: max x min y corner
  // 3: max x max y corner

  CUSPATIAL_RUN_TEST(
    this->run_test,
    {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
    {0, 1, 2, 3, 4},
    {0, 5, 10, 15, 20},
    {{-1.0, -1.0}, {-1.0, 0.0}, {0.0, 0.0},  {0.0, -1.0}, {-1.0, -1.0}, {-1.0, 0.0}, {-1.0, 1.0},
     {0.0, 1.0},   {-1.0, 0.0}, {-1.0, 0.0}, {0.0, -1.0}, {0.0, 0.0},   {1.0, 0.0},  {1.0, -1.0},
     {0.0, -1.0},  {0.0, 0.0},  {0.0, 1.0},  {1.0, 1.0},  {1.0, 0.0},   {0.0, 0.0}},
    {0b0, 0b0, 0b0, 0b0});
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

TYPED_TEST(PairwisePointInPolygonTest, 32PolygonSupport)
{
  using T = TypeParam;

  auto constexpr num_polys       = 32;
  auto constexpr num_poly_points = num_polys * 5;

  auto test_point = make_device_vector<vec_2d<T>>(
    {{0.0, 0.0}, {2.0, 0.0}, {0.0, 0.0}, {2.0, 0.0}, {0.0, 0.0}, {2.0, 0.0}, {0.0, 0.0},
     {2.0, 0.0}, {0.0, 0.0}, {2.0, 0.0}, {0.0, 0.0}, {2.0, 0.0}, {0.0, 0.0}, {2.0, 0.0},
     {0.0, 0.0}, {2.0, 0.0}, {0.0, 0.0}, {2.0, 0.0}, {0.0, 0.0}, {2.0, 0.0}, {0.0, 0.0},
     {2.0, 0.0}, {0.0, 0.0}, {2.0, 0.0}, {0.0, 0.0}, {2.0, 0.0}, {0.0, 0.0}, {2.0, 0.0},
     {0.0, 0.0}, {2.0, 0.0}, {0.0, 0.0}, {2.0, 0.0}});

  auto points_range = make_multipoint_range(
    test_point.size(), thrust::make_counting_iterator(0), test_point.size(), test_point.begin());

  auto offsets_iter = thrust::make_counting_iterator<std::size_t>(0);
  auto poly_ring_offsets_iter =
    thrust::make_transform_iterator(offsets_iter, OffsetIteratorFunctor{});
  auto poly_point_xs_iter =
    thrust::make_transform_iterator(offsets_iter, PolyPointIteratorFunctorA<T>{});
  auto poly_point_ys_iter =
    thrust::make_transform_iterator(offsets_iter, PolyPointIteratorFunctorB<T>{});
  auto poly_point_iter = make_vec_2d_iterator(poly_point_xs_iter, poly_point_ys_iter);

  auto polygons_range = make_multipolygon_range(num_polys,
                                                thrust::make_counting_iterator(0),
                                                num_polys,
                                                offsets_iter,
                                                num_polys,
                                                poly_ring_offsets_iter,
                                                num_poly_points,
                                                poly_point_iter);

  auto expected = std::vector<uint8_t>({1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                                        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0});
  auto got      = rmm::device_vector<uint8_t>(test_point.size());

  auto ret = pairwise_point_in_polygon(points_range, polygons_range, got.begin(), this->stream());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(got, expected);
  EXPECT_EQ(ret, got.end());
}

struct PairwisePointInPolygonErrorTest : public PairwisePointInPolygonTest<double> {};

TEST_F(PairwisePointInPolygonErrorTest, InsufficientPoints)
{
  using T = double;

  auto test_point   = make_device_vector<vec_2d<T>>({{0.0, 0.0}, {0.0, 0.0}});
  auto points_range = make_multipoint_range(
    test_point.size(), thrust::make_counting_iterator(0), test_point.size(), test_point.begin());

  auto poly_offsets      = make_device_vector({0, 1});
  auto num_polys         = poly_offsets.size() - 1;
  auto poly_ring_offsets = make_device_vector({0, 3});
  auto poly_point        = make_device_vector<vec_2d<T>>({{0.0, 1.0}, {1.0, 0.0}, {0.0, -1.0}});

  auto polygons_range = make_multipolygon_range(num_polys,
                                                thrust::make_counting_iterator(0),
                                                num_polys,
                                                poly_offsets.begin(),
                                                num_polys,
                                                poly_ring_offsets.begin(),
                                                poly_point.size(),
                                                poly_point.begin());

  auto got = rmm::device_vector<uint8_t>(test_point.size());

  EXPECT_THROW(pairwise_point_in_polygon(points_range, polygons_range, got.begin(), this->stream()),
               cuspatial::logic_error);
}

TEST_F(PairwisePointInPolygonErrorTest, InsufficientPolyOffsets)
{
  using T = double;

  auto test_point   = make_device_vector<vec_2d<T>>({{0.0, 0.0}, {0.0, 0.0}});
  auto points_range = make_multipoint_range(
    test_point.size(), thrust::make_counting_iterator(0), test_point.size(), test_point.begin());

  auto poly_offsets      = make_device_vector({0});
  auto num_polys         = poly_offsets.size() - 1;
  auto poly_ring_offsets = make_device_vector({0, 4});
  auto poly_point =
    make_device_vector<vec_2d<T>>({{0.0, 1.0}, {1.0, 0.0}, {0.0, -1.0}, {0.0, 1.0}});

  auto polygons_range = make_multipolygon_range(num_polys,
                                                thrust::make_counting_iterator(0),
                                                num_polys,
                                                poly_offsets.begin(),
                                                num_polys,
                                                poly_ring_offsets.begin(),
                                                poly_point.size(),
                                                poly_point.begin());

  auto got = rmm::device_vector<uint8_t>(test_point.size());

  EXPECT_THROW(pairwise_point_in_polygon(points_range, polygons_range, got.begin(), this->stream()),
               cuspatial::logic_error);
}
