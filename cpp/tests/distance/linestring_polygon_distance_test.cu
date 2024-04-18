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

#include <cuspatial_test/base_fixture.hpp>
#include <cuspatial_test/vector_equality.hpp>
#include <cuspatial_test/vector_factories.cuh>

#include <cuspatial/distance.cuh>
#include <cuspatial/geometry/vec_2d.hpp>
#include <cuspatial/iterator_factory.cuh>
#include <cuspatial/range/range.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <initializer_list>

using namespace cuspatial;
using namespace cuspatial::test;

template <typename T>
struct PairwiseLinestringPolygonDistanceTest : public BaseFixture {
  rmm::cuda_stream_view stream() { return rmm::cuda_stream_default; }
  rmm::device_async_resource_ref mr() { return rmm::mr::get_current_device_resource(); }

  void run_single(std::initializer_list<std::size_t> multilinestring_geometry_offsets,
                  std::initializer_list<std::size_t> multilinestring_part_offsets,
                  std::initializer_list<vec_2d<T>> multilinestring_coordinates,
                  std::initializer_list<std::size_t> multipolygon_geometry_offsets,
                  std::initializer_list<std::size_t> multipolygon_part_offsets,
                  std::initializer_list<std::size_t> multipolygon_ring_offsets,
                  std::initializer_list<vec_2d<T>> multipolygon_coordinates,
                  std::initializer_list<T> expected)
  {
    auto multilinestrings = make_multilinestring_array(
      multilinestring_geometry_offsets, multilinestring_part_offsets, multilinestring_coordinates);

    auto multipolygons = make_multipolygon_array(multipolygon_geometry_offsets,
                                                 multipolygon_part_offsets,
                                                 multipolygon_ring_offsets,
                                                 multipolygon_coordinates);

    auto got = rmm::device_uvector<T>(multilinestrings.size(), stream());

    auto ret = pairwise_linestring_polygon_distance(
      multilinestrings.range(), multipolygons.range(), got.begin(), stream());

    auto d_expected = make_device_vector(expected);

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(got, d_expected);
    EXPECT_EQ(ret, got.end());
  }
};

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_CASE(PairwiseLinestringPolygonDistanceTest, TestTypes);

// Inputs are empty columns
TYPED_TEST(PairwiseLinestringPolygonDistanceTest, ZeroPairs)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,
                     {0},
                     {0},
                     std::initializer_list<P>{},
                     {0},
                     {0},
                     {0},
                     std::initializer_list<P>{},
                     {});
}

// One Pair Test matrix:
// 1. One pair, one part multilinestring, one part, one ring multipolygon (111)
// 2. One pair, one part multilinestring, one part, two ring multipolygon (112)
// 3. One pair, one part multilinestring, two part, two ring multipolygon (122)
// 4. One pair, two part multilinestring, two part, two ring multipolygon (222)

// For each of the above, test the following:
// 1. Disjoint
// 2. Contains
// 3. Crosses

TYPED_TEST(PairwiseLinestringPolygonDistanceTest, OnePair111Disjoint)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,
                     {0, 1},
                     {0, 4},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{3, 3}},
                     {0, 1},
                     {0, 1},
                     {0, 4},
                     {P{-1, -1}, P{-2, -2}, P{-2, -1}, P{-1, -1}},
                     {std::sqrt(T{2})});
}

TYPED_TEST(PairwiseLinestringPolygonDistanceTest, OnePair111Contains)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(
    this->run_single,
    {0, 1},
    {0, 4},
    {P{0, 0}, P{1, 1}, P{2, 2}, P{3, 3}},
    {0, 1},
    {0, 1},
    {0, 5},
    {P{-1, -1}, P{5, -1}, P{5, 5}, P{-1, 5}, P{-1, -1}},  // Polygon contains linestring
    {0});
}

TYPED_TEST(PairwiseLinestringPolygonDistanceTest, OnePair111Crosses)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,
                     {0, 1},
                     {0, 4},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{3, 3}},
                     {0, 1},
                     {0, 1},
                     {0, 4},
                     {P{-1, 0}, P{1, 0}, P{0, 1}, P{-1, 0}},
                     {0});
}

TYPED_TEST(PairwiseLinestringPolygonDistanceTest, OnePair112Contains)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,
                     {0, 1},
                     {0, 4},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{3, 3}},
                     {0, 1},
                     {0, 2},
                     {0, 5, 9},
                     {P{-1, -1},
                      P{5, -1},
                      P{5, 5},
                      P{-1, 5},
                      P{-1, -1},
                      P{0, 0},
                      P{0, -1},
                      P{-1, -1},
                      P{-1, 0},
                      P{0, 0}},
                     {0});
}

TYPED_TEST(PairwiseLinestringPolygonDistanceTest, OnePair112Disjoint)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,
                     {0, 1},
                     {0, 4},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{2, 3}},
                     {0, 1},
                     {0, 2},
                     {0, 5, 10},
                     {P{-1, -1},
                      P{-4, -1},
                      P{-4, -4},
                      P{-1, -4},
                      P{-1, -1},
                      P{-2, -2},
                      P{-3, -2},
                      P{-3, -3},
                      P{-2, -3},
                      P{-2, -2}},
                     {std::sqrt(T{2})});
}

TYPED_TEST(PairwiseLinestringPolygonDistanceTest, OnePair112Crosses)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(
    this->run_single,
    {0, 1},
    {0, 4},
    {P{0, 0}, P{1, 1}, P{2, 2}, P{2, 3}},
    {0, 1},
    {0, 2},
    {0, 4, 8},
    {P{-1, -1}, P{-2, -2}, P{-2, -1}, P{-1, -1}, P{0, 1}, P{2, 1}, P{2, 0}, P{0, 1}},
    {0.0});
}

TYPED_TEST(PairwiseLinestringPolygonDistanceTest, OnePair122Disjoint)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(
    this->run_single,
    {0, 1},
    {0, 4},
    {P{0, 0}, P{1, 1}, P{2, 2}, P{3, 3}},
    {0, 2},
    {0, 1, 2},
    {0, 4, 9},
    {P{-1, -1}, P{-2, -2}, P{-2, -1}, P{-1, -1}, P{3, 4}, P{3, 5}, P{4, 5}, P{4, 4}, P{3, 4}},
    {1.0});
}

TYPED_TEST(PairwiseLinestringPolygonDistanceTest, OnePair122Contains)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,
                     {0, 1},
                     {0, 4},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{3, 3}},
                     {0, 2},
                     {0, 1, 2},
                     {0, 4, 9},
                     {
                       P{-1, -1},
                       P{-2, -2},
                       P{-2, -1},
                       P{-1, -1},
                       P{-1, -1},
                       P{5, -1},
                       P{5, 5},
                       P{-1, 5},
                       P{-1, -1}  // includes the multilinestring
                     },
                     {0});
}

TYPED_TEST(PairwiseLinestringPolygonDistanceTest, OnePair122Crosses)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(
    this->run_single,
    {0, 1},
    {0, 4},
    {P{0, 0}, P{1, 1}, P{2, 2}, P{3, 3}},
    {0, 2},
    {0, 1, 2},
    {0, 4, 8},
    {P{-1, -1}, P{-2, -2}, P{-2, -1}, P{-1, -1}, P{0, 1}, P{2, 1}, P{2, 0}, P{0, 1}},
    {0});
}

TYPED_TEST(PairwiseLinestringPolygonDistanceTest, OnePair222Disjoint)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(
    this->run_single,
    {0, 2},
    {0, 2, 4},
    {P{1, 1}, P{0, 0}, P{4, 6}, P{4, 7}},
    {0, 2},
    {0, 1, 2},
    {0, 4, 9},
    {P{-1, -1}, P{-2, -2}, P{-2, -1}, P{-1, -1}, P{3, 4}, P{3, 5}, P{4, 5}, P{4, 4}, P{3, 4}},
    {1.0});
}

TYPED_TEST(PairwiseLinestringPolygonDistanceTest, OnePair222Contains)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,
                     {0, 2},
                     {0, 2, 4},
                     {P{1, 1}, P{0, 0}, P{6, 6}, P{6, 7}},
                     {0, 2},
                     {0, 1, 2},
                     {0, 4, 9},
                     {
                       P{-1, -1},
                       P{-2, -2},
                       P{-2, -1},
                       P{-1, -1},
                       P{-1, -1},
                       P{5, -1},
                       P{5, 5},
                       P{-1, 5},
                       P{-1, -1}  // includes the multilinestring
                     },
                     {0});
}

TYPED_TEST(PairwiseLinestringPolygonDistanceTest, OnePair222Crosses)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(
    this->run_single,
    {0, 2},
    {0, 2, 4},
    {P{0, 0}, P{1, 1}, P{-1, 0}, P{0, -1}},
    {0, 2},
    {0, 1, 2},
    {0, 4, 8},
    {P{-1, -1}, P{-2, -2}, P{-2, -1}, P{-1, -1}, P{0, 1}, P{2, 1}, P{2, 0}, P{0, 1}},
    {0});
}

TYPED_TEST(PairwiseLinestringPolygonDistanceTest, TwoPairs)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,
                     {0, 1, 2},
                     {0, 4, 7},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{3, 3}, P{10, 10}, P{11, 11}, P{12, 12}},
                     {0, 1, 2},
                     {0, 1, 2},
                     {0, 4, 9},
                     {P{-1, -1},
                      P{-2, -2},
                      P{-2, -1},
                      P{-1, -1},
                      P{-10, -10},
                      P{-10, -11},
                      P{-11, -11},
                      P{-11, -10},
                      P{-10, -10}},
                     {std::sqrt(T{2}), 20 * std::sqrt(T{2})});
}

TYPED_TEST(PairwiseLinestringPolygonDistanceTest, TwoPairs2)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(
    this->run_single,
    {0, 1, 3},
    {0, 4, 7, 9},
    {P{0, 0}, P{1, 1}, P{2, 2}, P{3, 3}, P{10, 10}, P{11, 11}, P{12, 12}, P{20, 20}, P{20, 21}},
    {0, 1, 3},
    {0, 1, 2, 3},
    {0, 4, 9, 13},
    {P{-1, -1},
     P{-2, -2},
     P{-2, -1},
     P{-1, -1},
     P{-10, -10},
     P{-10, -11},
     P{-11, -11},
     P{-11, -10},
     P{-10, -10},
     P{20, -10},
     P{20, -20},
     P{30, -20},
     P{20, -10}},
    {std::sqrt(T{2}), 10 * std::sqrt(T{5})});
}

TYPED_TEST(PairwiseLinestringPolygonDistanceTest, TwoPairsCrosses)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,
                     {0, 1, 2},
                     {0, 4, 6},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{3, 3}, P{5, 5}, P{20, 20}},
                     {0, 1, 2},
                     {0, 1, 3},
                     {0, 4, 8, 12},
                     {P{-1, -1},
                      P{-2, -2},
                      P{-2, -1},
                      P{-1, -1},
                      P{0, 0},
                      P{20, 0},
                      P{0, 20},
                      P{0, 0},
                      P{5, 5},
                      P{15, 5},
                      P{5, 15},
                      P{5, 5}},
                     {std::sqrt(T{2}), 0.0});
}

// Empty Geometries Tests

/// Empty MultiLinestring vs Non-empty multipolygons
TYPED_TEST(PairwiseLinestringPolygonDistanceTest, ThreePairEmptyMultiLinestring)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,

                     {0, 1, 1, 2},
                     {0, 4, 7},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{3, 3}, P{10, 10}, P{11, 11}, P{12, 12}},

                     {0, 1, 2, 3},
                     {0, 1, 2, 3},
                     {0, 4, 9, 14},
                     {P{-1, -1},
                      P{-2, -2},
                      P{-2, -1},
                      P{-1, -1},

                      P{-20, -20},
                      P{-20, -21},
                      P{-21, -21},
                      P{-21, -20},
                      P{-20, -20},

                      P{-10, -10},
                      P{-10, -11},
                      P{-11, -11},
                      P{-11, -10},
                      P{-10, -10}},

                     {std::sqrt(T{2}), std::numeric_limits<T>::quiet_NaN(), 20 * std::sqrt(T{2})});
}

/// Non-empty MultiLinestring vs Empty multipolygons
TYPED_TEST(PairwiseLinestringPolygonDistanceTest, ThreePairEmptyMultiPolygon)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,

                     {0, 1, 2, 3},
                     {0, 4, 7, 10},
                     {P{0, 0},
                      P{1, 1},
                      P{2, 2},
                      P{3, 3},
                      P{20, 20},
                      P{21, 21},
                      P{22, 22},
                      P{10, 10},
                      P{11, 11},
                      P{12, 12}},

                     {0, 1, 1, 2},
                     {0, 1, 2},
                     {0, 4, 9},
                     {P{-1, -1},
                      P{-2, -2},
                      P{-2, -1},
                      P{-1, -1},

                      P{-10, -10},
                      P{-10, -11},
                      P{-11, -11},
                      P{-11, -10},
                      P{-10, -10}},
                     {std::sqrt(T{2}), std::numeric_limits<T>::quiet_NaN(), 20 * std::sqrt(T{2})});
}

/// FIXME: Empty MultiLinestring vs Empty multipolygons
/// This example fails at distance util, where point-polyogn intersection kernel doesn't handle
/// empty multipoint/multipolygons.
TYPED_TEST(PairwiseLinestringPolygonDistanceTest,
           DISABLED_ThreePairEmptyMultiLineStringEmptyMultiPolygon)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  CUSPATIAL_RUN_TEST(this->run_single,

                     {0, 1, 1, 3},
                     {0, 4, 7},
                     {P{0, 0}, P{1, 1}, P{2, 2}, P{3, 3}, P{10, 10}, P{11, 11}, P{12, 12}},

                     {0, 1, 1, 2},
                     {0, 1, 2, 3},
                     {0, 4, 9, 14},
                     {P{-1, -1},
                      P{-2, -2},
                      P{-2, -1},
                      P{-1, -1},

                      P{-10, -10},
                      P{-10, -11},
                      P{-11, -11},
                      P{-11, -10},
                      P{-10, -10}},
                     {std::sqrt(T{2}), std::numeric_limits<T>::quiet_NaN(), 20 * std::sqrt(T{2})});
}
