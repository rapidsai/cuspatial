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

#include <thrust/iterator/zip_iterator.h>

#include <initializer_list>

using namespace cuspatial;
using namespace cuspatial::test;

template <typename T>
struct PairwisePolygonDistanceTest : BaseFixture {
  template <typename MultipolygonRangeA, typename MultipolygonRangeB>
  void run_test(MultipolygonRangeA lhs,
                MultipolygonRangeB rhs,
                rmm::device_uvector<T> const& expected)
  {
    auto got = rmm::device_uvector<T>(lhs.size(), stream());
    auto ret = pairwise_polygon_distance(lhs, rhs, got.begin(), stream());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected, got);
    EXPECT_EQ(thrust::distance(got.begin(), ret), expected.size());
  }

  void run(std::initializer_list<std::size_t> lhs_multipolygon_geometry_offsets,
           std::initializer_list<std::size_t> lhs_multipolygon_part_offsets,
           std::initializer_list<std::size_t> lhs_multipolygon_ring_offsets,
           std::initializer_list<vec_2d<T>> lhs_multipolygon_coordinates,
           std::initializer_list<std::size_t> rhs_multipolygon_geometry_offsets,
           std::initializer_list<std::size_t> rhs_multipolygon_part_offsets,
           std::initializer_list<std::size_t> rhs_multipolygon_ring_offsets,
           std::initializer_list<vec_2d<T>> rhs_multipolygon_coordinates,
           std::initializer_list<T> expected)
  {
    auto lhs = make_multipolygon_array(lhs_multipolygon_geometry_offsets,
                                       lhs_multipolygon_part_offsets,
                                       lhs_multipolygon_ring_offsets,
                                       lhs_multipolygon_coordinates);

    auto rhs = make_multipolygon_array(rhs_multipolygon_geometry_offsets,
                                       rhs_multipolygon_part_offsets,
                                       rhs_multipolygon_ring_offsets,
                                       rhs_multipolygon_coordinates);

    auto lhs_range = lhs.range();
    auto rhs_range = rhs.range();

    auto d_expected = make_device_uvector(expected, stream(), mr());

    // Euclidean distance is symmetric
    run_test(lhs_range, rhs_range, d_expected);
    run_test(rhs_range, lhs_range, d_expected);
  }
};

TYPED_TEST_CASE(PairwisePolygonDistanceTest, FloatingPointTypes);

TYPED_TEST(PairwisePolygonDistanceTest, Empty)
{
  this->run({0}, {0}, {0}, {}, {0}, {0}, {0}, {}, {});
}

// Test Matrix
// One Pair:
//   lhs-rhs Relationship: Disjoint, Touching, Overlapping, Contained, Within
//   Holes: No, Yes
//   Multipolygon: No, Yes

TYPED_TEST(PairwisePolygonDistanceTest, OnePairSinglePolygonDisjointNoHole)
{
  this->run({0, 1},
            {0, 1},
            {0, 4},
            {{0, 0}, {0, 1}, {1, 1}, {1, 0}},
            {0, 1},
            {0, 1},
            {0, 4},
            {{-1, 0}, {-1, 1}, {-2, 0}, {-1, 0}},
            {1});
}

TYPED_TEST(PairwisePolygonDistanceTest, OnePairSinglePolygonTouchingNoHole)
{
  this->run({0, 1},
            {0, 1},
            {0, 4},
            {{0, 0}, {0, 1}, {1, 1}, {1, 0}},
            {0, 1},
            {0, 1},
            {0, 4},
            {{1, 0}, {2, 0}, {2, 1}, {1, 0}},
            {0});
}

TYPED_TEST(PairwisePolygonDistanceTest, OnePairSinglePolygonOverlappingNoHole)
{
  this->run({0, 1},
            {0, 1},
            {0, 4},
            {{0, 0}, {0, 1}, {1, 1}, {1, 0}},
            {0, 1},
            {0, 1},
            {0, 4},
            {{0.5, 0}, {2, 0}, {2, 1}, {0.5, 0}},
            {0});
}

TYPED_TEST(PairwisePolygonDistanceTest, OnePairSinglePolygonContainedNoHole)
{
  this->run({0, 1},
            {0, 1},
            {0, 5},
            {{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}},
            {0, 1},
            {0, 1},
            {0, 5},
            {{0.25, 0.25}, {0.75, 0.25}, {0.75, 0.75}, {0.25, 0.75}, {0.25, 0.25}},
            {0});
}

TYPED_TEST(PairwisePolygonDistanceTest, OnePairSinglePolygonWithinNoHole)
{
  this->run({0, 1},
            {0, 1},
            {0, 5},
            {{0.25, 0.25}, {0.75, 0.25}, {0.75, 0.75}, {0.25, 0.75}, {0.25, 0.25}},
            {0, 1},
            {0, 1},
            {0, 5},
            {{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}},
            {0});
}

TYPED_TEST(PairwisePolygonDistanceTest, OnePairSinglePolygonDisjointHasHole)
{
  this->run({0, 1},
            {0, 2},
            {0, 4, 8},
            {{0.0, 0.0},
             {2.0, 0.0},
             {2.0, 2.0},
             {0.0, 0.0},
             {1.0, 0.75},
             {1.5, 0.75},
             {1.25, 1.0},
             {1.0, 0.75}},
            {0, 1},
            {0, 2},
            {0, 4, 8},
            {{-1.0, 0.0},
             {-1.0, -1.0},
             {-2.0, 0.0},
             {-1.0, 0.0},
             {-1.25, -0.25},
             {-1.25, -0.5},
             {-1.5, -0.25},
             {-1.25, -0.25}},
            {1});
}

TYPED_TEST(PairwisePolygonDistanceTest, OnePairSinglePolygonDisjointHasHole2)
{
  this->run({0, 1},
            {0, 2},
            {0, 5, 10},
            {{0.0, 0.0},
             {10.0, 0.0},
             {10.0, 10.0},
             {0.0, 10.0},
             {0.0, 0.0},
             {2.0, 2.0},
             {2.0, 6.0},
             {6.0, 6.0},
             {6.0, 2.0},
             {2.0, 2.0}},
            {0, 1},
            {0, 1},
            {0, 5},
            {{3.0, 3.0}, {3.0, 4.0}, {4.0, 4.0}, {4.0, 3.0}, {3.0, 3.0}},
            {1});
}

TYPED_TEST(PairwisePolygonDistanceTest, OnePairSinglePolygonTouchingHasHole)
{
  this->run({0, 1},
            {0, 2},
            {0, 4, 8},
            {{0.0, 0.0},
             {2.0, 0.0},
             {2.0, 2.0},
             {0.0, 0.0},
             {1.0, 0.75},
             {1.5, 0.75},
             {1.25, 1.0},
             {1.0, 0.75}},
            {0, 1},
            {0, 2},
            {0, 4, 8},
            {{2.0, 0.0},
             {3.0, 0.0},
             {3.0, 1.0},
             {2.0, 0.0},
             {2.5, 0.25},
             {2.75, 0.25},
             {2.75, 0.5},
             {2.5, 0.25}},
            {0});
}

TYPED_TEST(PairwisePolygonDistanceTest, OnePairSinglePolygonOverlappingHasHole)
{
  this->run({0, 1},
            {0, 2},
            {0, 5, 10},
            {{0, 0}, {4, 0}, {4, 4}, {0, 4}, {0, 0}, {2, 2}, {2, 3}, {3, 3}, {3, 2}, {2, 2}},
            {0, 1},
            {0, 2},
            {0, 4, 8},
            {{2, -1}, {5, 4}, {5, -1}, {2, -1}, {3, -0.5}, {4, 0}, {4, -0.5}, {3, -0.5}},
            {0});
}

TYPED_TEST(PairwisePolygonDistanceTest, OnePairSinglePolygonContainedHasHole)
{
  this->run({0, 1},
            {0, 2},
            {0, 5, 10},
            {{0, 0}, {4, 0}, {4, 4}, {0, 4}, {0, 0}, {1, 3}, {1, 1}, {3, 1}, {1, 3}, {1, 1}},
            {0, 1},
            {0, 1},
            {0, 4},
            {{1, 3}, {3, 1}, {3, 3}, {1, 3}},
            {0});
}

TYPED_TEST(PairwisePolygonDistanceTest, OnePairSinglePolygonWithinHasHole)
{
  this->run({0, 1},
            {0, 1},
            {0, 4},
            {{1, 3}, {3, 1}, {3, 3}, {1, 3}},
            {0, 1},
            {0, 2},
            {0, 5, 9},
            {{0, 0}, {4, 0}, {4, 4}, {0, 4}, {0, 0}, {1, 1}, {3, 1}, {1, 3}, {1, 1}},
            {0});
}

TYPED_TEST(PairwisePolygonDistanceTest, OnePairMultiPolygonDisjointNoHole)
{
  this->run({0, 2},
            {0, 1, 2},
            {0, 4, 8},
            {{0.0, 0.0},
             {2.0, 0.0},
             {2.0, 2.0},
             {0.0, 0.0},
             {3.0, 3.0},
             {3.0, 4.0},
             {4.0, 4.0},
             {3.0, 3.0}},
            {0, 1},
            {0, 1},
            {0, 4},
            {{-1.0, 0.0}, {-1.0, -1.0}, {-2.0, 0.0}, {-1.0, 0.0}},
            {1});
}

TYPED_TEST(PairwisePolygonDistanceTest, OnePairMultiPolygonTouchingNoHole)
{
  this->run({0, 2},
            {0, 1, 2},
            {0, 4, 8},
            {{0.0, 0.0},
             {2.0, 0.0},
             {2.0, 2.0},
             {0.0, 0.0},
             {3.0, 3.0},
             {3.0, 4.0},
             {4.0, 4.0},
             {3.0, 3.0}},
            {0, 1},
            {0, 1},
            {0, 4},
            {{3.0, 3.0}, {2.0, 3.0}, {2.0, 2.0}, {3.0, 3.0}},
            {0});
}

TYPED_TEST(PairwisePolygonDistanceTest, OnePairMultiPolygonOverlappingNoHole)
{
  this->run({0, 2},
            {0, 1, 2},
            {0, 4, 8},
            {{0.0, 0.0},
             {2.0, 0.0},
             {2.0, 2.0},
             {0.0, 0.0},
             {3.0, 3.0},
             {3.0, 4.0},
             {4.0, 4.0},
             {3.0, 3.0}},
            {0, 1},
            {0, 1},
            {0, 4},
            {{1.0, 1.0}, {3.0, 1.0}, {3.0, 3.0}, {1.0, 1.0}},
            {0});
}

TYPED_TEST(PairwisePolygonDistanceTest, OnePairMultiPolygonContainedNoHole)
{
  this->run({0, 2},
            {0, 1, 2},
            {0, 4, 8},
            {{0.0, 0.0},
             {2.0, 0.0},
             {2.0, 2.0},
             {0.0, 0.0},
             {1.0, 1.0},
             {1.0, 2.0},
             {2.0, 2.0},
             {1.0, 1.0}},
            {0, 1},
            {0, 1},
            {0, 4},
            {{0.5, 0.25}, {1.5, 0.25}, {1.5, 1.25}, {0.5, 0.25}},
            {0});
}

// Two Pair Tests

TYPED_TEST(PairwisePolygonDistanceTest, TwoPairSinglePolygonNoHole)
{
  this->run({0, 1, 2},
            {0, 1, 2},
            {0, 4, 8},
            {{0.0, 0.0},
             {2.0, 0.0},
             {2.0, 2.0},
             {0.0, 0.0},
             {3.0, 3.0},
             {3.0, 4.0},
             {4.0, 4.0},
             {3.0, 3.0}},
            {0, 1, 2},
            {0, 1, 2},
            {0, 4, 8},
            {{3.0, 0.0},
             {4.0, 0.0},
             {4.0, 1.0},
             {3.0, 0.0},
             {3.0, 3.0},
             {3.0, 2.0},
             {2.0, 2.0},
             {3.0, 3.0}},
            {1, 0});
}

TYPED_TEST(PairwisePolygonDistanceTest, TwoPairSinglePolygonHasHole)
{
  this->run({0, 1, 2},
            {0, 1, 3},
            {0, 4, 8, 12},
            {{0.0, 0.0},
             {2.0, 0.0},
             {2.0, 2.0},
             {0.0, 0.0},
             {3.0, 3.0},
             {3.0, 4.0},
             {4.0, 4.0},
             {3.0, 3.0},
             {3.25, 3.5},
             {3.5, 3.5},
             {3.5, 3.75},
             {3.25, 3.5}},
            {0, 1, 2},
            {0, 1, 2},
            {0, 4, 8},
            {{3.0, 0.0},
             {4.0, 0.0},
             {4.0, 1.0},
             {3.0, 0.0},
             {3.0, 3.0},
             {3.0, 2.0},
             {2.0, 2.0},
             {3.0, 3.0}},
            {1, 0});
}

TYPED_TEST(PairwisePolygonDistanceTest, TwoPairMultiPolygonNoHole)
{
  this->run({0, 1, 3},
            {0, 1, 2, 3},
            {0, 4, 8, 12},
            {{0.0, 0.0},
             {2.0, 0.0},
             {2.0, 2.0},
             {0.0, 0.0},
             {3.0, 3.0},
             {3.0, 4.0},
             {4.0, 4.0},
             {3.0, 3.0},
             {3.0, 3.0},
             {5.0, 3.0},
             {4.0, 2.0},
             {3.0, 3.0}},
            {0, 1, 2},
            {0, 1, 2},
            {0, 4, 8},
            {{3.0, 0.0},
             {4.0, 0.0},
             {4.0, 1.0},
             {3.0, 0.0},
             {3.0, 3.0},
             {3.0, 2.0},
             {2.0, 2.0},
             {3.0, 3.0}},
            {1, 0});
}

TYPED_TEST(PairwisePolygonDistanceTest, TwoPairMultiPolygonHasHole)
{
  this->run({0, 1, 3},
            {0, 1, 2, 4},
            {0, 4, 8, 12, 16},
            {{0.0, 0.0},
             {2.0, 0.0},
             {2.0, 2.0},
             {0.0, 0.0},

             {3.0, 3.0},
             {3.0, 4.0},
             {4.0, 4.0},
             {3.0, 3.0},

             {3.0, 3.0},
             {5.0, 3.0},
             {4.0, 2.0},
             {3.0, 3.0},

             {3.5, 2.9},
             {4.5, 2.9},
             {4, 2.4},
             {3.5, 2.9}},
            {0, 1, 2},
            {0, 1, 2},
            {0, 4, 8},
            {{3.0, 0.0},
             {4.0, 0.0},
             {4.0, 1.0},
             {3.0, 0.0},
             {3.0, 3.0},
             {3.0, 2.0},
             {2.0, 2.0},
             {3.0, 3.0}},
            {1, 0});
}
