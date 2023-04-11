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

#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/experimental/polygon_distance.cuh>
#include <cuspatial/experimental/ranges/range.cuh>
#include <cuspatial/vec_2d.hpp>

#include <gtest/gtest-typed-test.h>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

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

  void run_single(std::initializer_list<std::size_t> lhs_multipolygon_geometry_offsets,
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
  this->run_single({0}, {0}, {0}, {}, {0}, {0}, {0}, {}, {});
}

// Test Matrix
// One Pair:
//   lhs-rhs Relationship: Disjoint, Touching, Overlapping, Contained, Within
//   Holes: No, Yes
//   Multipolygon: No, Yes

TYPED_TEST(PairwisePolygonDistanceTest, OnePairSinglePolygonDisjointNoHole)
{
  this->run_single({0, 1},
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
  this->run_single({0, 1},
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
  this->run_single({0, 1},
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
  this->run_single({0, 1},
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
  this->run_single({0, 1},
                   {0, 1},
                   {0, 5},
                   {{0.25, 0.25}, {0.75, 0.25}, {0.75, 0.75}, {0.25, 0.75}, {0.25, 0.25}},
                   {0, 1},
                   {0, 1},
                   {0, 5},
                   {{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}},
                   {0});
}
