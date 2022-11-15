/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <cuspatial/experimental/detail/linestring_intersection_count.cuh>
#include <cuspatial/experimental/iterator_factory.cuh>
#include <cuspatial/traits.hpp>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>
#include <thrust/uninitialized_fill.h>

#include <rmm/exec_policy.hpp>
#include <thrust/iterator/zip_iterator.h>

#include <initializer_list>
#include <type_traits>

using namespace cuspatial;
using namespace cuspatial::test;

template <typename T>
struct LinestringIntersectionCountTest : public ::testing::Test {
  rmm::cuda_stream_view default_stream() { return rmm::cuda_stream_default; }
};

// float and double are logically the same but would require seperate tests due to precision.
using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(LinestringIntersectionCountTest, TestTypes);

TYPED_TEST(LinestringIntersectionCountTest, SingleToSingleSimpleIntersectSinglePoint)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using count_type = unsigned;

  auto multilinestrings1 = make_multilinestring_array({0, 1}, {0, 2}, {P{0, 0}, P{1, 1}});

  auto multilinestrings2 = make_multilinestring_array({0, 1}, {0, 2}, {P{0, 1}, P{1, 0}});

  rmm::device_vector<count_type> num_intersecting_points(multilinestrings1.size());
  rmm::device_vector<count_type> num_overlapping_segments(multilinestrings1.size());

  std::vector<count_type> expected_intersecting_points_count{1};
  std::vector<count_type> expected_overlapping_segment_count{0};

  pairwise_linestring_intersection_upper_bound_count(multilinestrings1.range(),
                                                     multilinestrings2.range(),
                                                     num_intersecting_points.begin(),
                                                     num_overlapping_segments.begin(),
                                                     this->default_stream());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(num_intersecting_points, expected_intersecting_points_count);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(num_overlapping_segments, expected_overlapping_segment_count);
}

TYPED_TEST(LinestringIntersectionCountTest, SingleToSingleIntersectMultipoint)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using count_type = unsigned;

  auto multilinestrings1 = make_multilinestring_array({0, 1}, {0, 2}, {P{0, 0}, P{1, 1}});

  auto multilinestrings2 =
    make_multilinestring_array({0, 1}, {0, 3}, {P{0.5, 0.0}, P{0.0, 0.5}, P{1.0, 0.5}});

  rmm::device_vector<count_type> num_intersecting_points(multilinestrings1.size());
  rmm::device_vector<count_type> num_overlapping_segments(multilinestrings1.size());

  std::vector<count_type> expected_intersecting_points_count{2};
  std::vector<count_type> expected_overlapping_segment_count{0};

  pairwise_linestring_intersection_upper_bound_count(multilinestrings1.range(),
                                                     multilinestrings2.range(),
                                                     num_intersecting_points.begin(),
                                                     num_overlapping_segments.begin(),
                                                     this->default_stream());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(num_intersecting_points, expected_intersecting_points_count);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(num_overlapping_segments, expected_overlapping_segment_count);
}

TYPED_TEST(LinestringIntersectionCountTest, SingleToSingleOverlapSingleSegment)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using count_type = unsigned;

  auto multilinestrings1 = make_multilinestring_array({0, 1}, {0, 2}, {P{0, 0}, P{1, 1}});

  auto multilinestrings2 = make_multilinestring_array({0, 1}, {0, 2}, {P{0.5, 0.5}, P{1.5, 1.5}});

  rmm::device_vector<count_type> num_intersecting_points(multilinestrings1.size());
  rmm::device_vector<count_type> num_overlapping_segments(multilinestrings1.size());

  std::vector<count_type> expected_intersecting_points_count{0};
  std::vector<count_type> expected_overlapping_segment_count{1};

  pairwise_linestring_intersection_upper_bound_count(multilinestrings1.range(),
                                                     multilinestrings2.range(),
                                                     num_intersecting_points.begin(),
                                                     num_overlapping_segments.begin(),
                                                     this->default_stream());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(num_intersecting_points, expected_intersecting_points_count);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(num_overlapping_segments, expected_overlapping_segment_count);
}

TYPED_TEST(LinestringIntersectionCountTest, SingleToSingleOverlapSingleSegment2)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using count_type = unsigned;

  // The "upper bound intersection count" between
  // (0, 0) -> (1, 1) and (-1, -1) -> (0.25, 0.25) -> (0.25, 0.0) is
  // 1 intersection point(s) (0.25, 0.25) and
  // 1 overlapping segment(s) (0.0, 0.0) -> (0.25, 0.25)

  auto multilinestrings1 = make_multilinestring_array({0, 1}, {0, 2}, {P{0, 0}, P{1, 1}});

  auto multilinestrings2 =
    make_multilinestring_array({0, 1}, {0, 3}, {P{-1, -1}, P{0.25, 0.25}, P{0.25, 0.0}});

  rmm::device_vector<count_type> num_intersecting_points(multilinestrings1.size());
  rmm::device_vector<count_type> num_overlapping_segments(multilinestrings1.size());

  std::vector<count_type> expected_intersecting_points_count{1};
  std::vector<count_type> expected_overlapping_segment_count{1};

  pairwise_linestring_intersection_upper_bound_count(multilinestrings1.range(),
                                                     multilinestrings2.range(),
                                                     num_intersecting_points.begin(),
                                                     num_overlapping_segments.begin(),
                                                     this->default_stream());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(num_intersecting_points, expected_intersecting_points_count);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(num_overlapping_segments, expected_overlapping_segment_count);
}

TYPED_TEST(LinestringIntersectionCountTest, SingleToSingleOverlapMultipleSegment)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using count_type = unsigned;

  // The "upper bound intersection count" between
  // (0, 0) -> (1, 1) and (-1, -1) -> (0.25, 0.25) -> (0.25, 0.0) is
  // 2 intersection point(s) (0.25, 0.25) and (0.75, 0.75)
  // 2 overlapping segment(s) (0.0, 0.0) -> (0.25, 0.25), (0.75, 0.75) -> (1.0, 1.0)

  auto multilinestrings1 = make_multilinestring_array({0, 1}, {0, 2}, {P{0, 0}, P{1, 1}});

  auto multilinestrings2 = make_multilinestring_array(
    {0, 1}, {0, 5}, {P{-1, -1}, P{0.25, 0.25}, P{0.25, 0.0}, P{0.75, 0.75}, P{1.5, 1.5}});

  rmm::device_vector<count_type> num_intersecting_points(multilinestrings1.size());
  rmm::device_vector<count_type> num_overlapping_segments(multilinestrings1.size());

  std::vector<count_type> expected_intersecting_points_count{2};
  std::vector<count_type> expected_overlapping_segment_count{2};

  pairwise_linestring_intersection_upper_bound_count(multilinestrings1.range(),
                                                     multilinestrings2.range(),
                                                     num_intersecting_points.begin(),
                                                     num_overlapping_segments.begin(),
                                                     this->default_stream());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(num_intersecting_points, expected_intersecting_points_count);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(num_overlapping_segments, expected_overlapping_segment_count);
}

TYPED_TEST(LinestringIntersectionCountTest, SingleToSingleSimpleIntersectAndOverlap)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using count_type = unsigned;

  // The "upper bound intersection count" between
  // (0.25, 0) -> (0.25, 0.5) -> (0.75, 0.75) -> (1.5, 1.5) and (0, 0) -> (1, 1)
  // 2 intersection point(s) (0.25, 0.25) and (0.75, 0.75)
  // 1 overlapping segment(s) (0.75, 0.75) -> (1.0, 1.0)

  auto multilinestrings1 = make_multilinestring_array({0, 1}, {0, 2}, {P{0, 0}, P{1, 1}});

  auto multilinestrings2 = make_multilinestring_array(
    {0, 1}, {0, 4}, {P{0.25, 0.0}, P{0.25, 0.5}, P{0.75, 0.75}, P{1.5, 1.5}});

  rmm::device_vector<count_type> num_intersecting_points(multilinestrings1.size());
  rmm::device_vector<count_type> num_overlapping_segments(multilinestrings1.size());

  std::vector<count_type> expected_intersecting_points_count{2};
  std::vector<count_type> expected_overlapping_segment_count{1};

  pairwise_linestring_intersection_upper_bound_count(multilinestrings1.range(),
                                                     multilinestrings2.range(),
                                                     num_intersecting_points.begin(),
                                                     num_overlapping_segments.begin(),
                                                     this->default_stream());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(num_intersecting_points, expected_intersecting_points_count);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(num_overlapping_segments, expected_overlapping_segment_count);
}

TYPED_TEST(LinestringIntersectionCountTest, SingleToSingleSimpleDisjoint)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using count_type = unsigned;

  auto multilinestrings1 = make_multilinestring_array({0, 1}, {0, 2}, {P{0, 0}, P{1, 1}});

  auto multilinestrings2 = make_multilinestring_array({0, 1}, {0, 2}, {P{2, 2}, P{3, 3}});

  rmm::device_vector<count_type> num_intersecting_points(multilinestrings1.size());
  rmm::device_vector<count_type> num_overlapping_segments(multilinestrings1.size());

  std::vector<count_type> expected_intersecting_points_count{0};
  std::vector<count_type> expected_overlapping_segment_count{0};

  pairwise_linestring_intersection_upper_bound_count(multilinestrings1.range(),
                                                     multilinestrings2.range(),
                                                     num_intersecting_points.begin(),
                                                     num_overlapping_segments.begin(),
                                                     this->default_stream());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(num_intersecting_points, expected_intersecting_points_count);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(num_overlapping_segments, expected_overlapping_segment_count);
}

TYPED_TEST(LinestringIntersectionCountTest, SingleToSingleSimpleDisjoint2)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using count_type = unsigned;

  auto multilinestrings1 = make_multilinestring_array({0, 1}, {0, 2}, {P{0, 0}, P{1, 1}});

  auto multilinestrings2 = make_multilinestring_array({0, 1}, {0, 2}, {P{1, 0}, P{2, 0}});

  rmm::device_vector<count_type> num_intersecting_points(multilinestrings1.size());
  rmm::device_vector<count_type> num_overlapping_segments(multilinestrings1.size());

  std::vector<count_type> expected_intersecting_points_count{0};
  std::vector<count_type> expected_overlapping_segment_count{0};

  pairwise_linestring_intersection_upper_bound_count(multilinestrings1.range(),
                                                     multilinestrings2.range(),
                                                     num_intersecting_points.begin(),
                                                     num_overlapping_segments.begin(),
                                                     this->default_stream());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(num_intersecting_points, expected_intersecting_points_count);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(num_overlapping_segments, expected_overlapping_segment_count);
}

// FIXME
TYPED_TEST(LinestringIntersectionCountTest, SingleToSingleIntersectOverlapSameVertex)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using count_type = unsigned;

  // The "upper bound intersection count" between
  // (0, 1) -> (1, 0) -> (1, 1) -> (2, 1.5) and (0, 0) -> (1, 1) -> (2, 1)
  // 5 intersection point(s) (1, 1) (4 times), (0.5, 0.5)
  // 0 overlapping segment(s)

  auto stream = this->default_stream();

  auto multilinestrings1 = make_multilinestring_array({0, 1}, {0, 3}, {P{0, 0}, P{1, 1}, P{2, 1}});

  auto multilinestrings2 =
    make_multilinestring_array({0, 1}, {0, 4}, {P{0, 1}, P{1, 0}, P{1, 1}, P{2, 1.5}});

  rmm::device_uvector<count_type> num_intersecting_points(multilinestrings1.size(), stream);
  rmm::device_uvector<count_type> num_overlapping_segments(multilinestrings1.size(), stream);

  thrust::uninitialized_fill_n(
    rmm::exec_policy(stream), num_intersecting_points.begin(), multilinestrings1.size(), 0);
  thrust::uninitialized_fill_n(
    rmm::exec_policy(stream), num_overlapping_segments.begin(), multilinestrings1.size(), 0);

  std::vector<count_type> expected_intersecting_points_count{5};
  std::vector<count_type> expected_overlapping_segment_count{0};

  pairwise_linestring_intersection_upper_bound_count(multilinestrings1.range(),
                                                     multilinestrings2.range(),
                                                     num_intersecting_points.begin(),
                                                     num_overlapping_segments.begin(),
                                                     this->default_stream());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(num_intersecting_points, expected_intersecting_points_count);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(num_overlapping_segments, expected_overlapping_segment_count);
}

TYPED_TEST(LinestringIntersectionCountTest, SingleToSingleExample)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  // First pair:
  // (0, 0), (1, 1)
  // Second pair:
  // (1, 0), (0,1)
  // (0.5, 0), (0, 0.5), (1, 0.5)
  // (0.5, 0.5), (1.5, 1.5)
  // (-1, -1), (0.25, 0.25), (0.25, 0.0), (0.75, 0.75), (1.5, 1.5)
  // (0.25, 0.0), (0.25, 0.5), (0.75, 0.75), (1.5, 1.5)
  // (2,2), (3,3)
  // (1, 0), (2, 0)
  // Result:
  // intersecting points (upper bound):
  // 1, 2, 0, 2, 2, 0, 0
  //          ^  ^
  // row[3] points: (0.25, 0.25) and (0.75, 0.75)
  // row[4] points: (0.25, 0.25) and (0.75, 0.75)
  // overlapping segments (upper bound):
  // 0, 0, 1, 2, 1, 0, 0

  auto multilinestrings1 = make_multilinestring_array({0, 1, 2, 3, 4, 5, 6, 7},
                                                      {0, 2, 4, 6, 8, 10, 12, 14},
                                                      {P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1},
                                                       P{0, 0},
                                                       P{1, 1}});

  auto multilinestrings2 = make_multilinestring_array(
    {0, 1, 2, 3, 4, 5, 6, 7},
    {0, 2, 5, 7, 12, 16, 18, 20},
    {P{1, 0},       P{0, 1},     P{0.5, 0},    P{0, 0.5},     P{1, 0.5},
     P{0.5, 0.5},   P{1.5, 1.5}, P{-1, -1},    P{0.25, 0.25}, P{0.25, 0.0},
     P{0.75, 0.75}, P{1.5, 1.5}, P{0.25, 0.0}, P{0.25, 0.5},  P{0.75, 0.75},
     P{1.5, 1.5},   P{2, 2},     P{3, 3},      P{1, 0},       P{2, 0}});

  rmm::device_vector<unsigned> num_intersecting_points(multilinestrings1.size(), 0);
  rmm::device_vector<unsigned> num_overlapping_segments(multilinestrings1.size(), 0);

  std::vector<unsigned> expected_intersecting_points_count{1, 2, 0, 2, 2, 0, 0};
  std::vector<unsigned> expected_overlapping_segment_count{0, 0, 1, 2, 1, 0, 0};

  pairwise_linestring_intersection_upper_bound_count(multilinestrings1.range(),
                                                     multilinestrings2.range(),
                                                     num_intersecting_points.begin(),
                                                     num_overlapping_segments.begin(),
                                                     this->default_stream());

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(num_intersecting_points, expected_intersecting_points_count);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(num_overlapping_segments, expected_overlapping_segment_count);
}
