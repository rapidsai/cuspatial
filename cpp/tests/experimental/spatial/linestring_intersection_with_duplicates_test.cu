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
#include <cuspatial/experimental/detail/linestring_intersection_with_duplicates.cuh>
#include <cuspatial/experimental/iterator_factory.cuh>
#include <cuspatial/traits.hpp>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <thrust/iterator/zip_iterator.h>

#include <initializer_list>
#include <type_traits>

using namespace cuspatial;
using namespace cuspatial::test;

template <typename SegmentVector, typename T = typename SegmentVector::value_type::value_type>
std::pair<rmm::device_vector<vec_2d<T>>, rmm::device_vector<vec_2d<T>>> unpack_segment_vector(
  SegmentVector const& segments)
{
  rmm::device_vector<vec_2d<T>> first(segments.size()), second(segments.size());
  auto zipped_output = thrust::make_zip_iterator(first.begin(), second.begin());
  thrust::transform(
    segments.begin(), segments.end(), zipped_output, [] __device__(segment<T> const& segment) {
      return thrust::make_tuple(segment.first, segment.second);
    });
  return {std::move(first), std::move(second)};
}

template <typename SegmentVector1, typename SegmentVector2>
void expect_segment_equivalent(SegmentVector1 expected, SegmentVector2 got)
{
  auto [expected_first, expected_second] = unpack_segment_vector(expected);
  auto [got_first, got_second]           = unpack_segment_vector(got);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_first, got_first);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_second, got_second);
}

template <typename T>
struct LinestringIntersectionDuplicatesTest : public ::testing::Test {
  rmm::cuda_stream_view stream() { return rmm::cuda_stream_default; }
  rmm::mr::device_memory_resource* mr() { return rmm::mr::get_current_device_resource(); }
};

// float and double are logically the same but would require seperate tests due to precision.
using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(LinestringIntersectionDuplicatesTest, TestTypes);

// TODO: sort the points in the intersection result since the result order is arbitrary.
TYPED_TEST(LinestringIntersectionDuplicatesTest, Example)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = std::size_t;

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

  auto [points, segments] = detail::pairwise_linestring_intersection_with_duplicate<index_t, T>(
    multilinestrings1.range(), multilinestrings2.range(), this->mr(), this->stream());

  auto expected_points_offsets   = make_device_vector<index_t>({0, 1, 3, 3, 5, 7, 7, 7});
  auto expected_points_coords    = make_device_vector<P>({P{0.5, 0.5},
                                                          P{0.25, 0.25},
                                                          P{0.5, 0.5},
                                                          P{0.25, 0.25},
                                                          P{0.75, 0.75},
                                                          P{0.25, 0.25},
                                                          P{0.75, 0.75}});
  auto expected_segments_offsets = make_device_vector<index_t>({0, 0, 0, 1, 3, 4, 4, 4});
  auto expected_segments_coords =
    make_device_vector<segment<T>>({segment<T>{P{0.5, 0.5}, P{1, 1}},
                                    segment<T>{P{0, 0}, P{0.25, 0.25}},
                                    segment<T>{P{0.75, 0.75}, P{1, 1}},
                                    segment<T>{P{0.75, 0.75}, P{1, 1}}});

  auto expected_point_lhs_linestring_ids = make_device_vector<index_t>({0, 0, 0, 0, 0, 0, 0});
  auto expected_point_lhs_segment_ids    = make_device_vector<index_t>({0, 0, 0, 0, 0, 0, 0});

  auto expected_point_rhs_linestring_ids = make_device_vector<index_t>({0, 0, 0, 0, 0, 0, 0});
  auto expected_point_rhs_segment_ids    = make_device_vector<index_t>({0, 0, 1, 1, 2, 0, 1});

  auto expected_segment_lhs_linestring_ids = make_device_vector<index_t>({0, 0, 0, 0});
  auto expected_segment_lhs_segment_ids    = make_device_vector<index_t>({0, 0, 0, 0});

  auto expected_segment_rhs_linestring_ids = make_device_vector<index_t>({0, 0, 0, 0});
  auto expected_segment_rhs_segment_ids    = make_device_vector<index_t>({0, 0, 3, 2});

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_points_offsets, std::move(points.offsets));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_points_coords, std::move(points.geoms));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_segments_offsets, std::move(segments.offsets));
  expect_segment_equivalent(expected_segments_coords, std::move(segments.geoms));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_point_lhs_linestring_ids,
                                      std::move(points.lhs_linestring_ids));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_point_lhs_segment_ids,
                                      std::move(points.lhs_segment_ids));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_point_rhs_linestring_ids,
                                      std::move(points.rhs_linestring_ids));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_point_rhs_segment_ids,
                                      std::move(points.rhs_segment_ids));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_segment_lhs_linestring_ids,
                                      std::move(segments.lhs_linestring_ids));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_segment_lhs_segment_ids,
                                      std::move(segments.lhs_segment_ids));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_segment_rhs_linestring_ids,
                                      std::move(segments.rhs_linestring_ids));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_segment_rhs_segment_ids,
                                      std::move(segments.rhs_segment_ids));
}

// Same Test Case as above, reversing the order of multilinestrings1 and multilinestrings2
TYPED_TEST(LinestringIntersectionDuplicatesTest, ExampleReversed)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = std::size_t;

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

  auto [points, segments] = detail::pairwise_linestring_intersection_with_duplicate<index_t, T>(
    multilinestrings2.range(), multilinestrings1.range(), this->mr(), this->stream());

  auto expected_points_offsets   = make_device_vector<index_t>({0, 1, 3, 3, 5, 7, 7, 7});
  auto expected_points_coords    = make_device_vector<P>({P{0.5, 0.5},
                                                          P{0.25, 0.25},
                                                          P{0.5, 0.5},
                                                          P{0.25, 0.25},
                                                          P{0.75, 0.75},
                                                          P{0.25, 0.25},
                                                          P{0.75, 0.75}});
  auto expected_segments_offsets = make_device_vector<index_t>({0, 0, 0, 1, 3, 4, 4, 4});
  auto expected_segments_coords =
    make_device_vector<segment<T>>({segment<T>{P{0.5, 0.5}, P{1, 1}},
                                    segment<T>{P{0, 0}, P{0.25, 0.25}},
                                    segment<T>{P{0.75, 0.75}, P{1, 1}},
                                    segment<T>{P{0.75, 0.75}, P{1, 1}}});

  auto expected_point_rhs_linestring_ids = make_device_vector<index_t>({0, 0, 0, 0, 0, 0, 0});
  auto expected_point_rhs_segment_ids    = make_device_vector<index_t>({0, 0, 0, 0, 0, 0, 0});

  auto expected_point_lhs_linestring_ids = make_device_vector<index_t>({0, 0, 0, 0, 0, 0, 0});
  auto expected_point_lhs_segment_ids    = make_device_vector<index_t>({0, 0, 1, 1, 2, 0, 1});

  auto expected_segment_rhs_linestring_ids = make_device_vector<index_t>({0, 0, 0, 0});
  auto expected_segment_rhs_segment_ids    = make_device_vector<index_t>({0, 0, 0, 0});

  auto expected_segment_lhs_linestring_ids = make_device_vector<index_t>({0, 0, 0, 0});
  auto expected_segment_lhs_segment_ids    = make_device_vector<index_t>({0, 0, 3, 2});

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_points_offsets, std::move(points.offsets));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_points_coords, std::move(points.geoms));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_segments_offsets, std::move(segments.offsets));
  expect_segment_equivalent(expected_segments_coords, std::move(segments.geoms));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_point_lhs_linestring_ids,
                                      std::move(points.lhs_linestring_ids));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_point_lhs_segment_ids,
                                      std::move(points.lhs_segment_ids));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_point_rhs_linestring_ids,
                                      std::move(points.rhs_linestring_ids));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_point_rhs_segment_ids,
                                      std::move(points.rhs_segment_ids));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_segment_lhs_linestring_ids,
                                      std::move(segments.lhs_linestring_ids));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_segment_lhs_segment_ids,
                                      std::move(segments.lhs_segment_ids));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_segment_rhs_linestring_ids,
                                      std::move(segments.rhs_linestring_ids));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_segment_rhs_segment_ids,
                                      std::move(segments.rhs_segment_ids));
}

TYPED_TEST(LinestringIntersectionDuplicatesTest, MultilinestringsIntersectionWithDuplicates)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = std::size_t;

  auto multilinestrings1 = make_multilinestring_array(
    {0, 2, 5},
    {0, 2, 4, 6, 8, 10},
    {P{0, 0}, P{1, 1}, P{1, 0}, P{2, 1}, P{0, 2}, P{1, 2}, P{0, 3}, P{0, 2}, P{0, 3}, P{1, 2}});

  auto multilinestrings2 =
    make_multilinestring_array({0, 1, 2}, {0, 2, 4}, {P{0, 1}, P{2, 0}, P{0, 2.5}, P{1, 2.5}});

  auto [points, segments] = detail::pairwise_linestring_intersection_with_duplicate<index_t, T>(
    multilinestrings1.range(), multilinestrings2.range(), this->mr(), this->stream());

  auto expected_points_offsets = make_device_vector<index_t>({0, 2, 4});
  auto expected_points_coords =
    make_device_vector<P>({P{2 / 3., 2 / 3.}, P{4 / 3., 1 / 3.}, P{0, 2.5}, P{0.5, 2.5}});

  auto expected_segments_offsets = make_device_vector<index_t>({0, 0, 0});
  auto expected_segments_coords  = make_device_vector<segment<T>>({});

  auto expected_point_lhs_linestring_ids = make_device_vector<index_t>({0, 1, 1, 2});
  auto expected_point_lhs_segment_ids    = make_device_vector<index_t>({0, 0, 0, 0});

  auto expected_point_rhs_linestring_ids = make_device_vector<index_t>({0, 0, 0, 0});
  auto expected_point_rhs_segment_ids    = make_device_vector<index_t>({0, 0, 0, 0});

  auto expected_segment_lhs_linestring_ids = make_device_vector<index_t>({});
  auto expected_segment_lhs_segment_ids    = make_device_vector<index_t>({});

  auto expected_segment_rhs_linestring_ids = make_device_vector<index_t>({});
  auto expected_segment_rhs_segment_ids    = make_device_vector<index_t>({});

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_points_offsets, std::move(points.offsets));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_points_coords, std::move(points.geoms));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_segments_offsets, std::move(segments.offsets));
  expect_segment_equivalent(expected_segments_coords, std::move(segments.geoms));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_point_lhs_linestring_ids,
                                      std::move(points.lhs_linestring_ids));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_point_lhs_segment_ids,
                                      std::move(points.lhs_segment_ids));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_point_rhs_linestring_ids,
                                      std::move(points.rhs_linestring_ids));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_point_rhs_segment_ids,
                                      std::move(points.rhs_segment_ids));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_segment_lhs_linestring_ids,
                                      std::move(segments.lhs_linestring_ids));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_segment_lhs_segment_ids,
                                      std::move(segments.lhs_segment_ids));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_segment_rhs_linestring_ids,
                                      std::move(segments.rhs_linestring_ids));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_segment_rhs_segment_ids,
                                      std::move(segments.rhs_segment_ids));
}

TYPED_TEST(LinestringIntersectionDuplicatesTest, MultilinestringsIntersectionWithDuplicatesReversed)
{
  using T = TypeParam;
  using P = vec_2d<T>;

  using index_t = std::size_t;

  auto multilinestrings1 = make_multilinestring_array(
    {0, 2, 5},
    {0, 2, 4, 6, 8, 10},
    {P{0, 0}, P{1, 1}, P{1, 0}, P{2, 1}, P{0, 2}, P{1, 2}, P{0, 3}, P{0, 2}, P{0, 3}, P{1, 2}});

  auto multilinestrings2 =
    make_multilinestring_array({0, 1, 2}, {0, 2, 4}, {P{0, 1}, P{2, 0}, P{0, 2.5}, P{1, 2.5}});

  auto [points, segments] = detail::pairwise_linestring_intersection_with_duplicate<index_t, T>(
    multilinestrings2.range(), multilinestrings1.range(), this->mr(), this->stream());

  auto expected_points_offsets = make_device_vector<index_t>({0, 2, 4});
  auto expected_points_coords =
    make_device_vector<P>({P{2 / 3., 2 / 3.}, P{4 / 3., 1 / 3.}, P{0, 2.5}, P{0.5, 2.5}});

  auto expected_segments_offsets = make_device_vector<index_t>({0, 0, 0});
  auto expected_segments_coords  = make_device_vector<segment<T>>({});

  auto expected_point_rhs_linestring_ids = make_device_vector<index_t>({0, 1, 1, 2});
  auto expected_point_rhs_segment_ids    = make_device_vector<index_t>({0, 0, 0, 0});

  auto expected_point_lhs_linestring_ids = make_device_vector<index_t>({0, 0, 0, 0});
  auto expected_point_lhs_segment_ids    = make_device_vector<index_t>({0, 0, 0, 0});

  auto expected_segment_rhs_linestring_ids = make_device_vector<index_t>({});
  auto expected_segment_rhs_segment_ids    = make_device_vector<index_t>({});

  auto expected_segment_lhs_linestring_ids = make_device_vector<index_t>({});
  auto expected_segment_lhs_segment_ids    = make_device_vector<index_t>({});

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_points_offsets, std::move(points.offsets));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_points_coords, std::move(points.geoms));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_segments_offsets, std::move(segments.offsets));
  expect_segment_equivalent(expected_segments_coords, std::move(segments.geoms));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_point_lhs_linestring_ids,
                                      std::move(points.lhs_linestring_ids));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_point_lhs_segment_ids,
                                      std::move(points.lhs_segment_ids));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_point_rhs_linestring_ids,
                                      std::move(points.rhs_linestring_ids));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_point_rhs_segment_ids,
                                      std::move(points.rhs_segment_ids));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_segment_lhs_linestring_ids,
                                      std::move(segments.lhs_linestring_ids));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_segment_lhs_segment_ids,
                                      std::move(segments.lhs_segment_ids));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_segment_rhs_linestring_ids,
                                      std::move(segments.rhs_linestring_ids));
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_segment_rhs_segment_ids,
                                      std::move(segments.rhs_segment_ids));
}
