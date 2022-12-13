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

#include <cuspatial_test/base_fixture.hpp>
#include <cuspatial_test/vector_equality.hpp>

#include <cuspatial/cuda_utils.hpp>
#include <cuspatial/detail/utility/linestring.cuh>
#include <cuspatial/error.hpp>
#include <cuspatial/vec_2d.hpp>

#include <rmm/device_vector.hpp>

#include <thrust/execution_policy.h>
#include <thrust/optional.h>
#include <thrust/pair.h>

#include <gtest/gtest.h>

#include <optional>

using namespace cuspatial;
using namespace cuspatial::detail;
using namespace cuspatial::test;

template <typename T>
using optional_vec2d = thrust::optional<vec_2d<T>>;

template <typename T>
struct SegmentIntersectionTest : public BaseFixture {
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(SegmentIntersectionTest, TestTypes);

template <typename T>
segment<T> __device__ order_end_points(segment<T> const& seg)
{
  auto [a, b] = seg;
  return a < b ? segment<T>{a, b} : segment<T>{b, a};
}

template <typename T, typename Point, typename Segment>
void __global__
compute_intersection(segment<T> ab, segment<T> cd, Point point_out, Segment segment_out)
{
  auto [p, s]    = detail::segment_intersection(ab, cd);
  point_out[0]   = p;
  segment_out[0] = s.has_value() ? thrust::optional(order_end_points(s.value())) : s;
}

template <typename T>
struct unpack_optional_segment {
  thrust::tuple<optional_vec2d<T>, optional_vec2d<T>> CUSPATIAL_HOST_DEVICE
  operator()(thrust::optional<segment<T>> segment)
  {
    if (segment.has_value())
      return thrust::make_tuple(segment.value().v1, segment.value().v2);
    else
      return thrust::tuple<optional_vec2d<T>, optional_vec2d<T>>{thrust::nullopt, thrust::nullopt};
  }
};

template <typename T>
void run_single_intersection_test(
  segment<T> const& ab,
  segment<T> const& cd,
  std::vector<thrust::optional<vec_2d<T>>> const& points_expected,
  std::vector<thrust::optional<segment<T>>> const& segments_expected)
{
  rmm::device_vector<thrust::optional<vec_2d<T>>> points_got(points_expected.size());
  rmm::device_vector<thrust::optional<segment<T>>> segments_got(segments_expected.size());

  compute_intersection<<<1, 1>>>(ab, cd, points_got.data(), segments_got.data());

  // Unpack the segment into two separate optional vec_2d column.
  rmm::device_vector<thrust::optional<vec_2d<T>>> first(segments_got.size());
  rmm::device_vector<thrust::optional<vec_2d<T>>> second(segments_got.size());
  auto outit = thrust::make_zip_iterator(first.begin(), second.begin());

  thrust::transform(segments_got.begin(), segments_got.end(), outit, unpack_optional_segment<T>{});

  std::vector<thrust::optional<vec_2d<T>>> expected_first(segments_expected.size());
  std::vector<thrust::optional<vec_2d<T>>> expected_second(segments_expected.size());
  auto h_outit = thrust::make_zip_iterator(expected_first.begin(), expected_second.begin());

  thrust::transform(thrust::host,
                    segments_expected.begin(),
                    segments_expected.end(),
                    h_outit,
                    unpack_optional_segment<T>{});

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(points_got, points_expected);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(first, expected_first);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(second, expected_second);
}

TYPED_TEST(SegmentIntersectionTest, SimpleIntersect)
{
  using T = TypeParam;

  segment<T> ab{{0.0, 0.0}, {1.0, 1.0}};
  segment<T> cd{{0.0, 1.0}, {1.0, 0.0}};

  std::vector<thrust::optional<vec_2d<T>>> points_expected{vec_2d<T>{0.5, 0.5}};
  std::vector<thrust::optional<segment<T>>> segments_expected{thrust::nullopt};

  run_single_intersection_test(ab, cd, points_expected, segments_expected);
}

TYPED_TEST(SegmentIntersectionTest, IntersectAtEndPoint)
{
  using T = TypeParam;

  segment<T> ab{{0.0, 0.0}, {1.0, 1.0}};
  segment<T> cd{{1.0, 1.0}, {1.0, 0.0}};

  std::vector<thrust::optional<vec_2d<T>>> points_expected{vec_2d<T>{1.0, 1.0}};
  std::vector<thrust::optional<segment<T>>> segments_expected{thrust::nullopt};

  run_single_intersection_test(ab, cd, points_expected, segments_expected);
}

TYPED_TEST(SegmentIntersectionTest, UnparallelDisjoint1)
{
  using T = TypeParam;

  segment<T> ab{{0.0, 0.0}, {0.4, 1.0}};
  segment<T> cd{{1.0, 0.0}, {0.6, 1.0}};

  std::vector<thrust::optional<vec_2d<T>>> points_expected{thrust::nullopt};
  std::vector<thrust::optional<segment<T>>> segments_expected{thrust::nullopt};

  run_single_intersection_test(ab, cd, points_expected, segments_expected);
}

TYPED_TEST(SegmentIntersectionTest, UnparallelDisjoint2)
{
  using T = TypeParam;

  segment<T> ab{{0.0, 0.0}, {1.0, 0.0}};
  segment<T> cd{{2.0, 0.0}, {2.0, 1.0}};

  std::vector<thrust::optional<vec_2d<T>>> points_expected{thrust::nullopt};
  std::vector<thrust::optional<segment<T>>> segments_expected{thrust::nullopt};

  run_single_intersection_test(ab, cd, points_expected, segments_expected);
}

TYPED_TEST(SegmentIntersectionTest, ParallelDisjoint1)
{
  using T = TypeParam;

  segment<T> ab{{0.0, 0.0}, {0.0, 1.0}};
  segment<T> cd{{1.0, 0.0}, {1.0, 1.0}};

  std::vector<thrust::optional<vec_2d<T>>> points_expected{thrust::nullopt};
  std::vector<thrust::optional<segment<T>>> segments_expected{thrust::nullopt};

  run_single_intersection_test(ab, cd, points_expected, segments_expected);
}

TYPED_TEST(SegmentIntersectionTest, ParallelDisjoint2)
{
  using T = TypeParam;

  segment<T> ab{{0.0, 0.0}, {1.0, 0.0}};
  segment<T> cd{{0.0, 1.0}, {1.0, 1.0}};

  std::vector<thrust::optional<vec_2d<T>>> points_expected{thrust::nullopt};
  std::vector<thrust::optional<segment<T>>> segments_expected{thrust::nullopt};

  run_single_intersection_test(ab, cd, points_expected, segments_expected);
}

TYPED_TEST(SegmentIntersectionTest, ParallelDisjoint3)
{
  using T = TypeParam;

  segment<T> ab{{0.0, 0.0}, {1.0, 1.0}};
  segment<T> cd{{1.0, 0.0}, {2.0, 1.0}};

  std::vector<thrust::optional<vec_2d<T>>> points_expected{thrust::nullopt};
  std::vector<thrust::optional<segment<T>>> segments_expected{thrust::nullopt};

  run_single_intersection_test(ab, cd, points_expected, segments_expected);
}

TYPED_TEST(SegmentIntersectionTest, ParallelDisjoint4)
{
  using T = TypeParam;

  segment<T> ab{{0.0, 0.0}, {0.0, -1.0}};
  segment<T> cd{{1.0, 0.0}, {1.0, 1.0}};

  std::vector<thrust::optional<vec_2d<T>>> points_expected{thrust::nullopt};
  std::vector<thrust::optional<segment<T>>> segments_expected{thrust::nullopt};

  run_single_intersection_test(ab, cd, points_expected, segments_expected);
}

TYPED_TEST(SegmentIntersectionTest, CollinearDisjoint1)
{
  using T = TypeParam;

  segment<T> ab{{0.0, 0.0}, {1.0, 0.0}};
  segment<T> cd{{2.0, 0.0}, {3.0, 0.0}};

  std::vector<thrust::optional<vec_2d<T>>> points_expected{thrust::nullopt};
  std::vector<thrust::optional<segment<T>>> segments_expected{thrust::nullopt};

  run_single_intersection_test(ab, cd, points_expected, segments_expected);
}

TYPED_TEST(SegmentIntersectionTest, CollinearDisjoint2)
{
  using T = TypeParam;

  segment<T> ab{{0.0, 0.0}, {1.0, 0.0}};
  segment<T> cd{{-1.0, 0.0}, {-2.0, 0.0}};

  std::vector<thrust::optional<vec_2d<T>>> points_expected{thrust::nullopt};
  std::vector<thrust::optional<segment<T>>> segments_expected{thrust::nullopt};

  run_single_intersection_test(ab, cd, points_expected, segments_expected);
}

TYPED_TEST(SegmentIntersectionTest, CollinearDisjoint3)
{
  using T = TypeParam;

  segment<T> ab{{0.0, 0.0}, {0.0, 1.0}};
  segment<T> cd{{0.0, 2.0}, {0.0, 3.0}};

  std::vector<thrust::optional<vec_2d<T>>> points_expected{thrust::nullopt};
  std::vector<thrust::optional<segment<T>>> segments_expected{thrust::nullopt};

  run_single_intersection_test(ab, cd, points_expected, segments_expected);
}

TYPED_TEST(SegmentIntersectionTest, CollinearDisjoint4)
{
  using T = TypeParam;

  segment<T> ab{{0.0, 0.0}, {0.0, 1.0}};
  segment<T> cd{{0.0, -1.0}, {0.0, -2.0}};

  std::vector<thrust::optional<vec_2d<T>>> points_expected{thrust::nullopt};
  std::vector<thrust::optional<segment<T>>> segments_expected{thrust::nullopt};

  run_single_intersection_test(ab, cd, points_expected, segments_expected);
}

TYPED_TEST(SegmentIntersectionTest, CollinearDisjoint5)
{
  using T = TypeParam;

  segment<T> ab{{0.0, 0.0}, {1.0, 1.0}};
  segment<T> cd{{2.0, 2.0}, {3.0, 3.0}};

  std::vector<thrust::optional<vec_2d<T>>> points_expected{thrust::nullopt};
  std::vector<thrust::optional<segment<T>>> segments_expected{thrust::nullopt};

  run_single_intersection_test(ab, cd, points_expected, segments_expected);
}

TYPED_TEST(SegmentIntersectionTest, CollinearDisjoint6)
{
  using T = TypeParam;

  segment<T> ab{{0.0, 0.0}, {1.0, 1.0}};
  segment<T> cd{{-1.0, -1.0}, {-2.0, -2.0}};

  std::vector<thrust::optional<vec_2d<T>>> points_expected{thrust::nullopt};
  std::vector<thrust::optional<segment<T>>> segments_expected{thrust::nullopt};

  run_single_intersection_test(ab, cd, points_expected, segments_expected);
}

TYPED_TEST(SegmentIntersectionTest, Overlap1)
{
  using T = TypeParam;

  segment<T> ab{{0.0, 0.0}, {1.0, 1.0}};
  segment<T> cd{{0.5, 0.5}, {1.5, 1.5}};

  std::vector<thrust::optional<vec_2d<T>>> points_expected{thrust::nullopt};
  std::vector<thrust::optional<segment<T>>> segments_expected{segment<T>{{0.5, 0.5}, {1.0, 1.0}}};

  run_single_intersection_test(ab, cd, points_expected, segments_expected);
}

TYPED_TEST(SegmentIntersectionTest, Overlap2)
{
  using T = TypeParam;

  segment<T> ab{{0.0, 0.0}, {1.0, 1.0}};
  segment<T> cd{{0.5, 0.5}, {-1.5, -1.5}};

  std::vector<thrust::optional<vec_2d<T>>> points_expected{thrust::nullopt};
  std::vector<thrust::optional<segment<T>>> segments_expected{segment<T>{{0.0, 0.0}, {0.5, 0.5}}};

  run_single_intersection_test(ab, cd, points_expected, segments_expected);
}

TYPED_TEST(SegmentIntersectionTest, Overlap3)
{
  using T = TypeParam;

  segment<T> ab{{0.0, 0.0}, {1.0, 0.0}};
  segment<T> cd{{0.5, 0.0}, {2.0, 0.0}};

  std::vector<thrust::optional<vec_2d<T>>> points_expected{thrust::nullopt};
  std::vector<thrust::optional<segment<T>>> segments_expected{segment<T>{{0.5, 0.0}, {1.0, 0.0}}};

  run_single_intersection_test(ab, cd, points_expected, segments_expected);
}

TYPED_TEST(SegmentIntersectionTest, Overlap4)
{
  using T = TypeParam;

  segment<T> ab{{0.0, 0.0}, {1.0, 0.0}};
  segment<T> cd{{0.5, 0.0}, {-1.0, 0.0}};

  std::vector<thrust::optional<vec_2d<T>>> points_expected{thrust::nullopt};
  std::vector<thrust::optional<segment<T>>> segments_expected{segment<T>{{0.0, 0.0}, {0.5, 0.0}}};

  run_single_intersection_test(ab, cd, points_expected, segments_expected);
}

TYPED_TEST(SegmentIntersectionTest, Overlap5)
{
  using T = TypeParam;

  segment<T> ab{{0.0, 0.0}, {0.0, 1.0}};
  segment<T> cd{{0.0, 0.5}, {0.0, 2.0}};

  std::vector<thrust::optional<vec_2d<T>>> points_expected{thrust::nullopt};
  std::vector<thrust::optional<segment<T>>> segments_expected{segment<T>{{0.0, 0.5}, {0.0, 1.0}}};

  run_single_intersection_test(ab, cd, points_expected, segments_expected);
}

TYPED_TEST(SegmentIntersectionTest, Overlap6)
{
  using T = TypeParam;

  segment<T> ab{{0.0, 0.0}, {0.0, 1.0}};
  segment<T> cd{{0.0, 0.5}, {0.0, -2.0}};

  std::vector<thrust::optional<vec_2d<T>>> points_expected{thrust::nullopt};
  std::vector<thrust::optional<segment<T>>> segments_expected{segment<T>{{0.0, 0.0}, {0.0, 0.5}}};

  run_single_intersection_test(ab, cd, points_expected, segments_expected);
}

TYPED_TEST(SegmentIntersectionTest, Overlap7)
{
  using T = TypeParam;

  segment<T> ab{{0.0, 0.0}, {0.0, 1.0}};
  segment<T> cd{{0.0, 0.0}, {0.0, 0.5}};

  std::vector<thrust::optional<vec_2d<T>>> points_expected{thrust::nullopt};
  std::vector<thrust::optional<segment<T>>> segments_expected{segment<T>{{0.0, 0.0}, {0.0, 0.5}}};

  run_single_intersection_test(ab, cd, points_expected, segments_expected);
}

TYPED_TEST(SegmentIntersectionTest, Overlap8)
{
  using T = TypeParam;

  segment<T> ab{{0.0, 0.0}, {0.0, 1.0}};
  segment<T> cd{{0.0, 0.5}, {0.0, 1.0}};

  std::vector<thrust::optional<vec_2d<T>>> points_expected{thrust::nullopt};
  std::vector<thrust::optional<segment<T>>> segments_expected{segment<T>{{0.0, 0.5}, {0.0, 1.0}}};

  run_single_intersection_test(ab, cd, points_expected, segments_expected);
}

TYPED_TEST(SegmentIntersectionTest, Overlap9)
{
  using T = TypeParam;

  segment<T> ab{{0.0, 0.0}, {0.0, 1.0}};
  segment<T> cd{{0.0, 0.25}, {0.0, 0.75}};

  std::vector<thrust::optional<vec_2d<T>>> points_expected{thrust::nullopt};
  std::vector<thrust::optional<segment<T>>> segments_expected{cd};

  run_single_intersection_test(ab, cd, points_expected, segments_expected);
}

TYPED_TEST(SegmentIntersectionTest, Overlap10)
{
  using T = TypeParam;

  segment<T> ab{{0.0, 0.25}, {0.0, 0.75}};
  segment<T> cd{{0.0, 0.0}, {0.0, 1.0}};

  std::vector<thrust::optional<vec_2d<T>>> points_expected{thrust::nullopt};
  std::vector<thrust::optional<segment<T>>> segments_expected{ab};

  run_single_intersection_test(ab, cd, points_expected, segments_expected);
}
