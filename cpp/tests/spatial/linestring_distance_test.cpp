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

#include <cuspatial/distances/linestring_distance.hpp>
#include <cuspatial/error.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

namespace cuspatial {
namespace test {

using namespace cudf;
using namespace cudf::test;

template <typename T>
using wrapper = fixed_width_column_wrapper<T>;

template <typename T>
struct PairwiseLinestringDistanceTest : public BaseFixture {
};

// float and double are logically the same but would require separate tests due to precision.
using TestTypes = FloatingPointTypes;
TYPED_TEST_CASE(PairwiseLinestringDistanceTest, TestTypes);

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairLinestringParallel)
{
  using T = TypeParam;
  // Linestring 1: (0.0, 0.0), (1.0, 1.0)
  // Linestring 2: (1.0, 0.0), (2.0, 1.0)
  wrapper<cudf::size_type> linestring1_offsets{0};
  wrapper<T> linestring1_points_x{0.0, 1.0};
  wrapper<T> linestring1_points_y{0.0, 1.0};
  wrapper<cudf::size_type> linestring2_offsets{0};
  wrapper<T> linestring2_points_x{1.0, 2.0};
  wrapper<T> linestring2_points_y{0.0, 1.0};

  wrapper<T> expected{0.7071067811865476};

  auto got = pairwise_linestring_distance(column_view(linestring1_offsets),
                                          linestring1_points_x,
                                          linestring1_points_y,
                                          column_view(linestring2_offsets),
                                          linestring2_points_x,
                                          linestring2_points_y);
  expect_columns_equivalent(expected, *got, verbosity);
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairLinestringEndpointsDistance)
{
  using T = TypeParam;
  // Linestring 1: (0.0, 0.0), (1.0, 1.0), (2.0, 2.0)
  // Linestring 2: (2.0, 0.0), (1.0, -1.0), (0.0, -1.0)
  wrapper<cudf::size_type> linestring1_offsets{0};
  wrapper<T> linestring1_points_x{0.0, 1.0, 2.0};
  wrapper<T> linestring1_points_y{0.0, 1.0, 2.0};
  wrapper<cudf::size_type> linestring2_offsets{0};
  wrapper<T> linestring2_points_x{2.0, 1.0, 0.0};
  wrapper<T> linestring2_points_y{0.0, -1.0, -1.0};

  wrapper<T> expected{1.0};

  auto got = pairwise_linestring_distance(column_view(linestring1_offsets),
                                          linestring1_points_x,
                                          linestring1_points_y,
                                          column_view(linestring2_offsets),
                                          linestring2_points_x,
                                          linestring2_points_y);
  expect_columns_equivalent(expected, *got, verbosity);
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairLinestringProjectionNotOnLine)
{
  using T = TypeParam;
  // Linestring 1: (0.0, 0.0), (1.0, 1.0)
  // Linestring 2: (3.0, 1.5), (3.0, 2.0)
  wrapper<cudf::size_type> linestring1_offsets{0};
  wrapper<T> linestring1_points_x{0.0, 1.0};
  wrapper<T> linestring1_points_y{0.0, 1.0};
  wrapper<cudf::size_type> linestring2_offsets{0};
  wrapper<T> linestring2_points_x{3.0, 3.0};
  wrapper<T> linestring2_points_y{1.5, 2.0};

  wrapper<T> expected{2.0615528128088303};

  auto got = pairwise_linestring_distance(column_view(linestring1_offsets),
                                          linestring1_points_x,
                                          linestring1_points_y,
                                          column_view(linestring2_offsets),
                                          linestring2_points_x,
                                          linestring2_points_y);
  expect_columns_equivalent(expected, *got, verbosity);
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairLinestringPerpendicular)
{
  using T = TypeParam;
  // Linestring 1: (0.0, 0.0), (2.0, 0.0)
  // Linestring 2: (1.0, 1.0), (1.0, 2.0)
  wrapper<cudf::size_type> linestring1_offsets{0};
  wrapper<T> linestring1_points_x{0.0, 2.0};
  wrapper<T> linestring1_points_y{0.0, 0.0};
  wrapper<cudf::size_type> linestring2_offsets{0};
  wrapper<T> linestring2_points_x{1.0, 1.0};
  wrapper<T> linestring2_points_y{1.0, 2.0};

  wrapper<T> expected{1.0};

  auto got = pairwise_linestring_distance(column_view(linestring1_offsets),
                                          linestring1_points_x,
                                          linestring1_points_y,
                                          column_view(linestring2_offsets),
                                          linestring2_points_x,
                                          linestring2_points_y);
  expect_columns_equivalent(expected, *got, verbosity);
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairLinestringIntersects)
{
  using T = TypeParam;
  // Linestring 1: (0.0, 0.0), (1.0, 1.0)
  // Linestring 2: (0.0, 1.0), (1.0, 0.0)
  wrapper<cudf::size_type> linestring1_offsets{0};
  wrapper<T> linestring1_points_x{0.0, 1.0};
  wrapper<T> linestring1_points_y{0.0, 1.0};
  wrapper<cudf::size_type> linestring2_offsets{0};
  wrapper<T> linestring2_points_x{0.0, 1.0};
  wrapper<T> linestring2_points_y{1.0, 0.0};

  wrapper<T> expected{0.0};

  auto got = pairwise_linestring_distance(column_view(linestring1_offsets),
                                          linestring1_points_x,
                                          linestring1_points_y,
                                          column_view(linestring2_offsets),
                                          linestring2_points_x,
                                          linestring2_points_y);
  expect_columns_equivalent(expected, *got, verbosity);
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairLinestringSharedVertex)
{
  using T = TypeParam;
  // Linestring 1: (0.0, 0.0), (0.0, 2.0), (2.0, 2.0)
  // Linestring 2: (2.0, 2.0), (2.0, 1.0), (1.0, 1.0), (2.5, 0.0)
  wrapper<cudf::size_type> linestring1_offsets{0};
  wrapper<T> linestring1_points_x{0.0, 0.0, 2.0};
  wrapper<T> linestring1_points_y{0.0, 2.0, 2.0};
  wrapper<cudf::size_type> linestring2_offsets{0};
  wrapper<T> linestring2_points_x{2.0, 2.0, 1.0, 2.5};
  wrapper<T> linestring2_points_y{2.0, 1.0, 1.0, 0.0};

  wrapper<T> expected{0.0};

  auto got = pairwise_linestring_distance(column_view(linestring1_offsets),
                                          linestring1_points_x,
                                          linestring1_points_y,
                                          column_view(linestring2_offsets),
                                          linestring2_points_x,
                                          linestring2_points_y);
  expect_columns_equivalent(expected, *got, verbosity);
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairLinestringCoincide)
{
  using T = TypeParam;
  // Linestring 1: (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)
  // Linestring 2: (2.0, 1.0), (1.0, 1.0), (1.0, 0.0), (2.0, 0.0), (2.0, 0.5)
  wrapper<cudf::size_type> linestring1_offsets{0};
  wrapper<T> linestring1_points_x{0.0, 1.0, 1.0, 0.0};
  wrapper<T> linestring1_points_y{0.0, 0.0, 1.0, 1.0};
  wrapper<cudf::size_type> linestring2_offsets{0};
  wrapper<T> linestring2_points_x{2.0, 1.0, 1.0, 2.0, 2.0};
  wrapper<T> linestring2_points_y{1.0, 1.0, 0.0, 0.0, 0.5};

  wrapper<T> expected{0.0};

  auto got = pairwise_linestring_distance(column_view(linestring1_offsets),
                                          linestring1_points_x,
                                          linestring1_points_y,
                                          column_view(linestring2_offsets),
                                          linestring2_points_x,
                                          linestring2_points_y);
  expect_columns_equivalent(expected, *got, verbosity);
}

}  // namespace test
}  // namespace cuspatial
