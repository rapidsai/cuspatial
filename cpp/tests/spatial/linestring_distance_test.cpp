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

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairRandom)
{
  using T = TypeParam;
  wrapper<cudf::size_type> linestring1_offsets{0};
  wrapper<T> linestring1_points_x{-22556.235212018168, -16375.655690574613, -20082.724633593425};
  wrapper<T> linestring1_points_y{41094.0501840996, 42992.319790050366, 33759.13529113619};
  wrapper<cudf::size_type> linestring2_offsets{0};
  wrapper<T> linestring2_points_x{4365.496374409238, 1671.0269165650761};
  wrapper<T> linestring2_points_y{-59857.47177852941, -54931.9723439855};

  wrapper<T> expected{91319.97744223749};

  auto got = pairwise_linestring_distance(column_view(linestring1_offsets),
                                          linestring1_points_x,
                                          linestring1_points_y,
                                          column_view(linestring2_offsets),
                                          linestring2_points_x,
                                          linestring2_points_y);
  expect_columns_equivalent(expected, *got, verbosity);
}

TYPED_TEST(PairwiseLinestringDistanceTest, TwoPairs)
{
  using T = TypeParam;
  wrapper<cudf::size_type> linestring1_offsets{0, 4};
  wrapper<T> linestring1_points_x{
    41658.902315589876,
    46600.70359801489,
    47079.510547637154,
    51498.48049880379,
    -27429.917796286478,
    -21764.269974046114,
    -14460.71813363161,
    -18226.13032712476,
  };
  wrapper<T> linestring1_points_y{14694.11814724456,
                                  8771.431887804214,
                                  10199.68027155776,
                                  17049.62665643919,
                                  -33240.8339287343,
                                  -37974.45515744517,
                                  -31333.481529957502,
                                  -30181.03842467982};
  wrapper<cudf::size_type> linestring2_offsets{0, 2};
  wrapper<T> linestring2_points_x{
    24046.170375947084,
    20614.007047185743,
    48381.39607717942,
    53346.77764665915,
  };
  wrapper<T> linestring2_points_y{
    27878.56737867571,
    26489.74880629428,
    -8366.313156569413,
    -2066.3869793077383,
  };

  wrapper<T> expected{22000.86425379464, 66907.56415814416};

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
