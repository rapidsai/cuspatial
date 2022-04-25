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

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairGeolife)
{
  // Example extracted from a pair of trajectry in geolife dataset
  using T = TypeParam;
  wrapper<cudf::size_type> linestring1_offsets{0};
  wrapper<T> linestring1_points_x{39.97551667, 39.97585, 39.97598333, 39.9761, 39.97623333};
  wrapper<T> linestring1_points_y{116.33028333, 116.3304, 116.33046667, 116.3305, 116.33056667};
  wrapper<cudf::size_type> linestring2_offsets{0};
  wrapper<T> linestring2_points_x{
    39.97381667, 39.97341667, 39.9731,     39.97293333, 39.97233333, 39.97218333, 39.97218333,
    39.97215,    39.97168333, 39.97093333, 39.97073333, 39.9705,     39.96991667, 39.96961667,
    39.96918333, 39.96891667, 39.97531667, 39.97533333, 39.97535,    39.97515,    39.97506667,
    39.97508333, 39.9751,     39.97513333, 39.97511667, 39.97503333, 39.97513333, 39.97523333,
    39.97521667, 39.97503333, 39.97463333, 39.97443333, 39.96838333, 39.96808333, 39.96771667,
    39.96745,    39.96735,    39.9673,     39.96718333, 39.96751667, 39.9678,     39.9676,
    39.96741667, 39.9672,     39.97646667, 39.9764,     39.97625,    39.9762,     39.97603333,
    39.97581667, 39.9757,     39.97551667, 39.97535,    39.97543333, 39.97538333};
  wrapper<T> linestring2_points_y{
    116.34211667, 116.34215,    116.34218333, 116.34221667, 116.34225,    116.34243333,
    116.34296667, 116.34478333, 116.34486667, 116.34485,    116.34468333, 116.34461667,
    116.34465,    116.34465,    116.34466667, 116.34465,    116.33036667, 116.32961667,
    116.3292,     116.32903333, 116.32985,    116.33128333, 116.33195,    116.33618333,
    116.33668333, 116.33818333, 116.34,       116.34045,    116.34183333, 116.342,
    116.34203333, 116.3422,     116.3445,     116.34451667, 116.3445,     116.34453333,
    116.34493333, 116.34506667, 116.3451,     116.34483333, 116.3448,     116.3449,
    116.345,      116.34506667, 116.33006667, 116.33015,    116.33026667, 116.33038333,
    116.33036667, 116.3303,     116.33033333, 116.33035,    116.3304,     116.33078333,
    116.33066667};

  wrapper<T> expected{0.0};

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
