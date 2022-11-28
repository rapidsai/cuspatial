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

#include <cudf/types.hpp>
#include <cuspatial/distance/linestring_distance.hpp>
#include <cuspatial/error.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <gtest/gtest.h>
#include <optional>

using namespace cuspatial;
using namespace cudf;
using namespace cudf::test;

template <typename T>
using wrapper = fixed_width_column_wrapper<T>;

template <typename T>
struct PairwiseLinestringDistanceTest : public BaseFixture {
};

struct PairwiseLinestringDistanceTestUntyped : public BaseFixture {
};

// float and double are logically the same but would require separate tests due to precision.
using TestTypes = FloatingPointTypes;
TYPED_TEST_CASE(PairwiseLinestringDistanceTest, TestTypes);

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};

TYPED_TEST(PairwiseLinestringDistanceTest, EmptyInput)
{
  using T = TypeParam;
  wrapper<cudf::size_type> l1offsets{0};
  wrapper<T> xy1{};
  wrapper<cudf::size_type> l2offsets{0};
  wrapper<T> xy2{};

  wrapper<T> expected{};

  auto result = cuspatial::pairwise_linestring_distance(
    std::nullopt, column_view(l1offsets), xy1, std::nullopt, column_view(l2offsets), xy2);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view(), verbosity);
}

TYPED_TEST(PairwiseLinestringDistanceTest, FourPairSingleToMultiLineString)
{
  using T = TypeParam;

  wrapper<cudf::size_type> l1part_offset{0, 3, 5, 8, 10};
  wrapper<T> l1_xy{0, 1, 1, 0, -1, 0, 0, 0, 0, 1, 0, 0, 2, 2, -2, 0, 2, 2, -2, -2};
  wrapper<cudf::size_type> l2geom_offset{0, 1, 3, 5, 7};
  wrapper<cudf::size_type> l2part_offset{0, 4, 7, 10, 12, 14, 17, 20};
  wrapper<T> l2_xy{1, 1, 2, 1, 2, 0, 3,  0, 1, 0, 1, 1, 1,  2, 1,  -1, 1,  -2, 1,   -3,
                   2, 0, 0, 2, 0, 2, -2, 0, 1, 1, 5, 5, 10, 0, -1, -1, -5, -5, -10, 0};

  wrapper<T> expected{std::sqrt(2.0) / 2, 1.0, 0.0, 0.0};

  auto result = cuspatial::pairwise_linestring_distance(std::nullopt,
                                                        column_view(l1part_offset),
                                                        l1_xy,
                                                        column_view(l2geom_offset),
                                                        column_view(l2part_offset),
                                                        l2_xy);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view(), verbosity);
}

TYPED_TEST(PairwiseLinestringDistanceTest, FourPairSingleToSingleLineString)
{
  using T = TypeParam;

  wrapper<cudf::size_type> l1part_offset{0, 3, 5, 8, 10};
  wrapper<T> l1_xy{0, 1, 1, 0, -1, 0, 0, 0, 0, 1, 0, 0, 2, 2, -2, 0, 2, 2, -2, -2};
  wrapper<cudf::size_type> l2part_offset{0, 4, 7, 9, 11};
  wrapper<T> l2_xy{1, 1, 2, 1, 2, 0, 3, 0, 1, 0, 1, 1, 1, 2, 2, 0, 0, 2, 1, 1, 5, 5, 10, 0};

  wrapper<T> expected{std::sqrt(2.0) / 2, 1.0, 0.0, 0.0};

  auto result = cuspatial::pairwise_linestring_distance(std::nullopt,
                                                        column_view(l1part_offset),
                                                        l1_xy,
                                                        std::nullopt,
                                                        column_view(l2part_offset),
                                                        l2_xy);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view(), verbosity);
}

TYPED_TEST(PairwiseLinestringDistanceTest, TwoPairMultiToSingleLineString)
{
  using T = TypeParam;

  wrapper<cudf::size_type> l1geom_offset{0, 1, 3};
  wrapper<cudf::size_type> l1part_offset{0, 3, 6, 8};
  wrapper<T> l1_xy{0, 0, 0, 1, 0, 2, 0, 1, 1, 1, 2, 1, 2, 1, 2, 0};
  wrapper<cudf::size_type> l2part_offset{0, 2, 4};
  wrapper<T> l2_xy{1, 0, 1, 1, 0, 0, 1, 0};

  wrapper<T> expected{1.0, 1.0};

  auto result = cuspatial::pairwise_linestring_distance(column_view(l1geom_offset),
                                                        column_view(l1part_offset),
                                                        l1_xy,
                                                        std::nullopt,
                                                        column_view(l2part_offset),
                                                        l2_xy);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view(), verbosity);
}

TYPED_TEST(PairwiseLinestringDistanceTest, OnePairMultiToMultiLineString)
{
  using T = TypeParam;

  wrapper<cudf::size_type> l1geom_offset{0, 3};
  wrapper<cudf::size_type> l1part_offset{0, 3, 6, 8};
  wrapper<T> l1_xy{0, 0, 0, 1, 0, 2, 0, 1, 1, 1, 2, 1, 2, 1, 2, 0};
  wrapper<cudf::size_type> l2geom_offset{0, 3};
  wrapper<cudf::size_type> l2part_offset{0, 2, 4, 6};
  wrapper<T> l2_xy{0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3};

  wrapper<T> expected{0.0};

  auto result = cuspatial::pairwise_linestring_distance(column_view(l1geom_offset),
                                                        column_view(l1part_offset),
                                                        l1_xy,
                                                        column_view(l2geom_offset),
                                                        column_view(l2part_offset),
                                                        l2_xy);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view(), verbosity);
}

TEST_F(PairwiseLinestringDistanceTestUntyped, InputSizeMismatchSingletoSingle)
{
  wrapper<cudf::size_type> l1part_offset{0, 2};
  wrapper<float> l1_xy{0, 0, 1, 1};
  wrapper<cudf::size_type> l2part_offset{0, 2, 4};
  wrapper<float> l2_xy{0, 0, 1, 1, 2, 2, 3, 3};

  EXPECT_THROW(cuspatial::pairwise_linestring_distance(std::nullopt,
                                                       column_view(l1part_offset),
                                                       l1_xy,
                                                       std::nullopt,
                                                       column_view(l2part_offset),
                                                       l2_xy),
               cuspatial::logic_error);
}

TEST_F(PairwiseLinestringDistanceTestUntyped, InputSizeMismatchSingletoMulti)
{
  wrapper<cudf::size_type> l1part_offset{0, 2};
  wrapper<float> l1_xy{0, 0, 1, 1};
  wrapper<cudf::size_type> l2geom_offset{0, 1, 3};
  wrapper<cudf::size_type> l2part_offset{0, 2, 4, 6};
  wrapper<float> l2_xy{0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5};

  EXPECT_THROW(cuspatial::pairwise_linestring_distance(std::nullopt,
                                                       column_view(l1part_offset),
                                                       l1_xy,
                                                       column_view(l2geom_offset),
                                                       column_view(l2part_offset),
                                                       l2_xy),
               cuspatial::logic_error);
}

TEST_F(PairwiseLinestringDistanceTestUntyped, InputSizeMismatchMultitoSingle)
{
  wrapper<cudf::size_type> l1geom_offset{0, 1, 3};
  wrapper<cudf::size_type> l1part_offset{0, 2, 4, 6};
  wrapper<float> l1_xy{0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
  wrapper<cudf::size_type> l2part_offset{0, 2};
  wrapper<float> l2_xy{0, 0, 1, 1};

  EXPECT_THROW(cuspatial::pairwise_linestring_distance(column_view(l1geom_offset),
                                                       column_view(l1part_offset),
                                                       l1_xy,
                                                       std::nullopt,
                                                       column_view(l2part_offset),
                                                       l2_xy),
               cuspatial::logic_error);
}

TEST_F(PairwiseLinestringDistanceTestUntyped, InputSizeMismatchMultitoMulti)
{
  wrapper<cudf::size_type> l1geom_offset{0, 1, 3};
  wrapper<cudf::size_type> l1part_offset{0, 2, 4, 6};
  wrapper<float> l1_xy{0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
  wrapper<cudf::size_type> l2geom_offset{0, 2};
  wrapper<cudf::size_type> l2part_offset{0, 2, 4};
  wrapper<float> l2_xy{0, 0, 1, 1, 2, 2, 3, 3};

  EXPECT_THROW(cuspatial::pairwise_linestring_distance(column_view(l1geom_offset),
                                                       column_view(l1part_offset),
                                                       l1_xy,
                                                       column_view(l2geom_offset),
                                                       column_view(l2part_offset),
                                                       l2_xy),
               cuspatial::logic_error);
}

TEST_F(PairwiseLinestringDistanceTestUntyped, CoordinatesNotEven)
{
  wrapper<cudf::size_type> l1_part_offset{0, 2};
  wrapper<float> l1_xy{0, 0, 1, 1, 2, 2, 3};
  wrapper<cudf::size_type> l2_part_offset{0, 2};
  wrapper<float> l2_xy{0, 0, 1, 1, 2, 2, 3, 3};

  EXPECT_THROW(cuspatial::pairwise_linestring_distance(std::nullopt,
                                                       column_view(l1_part_offset),
                                                       l1_xy,
                                                       std::nullopt,
                                                       column_view(l2_part_offset),
                                                       l2_xy),
               cuspatial::logic_error);
}

TEST_F(PairwiseLinestringDistanceTestUntyped, TypeMismatch)
{
  wrapper<cudf::size_type> l1_part_offset{0, 2};
  wrapper<float> l1_xy{0, 0, 1, 1};
  wrapper<cudf::size_type> l2_part_offset{0, 2};
  wrapper<double> l2_xy{0, 0, 1, 1};

  EXPECT_THROW(cuspatial::pairwise_linestring_distance(std::nullopt,
                                                       column_view(l1_part_offset),
                                                       l1_xy,
                                                       std::nullopt,
                                                       column_view(l2_part_offset),
                                                       l2_xy),
               cuspatial::logic_error);
}

TEST_F(PairwiseLinestringDistanceTestUntyped, ContainsNull)
{
  wrapper<cudf::size_type> l1_part_offset{0, 2};
  wrapper<float> l1_xy{{0, 0, 1, 1}, {1, 0, 1, 1}};
  wrapper<cudf::size_type> l2_part_offset{0, 2};
  wrapper<float> l2_xy{0, 0, 1, 1};

  EXPECT_THROW(cuspatial::pairwise_linestring_distance(std::nullopt,
                                                       column_view(l1_part_offset),
                                                       l1_xy,
                                                       std::nullopt,
                                                       column_view(l2_part_offset),
                                                       l2_xy),
               cuspatial::logic_error);
}
