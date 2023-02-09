/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cuspatial/error.hpp>
#include <cuspatial/pairwise_point_in_polygon.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <type_traits>

using namespace cudf::test;

template <typename T, typename R = T>
using wrapper = fixed_width_column_wrapper<T, R>;

template <typename T>
struct PairwisePointInPolygonTest : public BaseFixture {
};

// float and double are logically the same but would require separate tests due to precision.
using TestTypes = FloatingPointTypes;
TYPED_TEST_CASE(PairwisePointInPolygonTest, TestTypes);

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};

TYPED_TEST(PairwisePointInPolygonTest, Empty)
{
  using T = TypeParam;

  auto test_point_xs     = wrapper<T>({});
  auto test_point_ys     = wrapper<T>({});
  auto poly_offsets      = wrapper<cudf::size_type>({});
  auto poly_ring_offsets = wrapper<cudf::size_type>({});
  auto poly_point_xs     = wrapper<T>({});
  auto poly_point_ys     = wrapper<T>({});

  auto expected = wrapper<int32_t>({});

  auto actual = cuspatial::pairwise_point_in_polygon(
    test_point_xs, test_point_ys, poly_offsets, poly_ring_offsets, poly_point_xs, poly_point_ys);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, actual->view(), verbosity);
}

template <typename T>
struct PairwisePointInPolygonUnsupportedTypesTest : public BaseFixture {
};

using UnsupportedTestTypes = RemoveIf<ContainedIn<TestTypes>, NumericTypes>;
TYPED_TEST_CASE(PairwisePointInPolygonUnsupportedTypesTest, UnsupportedTestTypes);

TYPED_TEST(PairwisePointInPolygonUnsupportedTypesTest, UnsupportedPointType)
{
  using T = TypeParam;

  auto test_point_xs     = wrapper<T>({0.0});
  auto test_point_ys     = wrapper<T>({0.0});
  auto poly_offsets      = wrapper<cudf::size_type>({0});
  auto poly_ring_offsets = wrapper<cudf::size_type>({0});
  auto poly_point_xs     = wrapper<T>({0.0, 1.0, 0.0, -1.0});
  auto poly_point_ys     = wrapper<T>({1.0, 0.0, -1.0, 0.0});

  EXPECT_THROW(
    cuspatial::pairwise_point_in_polygon(
      test_point_xs, test_point_ys, poly_offsets, poly_ring_offsets, poly_point_xs, poly_point_ys),
    cuspatial::logic_error);
}

template <typename T>
struct PairwisePointInPolygonUnsupportedChronoTypesTest : public BaseFixture {
};

TYPED_TEST_CASE(PairwisePointInPolygonUnsupportedChronoTypesTest, ChronoTypes);

TYPED_TEST(PairwisePointInPolygonUnsupportedChronoTypesTest, UnsupportedPointChronoType)
{
  using T = TypeParam;
  using R = typename T::rep;

  auto test_point_xs     = wrapper<T, R>({R{0}, R{0}});
  auto test_point_ys     = wrapper<T, R>({R{0}});
  auto poly_offsets      = wrapper<cudf::size_type>({0});
  auto poly_ring_offsets = wrapper<cudf::size_type>({0});
  auto poly_point_xs     = wrapper<T, R>({R{0}, R{1}, R{0}, R{-1}});
  auto poly_point_ys     = wrapper<T, R>({R{1}, R{0}, R{-1}, R{0}});

  EXPECT_THROW(
    cuspatial::pairwise_point_in_polygon(
      test_point_xs, test_point_ys, poly_offsets, poly_ring_offsets, poly_point_xs, poly_point_ys),
    cuspatial::logic_error);
}

struct PairwisePointInPolygonErrorTest : public BaseFixture {
};

TEST_F(PairwisePointInPolygonErrorTest, MismatchTestPointXYLength)
{
  using T = double;

  auto test_point_xs     = wrapper<T>({0.0, 0.0});
  auto test_point_ys     = wrapper<T>({0.0});
  auto poly_offsets      = wrapper<cudf::size_type>({0});
  auto poly_ring_offsets = wrapper<cudf::size_type>({0});
  auto poly_point_xs     = wrapper<T>({0.0, 1.0, 0.0, -1.0});
  auto poly_point_ys     = wrapper<T>({1.0, 0.0, -1.0, 0.0});

  EXPECT_THROW(
    cuspatial::pairwise_point_in_polygon(
      test_point_xs, test_point_ys, poly_offsets, poly_ring_offsets, poly_point_xs, poly_point_ys),
    cuspatial::logic_error);
}

TEST_F(PairwisePointInPolygonErrorTest, MismatchTestPointType)
{
  using T = double;

  auto test_point_xs     = wrapper<T>({0.0});
  auto test_point_ys     = wrapper<float>({0.0});
  auto poly_offsets      = wrapper<cudf::size_type>({0});
  auto poly_ring_offsets = wrapper<cudf::size_type>({0});
  auto poly_point_xs     = wrapper<T>({0.0, 1.0, 0.0});
  auto poly_point_ys     = wrapper<T>({1.0, 0.0, -1.0, 0.0});

  EXPECT_THROW(
    cuspatial::pairwise_point_in_polygon(
      test_point_xs, test_point_ys, poly_offsets, poly_ring_offsets, poly_point_xs, poly_point_ys),
    cuspatial::logic_error);
}

TEST_F(PairwisePointInPolygonErrorTest, MismatchPolyPointXYLength)
{
  using T = double;

  auto test_point_xs     = wrapper<T>({0.0});
  auto test_point_ys     = wrapper<T>({0.0});
  auto poly_offsets      = wrapper<cudf::size_type>({0});
  auto poly_ring_offsets = wrapper<cudf::size_type>({0});
  auto poly_point_xs     = wrapper<T>({0.0, 1.0, 0.0});
  auto poly_point_ys     = wrapper<T>({1.0, 0.0, -1.0, 0.0});

  EXPECT_THROW(
    cuspatial::pairwise_point_in_polygon(
      test_point_xs, test_point_ys, poly_offsets, poly_ring_offsets, poly_point_xs, poly_point_ys),
    cuspatial::logic_error);
}

TEST_F(PairwisePointInPolygonErrorTest, MismatchPolyPointType)
{
  using T = double;

  auto test_point_xs     = wrapper<T>({0.0});
  auto test_point_ys     = wrapper<T>({0.0});
  auto poly_offsets      = wrapper<cudf::size_type>({0});
  auto poly_ring_offsets = wrapper<cudf::size_type>({0});
  auto poly_point_xs     = wrapper<T>({0.0, 1.0, 0.0});
  auto poly_point_ys     = wrapper<float>({1.0, 0.0, -1.0, 0.0});

  EXPECT_THROW(
    cuspatial::pairwise_point_in_polygon(
      test_point_xs, test_point_ys, poly_offsets, poly_ring_offsets, poly_point_xs, poly_point_ys),
    cuspatial::logic_error);
}

TEST_F(PairwisePointInPolygonErrorTest, MismatchPointTypes)
{
  auto test_point_xs     = wrapper<float>({0.0});
  auto test_point_ys     = wrapper<float>({0.0});
  auto poly_offsets      = wrapper<cudf::size_type>({0});
  auto poly_ring_offsets = wrapper<cudf::size_type>({0});
  auto poly_point_xs     = wrapper<double>({0.0, 1.0, 0.0, -1.0});
  auto poly_point_ys     = wrapper<double>({1.0, 0.0, -1.0, 0.0});

  EXPECT_THROW(
    cuspatial::pairwise_point_in_polygon(
      test_point_xs, test_point_ys, poly_offsets, poly_ring_offsets, poly_point_xs, poly_point_ys),
    cuspatial::logic_error);
}

TEST_F(PairwisePointInPolygonErrorTest, MorePointsThanPolygons)
{
  auto test_point_xs     = wrapper<float>({0.0, 0.0});
  auto test_point_ys     = wrapper<float>({0.0, 0.0});
  auto poly_offsets      = wrapper<cudf::size_type>({0});
  auto poly_ring_offsets = wrapper<cudf::size_type>({0});
  auto poly_point_xs     = wrapper<float>({0.0, 1.0, 0.0, -1.0});
  auto poly_point_ys     = wrapper<float>({1.0, 0.0, -1.0, 0.0});

  EXPECT_THROW(
    cuspatial::pairwise_point_in_polygon(
      test_point_xs, test_point_ys, poly_offsets, poly_ring_offsets, poly_point_xs, poly_point_ys),
    cuspatial::logic_error);
}

TEST_F(PairwisePointInPolygonErrorTest, MorePolygonsThanPoints)
{
  auto test_point_xs     = wrapper<float>({0.0});
  auto test_point_ys     = wrapper<float>({0.0});
  auto poly_offsets      = wrapper<cudf::size_type>({0, 4});
  auto poly_ring_offsets = wrapper<cudf::size_type>({0, 4});
  auto poly_point_xs     = wrapper<float>({0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0});
  auto poly_point_ys     = wrapper<float>({1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0});

  EXPECT_THROW(
    cuspatial::pairwise_point_in_polygon(
      test_point_xs, test_point_ys, poly_offsets, poly_ring_offsets, poly_point_xs, poly_point_ys),
    cuspatial::logic_error);
}
