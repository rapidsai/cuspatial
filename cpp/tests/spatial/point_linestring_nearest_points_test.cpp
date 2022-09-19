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

#include <cuspatial/error.hpp>
#include <cuspatial/point_linestring_nearest_points.hpp>
#include <cuspatial/vec_2d.hpp>

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace cuspatial {

using namespace cudf;
using namespace cudf::test;

template <typename T>
struct PairwisePointLinestringNearestPointsTest : public ::testing::Test {
};

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_CASE(PairwisePointLinestringNearestPointsTest, TestTypes);

TYPED_TEST(PairwisePointLinestringNearestPointsTest, Empty)
{
  using T = TypeParam;

  auto xy      = fixed_width_column_wrapper<T>{};
  auto offset  = fixed_width_column_wrapper<size_type>{0};
  auto line_xy = fixed_width_column_wrapper<T>{};

  auto [point_idx, linestring_idx, segment_idx, nearest_points] =
    pairwise_point_linestring_nearest_points(
      std::nullopt, xy, std::nullopt, column_view(offset), line_xy);

  auto expect_segment_idx    = fixed_width_column_wrapper<size_type>{};
  auto expect_nearest_points = fixed_width_column_wrapper<T>{};

  EXPECT_EQ(point_idx, std::nullopt);
  EXPECT_EQ(linestring_idx, std::nullopt);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect_segment_idx, *segment_idx);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect_nearest_points, *nearest_points);
}

TYPED_TEST(PairwisePointLinestringNearestPointsTest, SinglePointMultiLineString)
{
  using T = TypeParam;

  auto xy            = fixed_width_column_wrapper<T>{0.0, 0.5};
  auto line_geometry = fixed_width_column_wrapper<size_type>{0, 2};
  auto line_offset   = fixed_width_column_wrapper<size_type>{0, 3, 5};
  auto line_xy = fixed_width_column_wrapper<T>{1.0, 1.0, 2.0, 2.0, 2.5, 1.3, -1.0, -3.6, -0.8, 1.0};

  auto [point_idx, linestring_idx, segment_idx, nearest_points] =
    pairwise_point_linestring_nearest_points(
      std::nullopt, xy, column_view(line_geometry), column_view(line_offset), line_xy);

  auto expect_linestring_idx = fixed_width_column_wrapper<size_type>{1};
  auto expect_segment_idx    = fixed_width_column_wrapper<size_type>{0};
  auto expect_nearest_points =
    fixed_width_column_wrapper<T>{-0.820188679245283, 0.5356603773584898};

  EXPECT_EQ(point_idx, std::nullopt);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect_linestring_idx, *(linestring_idx.value()));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect_segment_idx, *segment_idx);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect_nearest_points, *nearest_points);
}

TYPED_TEST(PairwisePointLinestringNearestPointsTest, MultiPointSingleLineString)
{
  using T = TypeParam;

  auto point_geometry = fixed_width_column_wrapper<size_type>{0, 2};
  auto xy             = fixed_width_column_wrapper<T>{0.5, 0.5, 0.5, 0.0};
  auto line_offset    = fixed_width_column_wrapper<size_type>{0, 4};
  auto line_xy        = fixed_width_column_wrapper<T>{0.0, 2.0, 2.0, 0.0, 0.0, -2.0, -2.0, 0.0};

  auto [point_idx, linestring_idx, segment_idx, nearest_points] =
    pairwise_point_linestring_nearest_points(
      column_view(point_geometry), xy, std::nullopt, column_view(line_offset), line_xy);

  auto expect_point_idx      = fixed_width_column_wrapper<size_type>{0};
  auto expect_segment_idx    = fixed_width_column_wrapper<size_type>{0};
  auto expect_nearest_points = fixed_width_column_wrapper<T>{1.0, 1.0};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect_point_idx, *(point_idx.value()));
  EXPECT_EQ(linestring_idx, std::nullopt);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect_segment_idx, *segment_idx);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect_nearest_points, *nearest_points);
}

TYPED_TEST(PairwisePointLinestringNearestPointsTest, MultiPointMultiLineString)
{
  using T = TypeParam;

  auto point_geometry = fixed_width_column_wrapper<size_type>{0, 2};
  auto xy             = fixed_width_column_wrapper<T>{-0.5, 0.0, 0.0, 1.0};
  auto line_geometry  = fixed_width_column_wrapper<size_type>{0, 2};
  auto line_offset    = fixed_width_column_wrapper<size_type>{0, 3, 6};
  auto line_xy =
    fixed_width_column_wrapper<T>{2.0, 2.0, 0.0, 0.0, 2.0, -2.0, -2.0, 2.0, 0.0, 0.0, -2.0, -2.0};

  auto [point_idx, linestring_idx, segment_idx, nearest_points] =
    pairwise_point_linestring_nearest_points(column_view(point_geometry),
                                             xy,
                                             column_view(line_geometry),
                                             column_view(line_offset),
                                             line_xy);

  auto expect_point_idx      = fixed_width_column_wrapper<size_type>{0};
  auto expect_linestring_idx = fixed_width_column_wrapper<size_type>{1};
  auto expect_segment_idx    = fixed_width_column_wrapper<size_type>{0};
  auto expect_nearest_points = fixed_width_column_wrapper<T>{-0.25, 0.25};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect_point_idx, *(point_idx.value()));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect_linestring_idx, *(linestring_idx.value()));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect_segment_idx, *segment_idx);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect_nearest_points, *nearest_points);
}

struct PairwisePointLinestringNearestPointsThrowTest : public ::testing::Test {
};

TEST_F(PairwisePointLinestringNearestPointsThrowTest, OddNumberOfCoordinates)
{
  auto xy      = fixed_width_column_wrapper<float>{1, 1, 2};
  auto offset  = fixed_width_column_wrapper<size_type>{0, 3};
  auto line_xy = fixed_width_column_wrapper<float>{1, 1, 2, 2, 3, 3};

  EXPECT_THROW(pairwise_point_linestring_nearest_points(
                 std::nullopt, xy, std::nullopt, column_view(offset), line_xy),
               cuspatial::logic_error);
}

TEST_F(PairwisePointLinestringNearestPointsThrowTest, NumPairsMismatchSinglePointSingleLinestring)
{
  auto xy      = fixed_width_column_wrapper<float>{1, 1, 2, 2};
  auto offset  = fixed_width_column_wrapper<size_type>{0, 3};
  auto line_xy = fixed_width_column_wrapper<float>{1, 1, 2, 2, 3, 3};

  EXPECT_THROW(pairwise_point_linestring_nearest_points(
                 std::nullopt, xy, std::nullopt, column_view(offset), line_xy),
               cuspatial::logic_error);
}

TEST_F(PairwisePointLinestringNearestPointsThrowTest, NumPairsMismatchSinglePointMultiLinestring)
{
  auto xy = fixed_width_column_wrapper<float>{1, 1, 2, 2};

  auto line_geometry = fixed_width_column_wrapper<size_type>{0, 2};
  auto offset        = fixed_width_column_wrapper<size_type>{0, 3, 5};
  auto line_xy       = fixed_width_column_wrapper<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5};

  EXPECT_THROW(pairwise_point_linestring_nearest_points(
                 std::nullopt, xy, column_view(line_geometry), column_view(offset), line_xy),
               cuspatial::logic_error);
}

TEST_F(PairwisePointLinestringNearestPointsThrowTest, NumPairsMismatchMultiPointSingleLinestring)
{
  auto point_geometry = fixed_width_column_wrapper<size_type>{0, 2, 4};
  auto xy             = fixed_width_column_wrapper<float>{1, 1, 2, 2, 3, 3, 4, 4};

  auto offset  = fixed_width_column_wrapper<size_type>{0, 3};
  auto line_xy = fixed_width_column_wrapper<float>{1, 1, 2, 2, 3, 3};

  EXPECT_THROW(pairwise_point_linestring_nearest_points(
                 column_view(point_geometry), xy, std::nullopt, column_view(offset), line_xy),
               cuspatial::logic_error);
}

TEST_F(PairwisePointLinestringNearestPointsThrowTest, NumPairsMismatchMultiPointMultiLinestring)
{
  auto point_geometry = fixed_width_column_wrapper<size_type>{0, 2, 4};
  auto xy             = fixed_width_column_wrapper<float>{1, 1, 2, 2, 3, 3, 4, 4};

  auto linestring_geometry = fixed_width_column_wrapper<size_type>{0, 2, 4, 6};
  auto offset              = fixed_width_column_wrapper<size_type>{0, 2, 4, 6, 8, 10};
  auto line_xy =
    fixed_width_column_wrapper<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10};

  EXPECT_THROW(pairwise_point_linestring_nearest_points(
                 column_view(point_geometry), xy, std::nullopt, column_view(offset), line_xy),
               cuspatial::logic_error);
}

TEST_F(PairwisePointLinestringNearestPointsThrowTest, MismatchType)
{
  auto point_geometry = fixed_width_column_wrapper<size_type>{0, 2, 4};
  auto xy             = fixed_width_column_wrapper<float>{1, 1, 2, 2, 3, 3, 4, 4};

  auto linestring_geometry = fixed_width_column_wrapper<size_type>{0, 2, 4};
  auto offset              = fixed_width_column_wrapper<size_type>{0, 2, 4, 6};
  auto line_xy             = fixed_width_column_wrapper<double>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6};

  EXPECT_THROW(pairwise_point_linestring_nearest_points(
                 column_view(point_geometry), xy, std::nullopt, column_view(offset), line_xy),
               cuspatial::logic_error);
}

TEST_F(PairwisePointLinestringNearestPointsThrowTest, ContainsNull)
{
  auto point_geometry = fixed_width_column_wrapper<size_type>{0, 2, 4};
  auto xy = fixed_width_column_wrapper<float>{{1, 1, 2, 2, 3, 3, 4, 4}, {1, 0, 1, 1, 1, 1, 1, 1}};

  auto linestring_geometry = fixed_width_column_wrapper<size_type>{0, 2, 4};
  auto offset              = fixed_width_column_wrapper<size_type>{0, 2, 4, 6};
  auto line_xy             = fixed_width_column_wrapper<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6};

  EXPECT_THROW(pairwise_point_linestring_nearest_points(
                 column_view(point_geometry), xy, std::nullopt, column_view(offset), line_xy),
               cuspatial::logic_error);
}

}  // namespace cuspatial
