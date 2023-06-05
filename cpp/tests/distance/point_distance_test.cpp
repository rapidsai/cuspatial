/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cuspatial_test/geometry_fixtures.hpp>

#include <cuspatial/distance.hpp>
#include <cuspatial/error.hpp>

#include <optional>

using namespace cuspatial;
using namespace cuspatial::test;

using namespace cudf;
using namespace cudf::test;

template <typename T>
struct PairwisePointDistanceTest : public EmptyGeometryColumnFixture<T> {};

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_CASE(PairwisePointDistanceTest, TestTypes);

TYPED_TEST(PairwisePointDistanceTest, SingleToSingleEmpty)
{
  auto got    = pairwise_point_distance(this->empty_point(), this->empty_point());
  auto expect = fixed_width_column_wrapper<TypeParam>{};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect, *got);
}

TYPED_TEST(PairwisePointDistanceTest, SingleToMultiEmpty)
{
  auto got    = pairwise_point_distance(this->empty_point(), this->empty_multipoint());
  auto expect = fixed_width_column_wrapper<TypeParam>{};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect, *got);
}

TYPED_TEST(PairwisePointDistanceTest, MultiToSingleEmpty)
{
  auto got    = pairwise_point_distance(this->empty_point(), this->empty_multipoint());
  auto expect = fixed_width_column_wrapper<TypeParam>{};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect, *got);
}

TYPED_TEST(PairwisePointDistanceTest, MultiToMultiEmpty)
{
  auto got    = pairwise_point_distance(this->empty_multipoint(), this->empty_multipoint());
  auto expect = fixed_width_column_wrapper<TypeParam>{};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect, *got);
}

struct PairwisePointDistanceFailOnSizeTest : public EmptyAndOneGeometryColumnFixture {};

TEST_F(PairwisePointDistanceFailOnSizeTest, SizeMismatch)
{
  EXPECT_THROW(pairwise_point_distance(this->empty_point(), this->one_point()),
               cuspatial::logic_error);
}

TEST_F(PairwisePointDistanceFailOnSizeTest, SizeMismatch2)
{
  EXPECT_THROW(pairwise_point_distance(this->one_point(), this->empty_multipoint()),
               cuspatial::logic_error);
}

struct PairwisePointDistanceFailOnTypeTest : public EmptyGeometryColumnFixtureMultipleTypes {};

TEST_F(PairwisePointDistanceFailOnTypeTest, CoordinateTypeMismatch)
{
  EXPECT_THROW(pairwise_point_distance(EmptyGeometryColumnBase<float>::empty_point(),
                                       EmptyGeometryColumnBase<double>::empty_point()),
               cuspatial::logic_error);
}

TEST_F(PairwisePointDistanceFailOnTypeTest, GeometryTypeMismatch)
{
  EXPECT_THROW(pairwise_point_distance(EmptyGeometryColumnBase<float>::empty_point(),
                                       EmptyGeometryColumnBase<float>::empty_polygon()),
               cuspatial::logic_error);
}
