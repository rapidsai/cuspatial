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
struct PairwiseLineStringDistanceTest : public EmptyGeometryColumnFixture<T> {};

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_CASE(PairwiseLineStringDistanceTest, TestTypes);

TYPED_TEST(PairwiseLineStringDistanceTest, SingleToSingleEmpty)
{
  auto got    = pairwise_linestring_distance(this->empty_linestring(), this->empty_linestring());
  auto expect = fixed_width_column_wrapper<TypeParam>{};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect, *got);
}

TYPED_TEST(PairwiseLineStringDistanceTest, SingleToMultiEmpty)
{
  auto got = pairwise_linestring_distance(this->empty_linestring(), this->empty_multilinestring());
  auto expect = fixed_width_column_wrapper<TypeParam>{};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect, *got);
}

TYPED_TEST(PairwiseLineStringDistanceTest, MultiToSingleEmpty)
{
  auto got = pairwise_linestring_distance(this->empty_multilinestring(), this->empty_linestring());
  auto expect = fixed_width_column_wrapper<TypeParam>{};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect, *got);
}

TYPED_TEST(PairwiseLineStringDistanceTest, MultiToMultiEmpty)
{
  auto got =
    pairwise_linestring_distance(this->empty_multilinestring(), this->empty_multilinestring());
  auto expect = fixed_width_column_wrapper<TypeParam>{};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect, *got);
}

struct PairwiseLineStringDistanceFailOnSizeTest : public EmptyAndOneGeometryColumnFixture {};

TEST_F(PairwiseLineStringDistanceFailOnSizeTest, SizeMismatch)
{
  EXPECT_THROW(pairwise_linestring_distance(this->empty_linestring(), this->one_linestring()),
               cuspatial::logic_error);
}

TEST_F(PairwiseLineStringDistanceFailOnSizeTest, SizeMismatch2)
{
  EXPECT_THROW(pairwise_linestring_distance(this->one_linestring(), this->empty_multilinestring()),
               cuspatial::logic_error);
}

struct PairwiseLineStringDistanceFailOnTypeTest : public EmptyGeometryColumnFixtureMultipleTypes {};

TEST_F(PairwiseLineStringDistanceFailOnTypeTest, CoordinateTypeMismatch)
{
  EXPECT_THROW(pairwise_linestring_distance(EmptyGeometryColumnBase<float>::empty_linestring(),
                                            EmptyGeometryColumnBase<double>::empty_linestring()),
               cuspatial::logic_error);
}

TEST_F(PairwiseLineStringDistanceFailOnTypeTest, GeometryTypeMismatch)
{
  EXPECT_THROW(pairwise_linestring_distance(EmptyGeometryColumnBase<float>::empty_linestring(),
                                            EmptyGeometryColumnBase<float>::empty_polygon()),
               cuspatial::logic_error);
}
