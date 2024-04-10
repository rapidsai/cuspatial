/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cuspatial_test/column_factories.hpp>
#include <cuspatial_test/geometry_fixtures.hpp>
#include <cuspatial_test/vector_equality.hpp>

#include <cuspatial/column/geometry_column_view.hpp>
#include <cuspatial/distance.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/geometry/vec_2d.hpp>
#include <cuspatial/types.hpp>

#include <cudf/utilities/default_stream.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <initializer_list>
#include <memory>

using namespace cuspatial;
using namespace cuspatial::test;

using namespace cudf;
using namespace cudf::test;

template <typename T>
struct PairwiseLinestringPolygonDistanceTest : EmptyGeometryColumnFixture<T> {
  void run_single(geometry_column_view linestrings,
                  geometry_column_view polygons,
                  std::initializer_list<T> expected)
  {
    auto got = pairwise_linestring_polygon_distance(linestrings, polygons);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*got, fixed_width_column_wrapper<T>(expected));
  }
};

struct PairwiseLinestringPolygonDistanceFailOnSizeTest : EmptyAndOneGeometryColumnFixture {};

struct PairwiseLinestringPolygonDistanceFailOnTypeTest : EmptyGeometryColumnFixtureMultipleTypes {};

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_CASE(PairwiseLinestringPolygonDistanceTest, TestTypes);

TYPED_TEST(PairwiseLinestringPolygonDistanceTest, SingleToSingleEmpty)
{
  CUSPATIAL_RUN_TEST(this->run_single, this->empty_linestring(), this->empty_polygon(), {});
};

TYPED_TEST(PairwiseLinestringPolygonDistanceTest, SingleToMultiEmpty)
{
  CUSPATIAL_RUN_TEST(this->run_single, this->empty_linestring(), this->empty_multipolygon(), {});
};

TYPED_TEST(PairwiseLinestringPolygonDistanceTest, MultiToSingleEmpty)
{
  CUSPATIAL_RUN_TEST(this->run_single, this->empty_multilinestring(), this->empty_polygon(), {});
};

TYPED_TEST(PairwiseLinestringPolygonDistanceTest, MultiToMultiEmpty)
{
  CUSPATIAL_RUN_TEST(
    this->run_single, this->empty_multilinestring(), this->empty_multipolygon(), {});
};

TEST_F(PairwiseLinestringPolygonDistanceFailOnSizeTest, SizeMismatch)
{
  EXPECT_THROW(pairwise_linestring_polygon_distance(this->empty_linestring(), this->one_polygon()),
               cuspatial::logic_error);
};

TEST_F(PairwiseLinestringPolygonDistanceFailOnTypeTest, CoordinateTypeMismatch)
{
  EXPECT_THROW(
    pairwise_linestring_polygon_distance(EmptyGeometryColumnBase<float>::empty_linestring(),
                                         EmptyGeometryColumnBase<double>::empty_polygon()),
    cuspatial::logic_error);
};

TEST_F(PairwiseLinestringPolygonDistanceFailOnTypeTest, WrongGeometryType)
{
  EXPECT_THROW(
    pairwise_linestring_polygon_distance(EmptyGeometryColumnBase<float>::empty_point(),
                                         EmptyGeometryColumnBase<float>::empty_polygon()),
    cuspatial::logic_error);
};
