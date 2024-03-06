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
struct PairwisePolygonDistanceTestBase : ::testing::Test {
  void run_single(geometry_column_view lhs,
                  geometry_column_view rhs,
                  std::initializer_list<T> expected)
  {
    auto got = pairwise_polygon_distance(lhs, rhs);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*got, fixed_width_column_wrapper<T>(expected));
  }
  rmm::cuda_stream_view stream() { return cudf::get_default_stream(); }
};

template <typename T>
struct PairwisePolygonDistanceTestEmpty : PairwisePolygonDistanceTestBase<T> {
  void SetUp()
  {
    [[maybe_unused]] collection_type_id _;
    std::tie(_, empty_polygon_column) = make_polygon_column<T>({0}, {0}, {}, this->stream());
    std::tie(_, empty_multipolygon_column) =
      make_polygon_column<T>({0}, {0}, {0}, {}, this->stream());
  }

  geometry_column_view empty_polygon()
  {
    return geometry_column_view(
      empty_polygon_column->view(), collection_type_id::SINGLE, geometry_type_id::POLYGON);
  }

  geometry_column_view empty_multipolygon()
  {
    return geometry_column_view(
      empty_multipolygon_column->view(), collection_type_id::MULTI, geometry_type_id::POLYGON);
  }

  std::unique_ptr<cudf::column> empty_polygon_column;
  std::unique_ptr<cudf::column> empty_multipolygon_column;
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(PairwisePolygonDistanceTestEmpty, TestTypes);

struct PairwisePolygonDistanceTestUntyped : testing::Test {
  rmm::cuda_stream_view stream() { return cudf::get_default_stream(); }
};
TYPED_TEST(PairwisePolygonDistanceTestEmpty, SingleToSingleEmpty)
{
  CUSPATIAL_RUN_TEST(this->run_single, this->empty_polygon(), this->empty_polygon(), {});
};

TYPED_TEST(PairwisePolygonDistanceTestEmpty, SingleToMultiEmpty)
{
  CUSPATIAL_RUN_TEST(this->run_single, this->empty_polygon(), this->empty_multipolygon(), {});
};

TYPED_TEST(PairwisePolygonDistanceTestEmpty, MultiToSingleEmpty)
{
  CUSPATIAL_RUN_TEST(this->run_single, this->empty_multipolygon(), this->empty_polygon(), {});
};

TYPED_TEST(PairwisePolygonDistanceTestEmpty, MultiToMultiEmpty)
{
  CUSPATIAL_RUN_TEST(this->run_single, this->empty_multipolygon(), this->empty_multipolygon(), {});
};

TEST_F(PairwisePolygonDistanceTestUntyped, SizeMismatch)
{
  auto [ptype, polygons1] = make_polygon_column<float>(
    {0, 1, 2}, {0, 4, 8}, {0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0}, this->stream());

  auto [polytype, polygons2] =
    make_polygon_column<float>({0, 1}, {0, 1}, {0, 4}, {1, 1, 1, 2, 2, 2, 1, 1}, this->stream());

  auto polygons1_view =
    geometry_column_view(polygons1->view(), ptype, geometry_type_id::LINESTRING);
  auto polygons2_view =
    geometry_column_view(polygons1->view(), polytype, geometry_type_id::POLYGON);

  EXPECT_THROW(pairwise_polygon_distance(polygons1_view, polygons2_view), cuspatial::logic_error);
};

TEST_F(PairwisePolygonDistanceTestUntyped, TypeMismatch)
{
  auto [ptype, polygons1] = make_polygon_column<double>(
    {0, 1, 2}, {0, 4, 8}, {0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0}, this->stream());

  auto [polytype, polygons2] =
    make_polygon_column<float>({0, 1}, {0, 1}, {0, 4}, {1, 1, 1, 2, 2, 2, 1, 1}, this->stream());

  auto polygons1_view =
    geometry_column_view(polygons1->view(), ptype, geometry_type_id::LINESTRING);
  auto polygons2_view =
    geometry_column_view(polygons2->view(), polytype, geometry_type_id::POLYGON);

  EXPECT_THROW(pairwise_polygon_distance(polygons1_view, polygons2_view), cuspatial::logic_error);
};

TEST_F(PairwisePolygonDistanceTestUntyped, WrongGeometryType)
{
  auto [ptype, points] = make_point_column<float>({0, 1}, {0.0, 0.0}, this->stream());

  auto [polytype, polygons] =
    make_polygon_column<float>({0, 1}, {0, 1}, {0, 4}, {1, 1, 1, 2, 2, 2, 1, 1}, this->stream());

  auto points_view   = geometry_column_view(points->view(), ptype, geometry_type_id::POINT);
  auto polygons_view = geometry_column_view(polygons->view(), polytype, geometry_type_id::POLYGON);

  EXPECT_THROW(pairwise_polygon_distance(points_view, polygons_view), cuspatial::logic_error);
};
