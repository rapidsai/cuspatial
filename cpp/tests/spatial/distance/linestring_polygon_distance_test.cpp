/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <initializer_list>
#include <memory>

using namespace cuspatial;
using namespace cuspatial::test;

using namespace cudf;
using namespace cudf::test;

template <typename T>
struct PairwiseLinestringPolygonDistanceTest : ::testing::Test {
  rmm::cuda_stream_view stream() { return cudf::get_default_stream(); }
  void SetUp()
  {
    collection_type_id _;
    std::tie(_, empty_linestring_column)      = make_linestring_column<T>({0}, {}, stream());
    std::tie(_, empty_multilinestring_column) = make_linestring_column<T>({0}, {0}, {}, stream());
    std::tie(_, empty_polygon_column)         = make_polygon_column<T>({0}, {0}, {}, stream());
    std::tie(_, empty_multipolygon_column)    = make_polygon_column<T>({0}, {0}, {0}, {}, stream());
  }

  geometry_column_view empty_linestring()
  {
    return geometry_column_view(
      empty_linestring_column->view(), collection_type_id::SINGLE, geometry_type_id::LINESTRING);
  }

  geometry_column_view empty_multilinestring()
  {
    return geometry_column_view(empty_multilinestring_column->view(),
                                collection_type_id::MULTI,
                                geometry_type_id::LINESTRING);
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

  void run_single(geometry_column_view linestrings,
                  geometry_column_view polygons,
                  std::initializer_list<T> expected)
  {
    auto got = pairwise_linestring_polygon_distance(linestrings, polygons);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*got, fixed_width_column_wrapper<T>(expected));
  }

  std::unique_ptr<cudf::column> empty_linestring_column;
  std::unique_ptr<cudf::column> empty_multilinestring_column;
  std::unique_ptr<cudf::column> empty_polygon_column;
  std::unique_ptr<cudf::column> empty_multipolygon_column;
};

struct PairwiseLinestringPolygonDistanceTestUntyped : testing::Test {
  rmm::cuda_stream_view stream() { return cudf::get_default_stream(); }
};

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

TEST_F(PairwiseLinestringPolygonDistanceTestUntyped, SizeMismatch)
{
  auto [ptype, linestrings] =
    make_linestring_column<float>({0, 1, 2}, {0, 1, 2}, {0.0, 0.0, 1.0, 1.0}, this->stream());

  auto [polytype, polygons] =
    make_polygon_column<float>({0, 1}, {0, 1}, {0, 4}, {1, 1, 1, 2, 2, 2, 1, 1}, this->stream());

  auto linestrings_view =
    geometry_column_view(linestrings->view(), ptype, geometry_type_id::LINESTRING);
  auto polygons_view = geometry_column_view(polygons->view(), polytype, geometry_type_id::POLYGON);

  EXPECT_THROW(pairwise_linestring_polygon_distance(linestrings_view, polygons_view),
               cuspatial::logic_error);
};

TEST_F(PairwiseLinestringPolygonDistanceTestUntyped, TypeMismatch)
{
  auto [ptype, linestrings] =
    make_linestring_column<double>({0, 1}, {0, 1}, {0.0, 0.0}, this->stream());

  auto [polytype, polygons] =
    make_polygon_column<float>({0, 1}, {0, 1}, {0, 4}, {1, 1, 1, 2, 2, 2, 1, 1}, this->stream());

  auto linestrings_view =
    geometry_column_view(linestrings->view(), ptype, geometry_type_id::LINESTRING);
  auto polygons_view = geometry_column_view(polygons->view(), polytype, geometry_type_id::POLYGON);

  EXPECT_THROW(pairwise_linestring_polygon_distance(linestrings_view, polygons_view),
               cuspatial::logic_error);
};

TEST_F(PairwiseLinestringPolygonDistanceTestUntyped, WrongGeometryType)
{
  auto [ptype, points] = make_point_column<double>({0, 1}, {0.0, 0.0}, this->stream());

  auto [polytype, polygons] =
    make_polygon_column<float>({0, 1}, {0, 1}, {0, 4}, {1, 1, 1, 2, 2, 2, 1, 1}, this->stream());

  auto points_view   = geometry_column_view(points->view(), ptype, geometry_type_id::POINT);
  auto polygons_view = geometry_column_view(polygons->view(), polytype, geometry_type_id::POLYGON);

  EXPECT_THROW(pairwise_linestring_polygon_distance(points_view, polygons_view),
               cuspatial::logic_error);
};
