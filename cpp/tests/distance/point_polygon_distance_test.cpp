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

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <initializer_list>

using namespace cuspatial;
using namespace cuspatial::test;

using namespace cudf;
using namespace cudf::test;

template <typename T>
struct PairwisePointPolygonDistanceTest : public ::testing::Test {
  rmm::cuda_stream_view stream() { return rmm::cuda_stream_default; }

  void run_single(geometry_column_view points,
                  geometry_column_view polygons,
                  std::initializer_list<T> expected)
  {
    auto got = pairwise_point_polygon_distance(points, polygons);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*got, fixed_width_column_wrapper<T>(expected));
  }
};

struct PairwisePointPolygonDistanceTestUntyped : public ::testing::Test {
  rmm::cuda_stream_view stream() { return rmm::cuda_stream_default; }
};

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_CASE(PairwisePointPolygonDistanceTest, TestTypes);

TYPED_TEST(PairwisePointPolygonDistanceTest, SingleToSingleEmpty)
{
  using T = TypeParam;

  auto [ptype, points] = make_point_column<T>(std::initializer_list<T>{}, this->stream());

  auto [polytype, polygons] =
    make_polygon_column<T>({0}, {0}, std::initializer_list<T>{}, this->stream());

  CUSPATIAL_RUN_TEST(this->run_single,
                     geometry_column_view(points->view(), ptype, geometry_type_id::POINT),
                     geometry_column_view(polygons->view(), polytype, geometry_type_id::POLYGON),
                     {});
};

TYPED_TEST(PairwisePointPolygonDistanceTest, SingleToMultiEmpty)
{
  using T = TypeParam;

  auto [ptype, points] = make_point_column<T>(std::initializer_list<T>{}, this->stream());

  auto [polytype, polygons] =
    make_polygon_column<T>({0}, {0}, {0}, std::initializer_list<T>{}, this->stream());

  CUSPATIAL_RUN_TEST(this->run_single,
                     geometry_column_view(points->view(), ptype, geometry_type_id::POINT),
                     geometry_column_view(polygons->view(), polytype, geometry_type_id::POLYGON),
                     {});
};

TYPED_TEST(PairwisePointPolygonDistanceTest, MultiToSingleEmpty)
{
  using T = TypeParam;

  auto [ptype, points] = make_point_column<T>({0}, std::initializer_list<T>{}, this->stream());

  auto [polytype, polygons] =
    make_polygon_column<T>({0}, {0}, std::initializer_list<T>{}, this->stream());

  CUSPATIAL_RUN_TEST(this->run_single,
                     geometry_column_view(points->view(), ptype, geometry_type_id::POINT),
                     geometry_column_view(polygons->view(), polytype, geometry_type_id::POLYGON),
                     {});
};

TYPED_TEST(PairwisePointPolygonDistanceTest, MultiToMultiEmpty)
{
  using T = TypeParam;

  auto [ptype, points] = make_point_column<T>({0}, std::initializer_list<T>{}, this->stream());

  auto [polytype, polygons] =
    make_polygon_column<T>({0}, {0}, {0}, std::initializer_list<T>{}, this->stream());

  CUSPATIAL_RUN_TEST(this->run_single,
                     geometry_column_view(points->view(), ptype, geometry_type_id::POINT),
                     geometry_column_view(polygons->view(), polytype, geometry_type_id::POLYGON),
                     {});
};

TYPED_TEST(PairwisePointPolygonDistanceTest, SingleToSingleOnePair)
{
  using T = TypeParam;

  auto [ptype, points] = make_point_column<T>({0.0, 0.0}, this->stream());

  auto [polytype, polygons] =
    make_polygon_column<T>({0, 1}, {0, 4}, {1, 1, 1, 2, 2, 2, 1, 1}, this->stream());

  CUSPATIAL_RUN_TEST(this->run_single,
                     geometry_column_view(points->view(), ptype, geometry_type_id::POINT),
                     geometry_column_view(polygons->view(), polytype, geometry_type_id::POLYGON),
                     {1.4142135623730951});
};

TYPED_TEST(PairwisePointPolygonDistanceTest, SingleToMultiOnePair)
{
  using T = TypeParam;

  auto [ptype, points] = make_point_column<T>({0.0, 0.0}, this->stream());

  auto [polytype, polygons] =
    make_polygon_column<T>({0, 1}, {0, 1}, {0, 4}, {1, 1, 1, 2, 2, 2, 1, 1}, this->stream());

  CUSPATIAL_RUN_TEST(this->run_single,
                     geometry_column_view(points->view(), ptype, geometry_type_id::POINT),
                     geometry_column_view(polygons->view(), polytype, geometry_type_id::POLYGON),
                     {1.4142135623730951});
};

TYPED_TEST(PairwisePointPolygonDistanceTest, MultiToSingleOnePair)
{
  using T = TypeParam;

  auto [ptype, points] = make_point_column<T>({0, 1}, {0.0, 0.0}, this->stream());

  auto [polytype, polygons] =
    make_polygon_column<T>({0, 1}, {0, 4}, {1, 1, 1, 2, 2, 2, 1, 1}, this->stream());

  CUSPATIAL_RUN_TEST(this->run_single,
                     geometry_column_view(points->view(), ptype, geometry_type_id::POINT),
                     geometry_column_view(polygons->view(), polytype, geometry_type_id::POLYGON),
                     {1.4142135623730951});
};

TYPED_TEST(PairwisePointPolygonDistanceTest, MultiToMultiOnePair)
{
  using T = TypeParam;

  auto [ptype, points] = make_point_column<T>({0, 1}, {0.0, 0.0}, this->stream());

  auto [polytype, polygons] =
    make_polygon_column<T>({0, 1}, {0, 1}, {0, 4}, {1, 1, 1, 2, 2, 2, 1, 1}, this->stream());

  CUSPATIAL_RUN_TEST(this->run_single,
                     geometry_column_view(points->view(), ptype, geometry_type_id::POINT),
                     geometry_column_view(polygons->view(), polytype, geometry_type_id::POLYGON),
                     {1.4142135623730951});
};

TEST_F(PairwisePointPolygonDistanceTestUntyped, SizeMismatch)
{
  auto [ptype, points] = make_point_column<float>({0, 1, 2}, {0.0, 0.0, 1.0, 1.0}, this->stream());

  auto [polytype, polygons] =
    make_polygon_column<float>({0, 1}, {0, 1}, {0, 4}, {1, 1, 1, 2, 2, 2, 1, 1}, this->stream());

  auto points_view   = geometry_column_view(points->view(), ptype, geometry_type_id::POINT);
  auto polygons_view = geometry_column_view(polygons->view(), polytype, geometry_type_id::POLYGON);

  EXPECT_THROW(pairwise_point_polygon_distance(points_view, polygons_view), cuspatial::logic_error);
};

TEST_F(PairwisePointPolygonDistanceTestUntyped, TypeMismatch)
{
  auto [ptype, points] = make_point_column<double>({0, 1}, {0.0, 0.0}, this->stream());

  auto [polytype, polygons] =
    make_polygon_column<float>({0, 1}, {0, 1}, {0, 4}, {1, 1, 1, 2, 2, 2, 1, 1}, this->stream());

  auto points_view   = geometry_column_view(points->view(), ptype, geometry_type_id::POINT);
  auto polygons_view = geometry_column_view(polygons->view(), polytype, geometry_type_id::POLYGON);

  EXPECT_THROW(pairwise_point_polygon_distance(points_view, polygons_view), cuspatial::logic_error);
};

TEST_F(PairwisePointPolygonDistanceTestUntyped, WrongGeometryType)
{
  auto [ltype, lines] = make_linestring_column<double>({0, 1}, {0, 1}, {0.0, 0.0}, this->stream());

  auto [polytype, polygons] =
    make_polygon_column<float>({0, 1}, {0, 1}, {0, 4}, {1, 1, 1, 2, 2, 2, 1, 1}, this->stream());

  auto lines_view    = geometry_column_view(lines->view(), ltype, geometry_type_id::LINESTRING);
  auto polygons_view = geometry_column_view(polygons->view(), polytype, geometry_type_id::POLYGON);

  EXPECT_THROW(pairwise_point_polygon_distance(lines_view, polygons_view), cuspatial::logic_error);
};
