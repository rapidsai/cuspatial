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

#pragma once

#include <cuspatial_test/base_fixture.hpp>
#include <cuspatial_test/column_factories.hpp>

#include <cuspatial/column/geometry_column_view.hpp>
#include <cuspatial/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <cudf/utilities/default_stream.hpp>

#include <gtest/gtest.h>

namespace cuspatial {
namespace test {

/**
 * @brief Test Fixture that initializes several commonly used geometry columns.
 *
 * @tparam T Type of the coordinates
 */
template <typename T>
class CommonGeometryColumnFixture : BaseFixture {
 protected:

  // TODO: explore SetUpTestSuite to perform per-test-suite initialization, saving expenses.
  // However, this requires making `stream()` method a static member.
  void SetUp()
  {
    collection_type_id _;

    std::tie(_, empty_point_column)           = make_point_column<T>({}, stream());
    std::tie(_, empty_linestring_column)      = make_linestring_column<T>({0}, {}, stream());
    std::tie(_, empty_polygon_column)         = make_polygon_column<T>({0}, {0}, {}, stream());
    std::tie(_, empty_multipoint_column)      = make_point_column<T>({0}, {}, stream());
    std::tie(_, empty_multilinestring_column) = make_linestring_column<T>({0}, {0}, {}, stream());
    std::tie(_, empty_multipolygon_column)    = make_polygon_column<T>({0}, {0}, {0}, {}, stream());
  }

  geometry_column_view empty_point()
  {
    return geometry_column_view(
      empty_point_column->view(), collection_type_id::SINGLE, geometry_type_id::POINT);
  }

  geometry_column_view empty_multipoint()
  {
    return geometry_column_view(
      empty_multipoint_column->view(), collection_type_id::MULTI, geometry_type_id::POINT);
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

  std::unique_ptr<cudf::column> empty_point_column;
  std::unique_ptr<cudf::column> empty_linestring_column;
  std::unique_ptr<cudf::column> empty_polygon_column;
  std::unique_ptr<cudf::column> empty_multipoint_column;
  std::unique_ptr<cudf::column> empty_multilinestring_column;
  std::unique_ptr<cudf::column> empty_multipolygon_column;
};

}  // namespace test
}  // namespace cuspatial
