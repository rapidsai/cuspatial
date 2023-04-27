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

#include <cuspatial/column/mixed_geometry_column_view.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/types.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

namespace cuspatial {

mixed_geometry_column_view::mixed_geometry_column_view(
  cudf::column_view const& types_buffer,
  cudf::column_view const& offsets_buffer,
  geometry_column_view const& points_column,
  geometry_column_view const& linestrings_column,
  geometry_column_view const& polygons_column,
  geometry_column_view const& multipoints_column,
  geometry_column_view const& multilinestrings_column,
  geometry_column_view const& multipolygons_column)
  : types_buffer(types_buffer),
    offsets_buffer(offsets_buffer),
    points_column(points_column),
    linestrings_column(linestrings_column),
    polygons_column(polygons_column),
    multipoints_column(multipoints_column),
    multilinestrings_column(multilinestrings_column),
    multipolygons_column(multipolygons_column)
{
  CUSPATIAL_EXPECTS(
    types_buffer.type() == cudf::data_type{cudf::type_to_id<mixed_geometry_type_t>()},
    "types_buffer must have INT8 data type.");
  CUSPATIAL_EXPECTS(offsets_buffer.type() == cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                    "offsets_buffer must use cudf's size type.");

  CUSPATIAL_EXPECTS(points_column.collection_type() == collection_type_id::SINGLE &&
                      points_column.geometry_type() == geometry_type_id::POINT,
                    "points_column must be a single point column.");
  CUSPATIAL_EXPECTS(linestrings_column.collection_type() == collection_type_id::SINGLE &&
                      linestrings_column.geometry_type() == geometry_type_id::LINESTRING,
                    "linestrings_column must be a single linestring column.");
  CUSPATIAL_EXPECTS(polygons_column.collection_type() == collection_type_id::SINGLE &&
                      polygons_column.geometry_type() == geometry_type_id::POLYGON,
                    "polygons_column must be a single polygon column.");
  CUSPATIAL_EXPECTS(multipoints_column.collection_type() == collection_type_id::MULTI &&
                      multipoints_column.geometry_type() == geometry_type_id::POINT,
                    "multipoints_column must be a multipoint column.");
  CUSPATIAL_EXPECTS(multilinestrings_column.collection_type() == collection_type_id::MULTI &&
                      multilinestrings_column.geometry_type() == geometry_type_id::LINESTRING,
                    "multilinestrings_column must be a multilinestring column.");
  CUSPATIAL_EXPECTS(multipolygons_column.collection_type() == collection_type_id::MULTI &&
                      multipolygons_column.geometry_type() == geometry_type_id::POLYGON,
                    "multipolygons_column must be a multipolygon column.");
}

}  // namespace cuspatial
