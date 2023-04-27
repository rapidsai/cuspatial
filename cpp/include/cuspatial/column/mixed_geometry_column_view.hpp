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

#include <cuspatial/column/geometry_column_view.hpp>

#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

namespace cuspatial {

/**
 * @ingroup cuspatial_types
 * @brief A non-owning, immutable view to mixed type geometry view
 *
 * This column view implements a specialization of [arrow dense union layout][1], with
 * exactly 6 child column representing 6 types of geometry column used in cuspatial.
 *
 * @note A union layout does not hold null fields on the top level. This means `types_buffer`
 * and `offsets_buffer` must be non-nullable. This also implies that union column cannot hold
 * a "typeless null": each row must contain a null one of its children. We compensate for this
 * missing capability by allowing the type of a typeless null row to be set as union_type::NULL.
 * The corresponding offset of that row is undefined.
 *
 * @throw cuspatial::logic_error if `types_buffer` data type is not INT8.
 * @throw cuspatial::logic_error if `offsets_buffer` data type is not INT32.
 * @throw cuspatial::logic_error if `points_column` is not a `SINGLE` `POINT` geometry column.
 * @throw cuspatial::logic_error if `linestrings_column` is not a `SINGLE` `LINESTRING` geometry
 * column.
 * @throw cuspatial::logic_error if `polygons_column` is not a `SINGLE` `POLYGON` geometry column.
 * @throw cuspatial::logic_error if `multipoints_column` is not a `MULTI` `POINT` geometry column.
 * @throw cuspatial::logic_error if `multilinestrings_column` is not a `MULTI` `LINESTRING` geometry
 * column.
 * @throw cuspatial::logic_error if `multipolygons_column` is not a `MULTI` `POLYGON` geometry
 * column.
 *
 * [1] https://arrow.apache.org/docs/format/Columnar.html#dense-union
 */
class mixed_geometry_column_view {
 public:
  mixed_geometry_column_view(cudf::column_view const& types_buffer,
                             cudf::column_view const& offsets_buffer,
                             geometry_column_view const& points_column,
                             geometry_column_view const& linestrings_column,
                             geometry_column_view const& polygons_column,
                             geometry_column_view const& multipoints_column,
                             geometry_column_view const& multilinestrings_column,
                             geometry_column_view const& multipolygons_column);

  mixed_geometry_column_view(mixed_geometry_column_view&&)      = default;
  mixed_geometry_column_view(const mixed_geometry_column_view&) = default;
  ~mixed_geometry_column_view()                                 = default;

  mixed_geometry_column_view& operator=(mixed_geometry_column_view const&) = default;
  mixed_geometry_column_view& operator=(mixed_geometry_column_view&&)      = default;

 private:
  cudf::column_view types_buffer;
  cudf::column_view offsets_buffer;

  geometry_column_view points_column;
  geometry_column_view linestrings_column;
  geometry_column_view polygons_column;
  geometry_column_view multipoints_column;
  geometry_column_view multilinestrings_column;
  geometry_column_view multipolygons_column;
};

}  // namespace cuspatial
