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

#include <cuspatial/column/geometry_column_view.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/types.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/types.hpp>

namespace cuspatial {

namespace {

cudf::data_type leaf_data_type(cudf::column_view const& column)
{
  if (column.type() != cudf::data_type{cudf::type_id::LIST}) return column.type();
  return leaf_data_type(column.child(cudf::lists_column_view::child_column_index));
}

}  // namespace

geometry_column_view::geometry_column_view(cudf::column_view const& column,
                                           collection_type_id collection_type,
                                           geometry_type_id geometry_type)
  : cudf::column_view(column), _collection_type(collection_type), _geometry_type(geometry_type)
{
  // Single point array is FixedSizeList<f>[n_dim]
  if (geometry_type == geometry_type_id::POINT && collection_type == collection_type_id::SINGLE) {
    CUSPATIAL_EXPECTS(type() == cudf::data_type{cudf::type_id::FLOAT32} ||
                        type() == cudf::data_type{cudf::type_id::FLOAT64},
                      "Single point column must be a non-nested column.");
  } else {
    CUSPATIAL_EXPECTS(
      type() == cudf::data_type{cudf::type_id::LIST},
      "Geometry column other than a single point column must be a list type column.");
  }
}

cudf::data_type geometry_column_view::coordinate_type() const { return leaf_data_type(*this); }

cudf::column_view geometry_column_view::child() const
{
  CUSPATIAL_EXPECTS(type() == cudf::data_type{cudf::type_id::LIST},
                    "Only LIST column can contain offsets and child.");
  return cudf::column_view::child(cudf::lists_column_view::child_column_index);
}

cudf::column_view geometry_column_view::offsets() const
{
  CUSPATIAL_EXPECTS(type() == cudf::data_type{cudf::type_id::LIST},
                    "Only LIST column can contain offsets and child.");
  return cudf::column_view::child(cudf::lists_column_view::offsets_column_index);
}

}  // namespace cuspatial
