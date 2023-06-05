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

#include <cuspatial/types.hpp>

#include <cudf/lists/lists_column_view.hpp>
#include <cudf/types.hpp>

namespace cuspatial {

/**
 * @brief A non-owning, immutable view of a geometry column.
 *
 * @ingroup cuspatial_types
 *
 * A geometry column is GeoArrow compliant, except that the data type for
 * the coordinates is List<T>, instead of FixedSizeList<T>[n_dim]. This is
 * because libcudf does not support FixedSizeList type. Currently, an
 * even sequence (0, 2, 4, ...) is used for the offsets of the coordinate
 * column.
 */
class geometry_column_view : private cudf::lists_column_view {
 public:
  geometry_column_view(cudf::column_view const& column,
                       collection_type_id collection_type,
                       geometry_type_id geometry_type);
  geometry_column_view(geometry_column_view&&)      = default;
  geometry_column_view(const geometry_column_view&) = default;
  ~geometry_column_view()                           = default;

  geometry_column_view& operator=(geometry_column_view const&) = default;

  geometry_column_view& operator=(geometry_column_view&&) = default;

  geometry_type_id geometry_type() const { return _geometry_type; }

  collection_type_id collection_type() const { return _collection_type; }

  cudf::data_type coordinate_type() const;

  using cudf::lists_column_view::child;
  using cudf::lists_column_view::offsets;
  using cudf::lists_column_view::size;

 protected:
  collection_type_id _collection_type;
  geometry_type_id _geometry_type;
};

}  // namespace cuspatial
