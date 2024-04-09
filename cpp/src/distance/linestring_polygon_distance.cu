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

#include "../utility/multi_geometry_dispatch.hpp"

#include <cuspatial/column/geometry_column_view.hpp>
#include <cuspatial/distance.cuh>
#include <cuspatial/error.hpp>
#include <cuspatial/iterator_factory.cuh>
#include <cuspatial/range/multilinestring_range.cuh>
#include <cuspatial/range/multipolygon_range.cuh>
#include <cuspatial/types.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <memory>
#include <type_traits>

namespace cuspatial {

namespace detail {

namespace {

template <collection_type_id is_multi_linestring, collection_type_id is_multi_polygon>
struct pairwise_linestring_polygon_distance_impl {
  using SizeType = cudf::device_span<cudf::size_type const>::size_type;

  template <typename T, CUDF_ENABLE_IF(std::is_floating_point_v<T>)>
  std::unique_ptr<cudf::column> operator()(geometry_column_view const& multilinestrings,
                                           geometry_column_view const& multipolygons,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
  {
    auto multilinestrings_range =
      make_multilinestring_range<is_multi_linestring, T, cudf::size_type>(multilinestrings);
    auto multipolygons_range =
      make_multipolygon_range<is_multi_polygon, T, cudf::size_type>(multipolygons);

    auto output = cudf::make_numeric_column(multilinestrings.coordinate_type(),
                                            multilinestrings.size(),
                                            cudf::mask_state::UNALLOCATED,
                                            stream,
                                            mr);

    cuspatial::pairwise_linestring_polygon_distance(
      multilinestrings_range, multipolygons_range, output->mutable_view().begin<T>(), stream);
    return output;
  }

  template <typename T, CUDF_ENABLE_IF(!std::is_floating_point_v<T>), typename... Args>
  std::unique_ptr<cudf::column> operator()(Args&&...)

  {
    CUSPATIAL_FAIL("linestring-polygon distance API only supports floating point coordinates.");
  }
};

}  // namespace

template <collection_type_id is_multi_linestring, collection_type_id is_multi_polygon>
struct pairwise_linestring_polygon_distance {
  std::unique_ptr<cudf::column> operator()(geometry_column_view const& multilinestrings,
                                           geometry_column_view const& multipolygons,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
  {
    CUSPATIAL_EXPECTS(multilinestrings.geometry_type() == geometry_type_id::LINESTRING &&
                        multipolygons.geometry_type() == geometry_type_id::POLYGON,
                      "Unexpected input geometry types.");

    CUSPATIAL_EXPECTS(multilinestrings.coordinate_type() == multipolygons.coordinate_type(),
                      "Inputs must have the same coordinate type.");

    CUSPATIAL_EXPECTS(multilinestrings.size() == multipolygons.size(),
                      "Inputs must have the same number of rows.");

    return cudf::type_dispatcher(
      multilinestrings.coordinate_type(),
      pairwise_linestring_polygon_distance_impl<is_multi_linestring, is_multi_polygon>{},
      multilinestrings,
      multipolygons,
      stream,
      mr);
  }
};

}  // namespace detail

std::unique_ptr<cudf::column> pairwise_linestring_polygon_distance(
  geometry_column_view const& multilinestrings,
  geometry_column_view const& multipolygons,
  rmm::device_async_resource_ref mr)
{
  return multi_geometry_double_dispatch<detail::pairwise_linestring_polygon_distance>(
    multilinestrings.collection_type(),
    multipolygons.collection_type(),
    multilinestrings,
    multipolygons,
    rmm::cuda_stream_default,
    mr);
}

}  // namespace cuspatial
