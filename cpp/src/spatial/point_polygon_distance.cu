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

#include "../utility/iterator.hpp"
#include "../utility/multi_geometry_dispatch.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuspatial/column/geometry_column_view.hpp>
#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/experimental/iterator_factory.cuh>
#include <cuspatial/experimental/point_polygon_distance.cuh>
#include <cuspatial/experimental/ranges/multipoint_range.cuh>
#include <cuspatial/experimental/ranges/multipolygon_range.cuh>
#include <cuspatial/types.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <memory>
#include <type_traits>

namespace cuspatial {

namespace detail {

namespace {

template <collection_type_id is_multi_point, collection_type_id is_multi_polygon>
struct pairwise_point_polygon_distance_impl {
  using SizeType = cudf::device_span<cudf::size_type const>::size_type;

  template <typename T, CUDF_ENABLE_IF(std::is_floating_point_v<T>)>
  std::unique_ptr<cudf::column> operator()(geometry_column_view const& multipoints,
                                           geometry_column_view const& multipolygons,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
  {
    auto multipoints_range = make_multipoint_range<is_multi_point, T, cudf::size_type>(multipoints);
    auto multipolygons_range =
      make_multipolygon_range<is_multi_polygon, T, cudf::size_type>(multipolygons);

    auto output = cudf::make_numeric_column(
      multipoints.coordinate_type(), multipoints.size(), cudf::mask_state::UNALLOCATED, stream, mr);

    cuspatial::pairwise_point_polygon_distance(
      multipoints_range, multipolygons_range, output->mutable_view().begin<T>(), stream);
    return output;
  }

  template <typename T, CUDF_ENABLE_IF(!std::is_floating_point_v<T>), typename... Args>
  std::unique_ptr<cudf::column> operator()(Args&&...)

  {
    CUSPATIAL_FAIL("Point-polygon distance API only supports floating point coordinates.");
  }
};

}  // namespace

template <collection_type_id is_multi_point, collection_type_id is_multi_polygon>
struct pairwise_point_polygon_distance {
  std::unique_ptr<cudf::column> operator()(geometry_column_view const& multipoints,
                                           geometry_column_view const& multipolygons,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
  {
    return cudf::type_dispatcher(
      multipoints.coordinate_type(),
      pairwise_point_polygon_distance_impl<is_multi_point, is_multi_polygon>{},
      multipoints,
      multipolygons,
      stream,
      mr);
  }
};

}  // namespace detail

std::unique_ptr<cudf::column> pairwise_point_polygon_distance(
  geometry_column_view const& multipoints,
  geometry_column_view const& multipolygons,
  rmm::mr::device_memory_resource* mr)
{
  CUSPATIAL_EXPECTS(multipoints.geometry_type() == geometry_type_id::POINT &&
                      multipolygons.geometry_type() == geometry_type_id::POLYGON,
                    "Unexpected input geometry types.");

  CUSPATIAL_EXPECTS(multipoints.coordinate_type() == multipolygons.coordinate_type(),
                    "Input geometries must have the same coordinate data types.");

  return multi_geometry_double_dispatch<detail::pairwise_point_polygon_distance>(
    multipoints.collection_type(),
    multipolygons.collection_type(),
    multipoints,
    multipolygons,
    rmm::cuda_stream_default,
    mr);
}

}  // namespace cuspatial
