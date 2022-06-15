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

#include <cuspatial/error.hpp>
#include <cuspatial/experimental/point_linestring_distance.cuh>
#include <cuspatial/experimental/type_utils.hpp>
#include <cuspatial/utility/vec_2d.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <limits>
#include <memory>
#include <type_traits>

namespace cuspatial {
namespace detail {

struct pairwise_point_linestring_distance_functor {
  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    cudf::column_view const& points_x,
    cudf::column_view const& points_y,
    cudf::device_span<cudf::size_type const> linestring_offsets,
    cudf::column_view const& linestring_points_x,
    cudf::column_view const& linestring_points_y,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    auto const num_pairs = static_cast<cudf::size_type>(linestring_offsets.size());

    auto distances = cudf::make_numeric_column(
      cudf::data_type{cudf::type_to_id<T>()}, num_pairs, cudf::mask_state::UNALLOCATED, stream, mr);

    auto points_it = make_cartesian_2d_iterator(points_x.begin<T>(), points_y.begin<T>());
    auto linestring_coords_it =
      make_cartesian_2d_iterator(linestring_points_x.begin<T>(), linestring_points_y.begin<T>());

    pairwise_point_linestring_distance(points_it,
                                       points_it + points_x.size(),
                                       linestring_offsets.begin(),
                                       linestring_coords_it,
                                       linestring_coords_it + linestring_points_x.size(),
                                       distances->mutable_view().begin<T>(),
                                       stream);

    return distances;
  }

  template <typename T, typename... Args>
  std::enable_if_t<not std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    Args&&...)
  {
    CUSPATIAL_FAIL("Linestring distances only supports floating point coordinates.");
  }
};

std::unique_ptr<cudf::column> pairwise_point_linestring_distance(
  cudf::column_view const& points_x,
  cudf::column_view const& points_y,
  cudf::device_span<cudf::size_type const> linestring_offsets,
  cudf::column_view const& linestring_points_x,
  cudf::column_view const& linestring_points_y,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  CUSPATIAL_EXPECTS(points_x.size() == points_y.size(),
                    "The lengths of linestring coordinates arrays mismatch.");

  CUSPATIAL_EXPECTS(points_x.size() == static_cast<cudf::size_type>(linestring_offsets.size()),
                    "The number of points mismatches the number of linestrings.");

  CUSPATIAL_EXPECTS(
    points_x.type() == points_y.type() and linestring_points_x.type() == linestring_points_y.type(),
    "The types of the coordinate arrays mismatch.");

  if (points_x.size() == 0) { return cudf::empty_like(points_x); }

  return cudf::type_dispatcher(points_x.type(),
                               pairwise_point_linestring_distance_functor{},
                               points_x,
                               points_y,
                               linestring_offsets,
                               linestring_points_x,
                               linestring_points_y,
                               stream,
                               mr);
}

}  // namespace detail

std::unique_ptr<cudf::column> pairwise_point_linestring_distance(
  cudf::column_view const& points_x,
  cudf::column_view const& points_y,
  cudf::device_span<cudf::size_type const> linestring_offsets,
  cudf::column_view const& linestring_points_x,
  cudf::column_view const& linestring_points_y,
  rmm::mr::device_memory_resource* mr)
{
  return detail::pairwise_point_linestring_distance(points_x,
                                                    points_y,
                                                    linestring_offsets,
                                                    linestring_points_x,
                                                    linestring_points_y,
                                                    rmm::cuda_stream_default,
                                                    mr);
}

}  // namespace cuspatial
