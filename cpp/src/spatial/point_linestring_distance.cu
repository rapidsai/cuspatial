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

#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/experimental/point_linestring_distance.cuh>
#include <cuspatial/experimental/type_utils.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <type_traits>

namespace cuspatial {

namespace {

struct pairwise_point_linestring_distance_functor {
  template <typename T, CUDF_ENABLE_IF(std::is_floating_point_v<T>)>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& points_x,
                                           cudf::column_view const& points_y,
                                           cudf::column_view const& linestring_offsets,
                                           cudf::column_view const& linestring_points_x,
                                           cudf::column_view const& linestring_points_y,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
  {
    auto const num_pairs = static_cast<cudf::size_type>(points_x.size());
    auto output          = cudf::make_numeric_column(
      points_x.type(), num_pairs, cudf::mask_state::UNALLOCATED, stream, mr);

    auto points_begin = make_cartesian_2d_iterator(points_x.begin<T>(), points_y.begin<T>());
    auto linestring_points_begin =
      make_cartesian_2d_iterator(linestring_points_x.begin<T>(), linestring_points_y.begin<T>());
    auto output_begin = output->mutable_view().begin<T>();

    pairwise_point_linestring_distance(points_begin,
                                       points_begin + points_x.size(),
                                       linestring_offsets.begin<T>(),
                                       linestring_points_begin,
                                       linestring_points_begin + linestring_points_x.size(),
                                       output_begin,
                                       stream);

    return output;
  }

  template <typename T, CUDF_ENABLE_IF(!std::is_floating_point_v<T>), typename... Args>
  std::unique_ptr<cudf::column> operator()(Args&&...)

  {
    CUSPATIAL_FAIL("Point-linestring distance API only supports floating point coordinates.");
  }
};

}  // namespace

namespace detail {

std::unique_ptr<cudf::column> pairwise_point_linestring_distance(
  cudf::column_view const& points_x,
  cudf::column_view const& points_y,
  cudf::column_view const& linestring_offsets,
  cudf::column_view const& linestring_points_x,
  cudf::column_view const& linestring_points_y,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  CUSPATIAL_EXPECTS(points_x.size() == points_y.size(),
                    "Mismatch number of coordinates in the points array.");

  CUSPATIAL_EXPECTS(linestring_points_x.size() == linestring_points_y.size(),
                    "Mismatch number of coordinates in the linestring array.");

  CUSPATIAL_EXPECTS(points_x.size() == linestring_offsets.size(),
                    "Mismatch number of points and linestrings.");

  CUSPATIAL_EXPECTS(points_x.type() == points_y.type() &&
                      points_x.type() == linestring_points_x.type() &&
                      points_x.type() == linestring_points_y.type(),
                    "Points and linestring coordinates must have the same type.");

  CUSPATIAL_EXPECTS(
    !(points_x.has_nulls() || points_y.has_nulls() || linestring_offsets.has_nulls() ||
      linestring_points_x.has_nulls() || linestring_points_y.has_nulls()),
    "All inputs must not have nulls.");

  if (linestring_offsets.size() == 0) return cudf::empty_like(points_x);

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
  cudf::column_view const& linestring_offsets,
  cudf::column_view const& linestring_points_x,
  cudf::column_view const& linestring_points_y,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
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
