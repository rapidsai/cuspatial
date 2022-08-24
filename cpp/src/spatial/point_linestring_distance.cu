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

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/experimental/point_linestring_distance.cuh>
#include <cuspatial/experimental/type_utils.hpp>

#include <memory>
#include <type_traits>

namespace cuspatial {

namespace {

struct pairwise_point_linestring_distance_functor {
  template <typename T, CUDF_ENABLE_IF(std::is_floating_point_v<T>)>
  std::unique_ptr<cudf::column> operator()(
    cudf::column_view const& points_xy,
    cudf::device_span<cudf::size_type const> linestring_offsets,
    cudf::column_view const& linestring_points_xy,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    auto const num_pairs = static_cast<cudf::size_type>(points_xy.size() / 2);
    auto const num_linestring_points =
      static_cast<cudf::size_type>(linestring_points_xy.size() / 2);

    auto output = cudf::make_numeric_column(
      points_xy.type(), num_pairs, cudf::mask_state::UNALLOCATED, stream, mr);

    auto points_it = interleaved_iterator_to_vec_2d_iterator(points_xy.begin<T>());
    auto linestring_points_it =
      interleaved_iterator_to_vec_2d_iterator(linestring_points_xy.begin<T>());
    auto output_begin = output->mutable_view().begin<T>();

    pairwise_point_linestring_distance(points_it,
                                       points_it + num_pairs,
                                       linestring_offsets.begin(),
                                       linestring_points_it,
                                       linestring_points_it + num_linestring_points,
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
  cudf::column_view const& points_xy,
  cudf::device_span<cudf::size_type const> linestring_offsets,
  cudf::column_view const& linestring_points_xy,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  CUSPATIAL_EXPECTS(points_xy.size() % 2 == 0 && linestring_points_xy.size() % 2 == 0,
                    "Points array must contain even number of coordinates.");

  CUSPATIAL_EXPECTS(static_cast<cudf::device_span<const int>::size_type>(points_xy.size() + 1) ==
                      linestring_offsets.size(),
                    "Mismatch number of points and linestrings.");

  CUSPATIAL_EXPECTS(points_xy.type() == linestring_points_xy.type(),
                    "Points and linestring coordinates must have the same type.");

  CUSPATIAL_EXPECTS(!(points_xy.has_nulls() || linestring_points_xy.has_nulls()),
                    "All inputs must not have nulls.");

  if (linestring_offsets.size() == 0) return cudf::make_empty_column(points_xy.type());

  return cudf::type_dispatcher(points_xy.type(),
                               pairwise_point_linestring_distance_functor{},
                               points_xy,
                               linestring_offsets,
                               linestring_points_xy,
                               stream,
                               mr);
}

}  // namespace detail

std::unique_ptr<cudf::column> pairwise_point_linestring_distance(
  cudf::column_view const& points_xy,
  cudf::device_span<cudf::size_type const> linestring_offsets,
  cudf::column_view const& linestring_points_xy,
  rmm::mr::device_memory_resource* mr)
{
  return detail::pairwise_point_linestring_distance(
    points_xy, linestring_offsets, linestring_points_xy, rmm::cuda_stream_default, mr);
}

}  // namespace cuspatial
