/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
#include <cuspatial/experimental/bounding_box.cuh>
#include <cuspatial/experimental/iterator_factory.cuh>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/zip_iterator.h>

#include <type_traits>
#include <vector>

namespace cuspatial {

namespace {

struct dispatch_element {
  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<cudf::table>> operator()(
    cudf::size_type num_trajectories,
    cudf::column_view const& object_id,
    cudf::column_view const& x,
    cudf::column_view const& y,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    // Construct output columns
    auto type = cudf::data_type{cudf::type_to_id<T>()};
    std::vector<std::unique_ptr<cudf::column>> cols{};
    cols.reserve(4);
    // allocate bbox_x1 output column
    cols.push_back(
      cudf::make_numeric_column(type, num_trajectories, cudf::mask_state::UNALLOCATED, stream, mr));
    // allocate bbox_y1 output column
    cols.push_back(
      cudf::make_numeric_column(type, num_trajectories, cudf::mask_state::UNALLOCATED, stream, mr));
    // allocate bbox_x2 output column
    cols.push_back(
      cudf::make_numeric_column(type, num_trajectories, cudf::mask_state::UNALLOCATED, stream, mr));
    // allocate bbox_y2 output column
    cols.push_back(
      cudf::make_numeric_column(type, num_trajectories, cudf::mask_state::UNALLOCATED, stream, mr));

    auto points_begin = cuspatial::make_vec_2d_iterator(x.begin<T>(), y.begin<T>());

    auto bounding_boxes_begin =
      cuspatial::make_box_output_iterator(cols.at(0)->mutable_view().begin<T>(),
                                          cols.at(1)->mutable_view().begin<T>(),
                                          cols.at(2)->mutable_view().begin<T>(),
                                          cols.at(3)->mutable_view().begin<T>());

    point_bounding_boxes(object_id.begin<cudf::size_type>(),
                         object_id.end<cudf::size_type>(),
                         points_begin,
                         bounding_boxes_begin,
                         T{0},
                         stream);

    // check for errors
    CUSPATIAL_CHECK_CUDA(stream.value());

    return std::make_unique<cudf::table>(std::move(cols));
  }

  template <typename Element, typename... Args>
  std::enable_if_t<not std::is_floating_point<Element>::value, std::unique_ptr<cudf::table>>
  operator()(Args&&...)
  {
    CUSPATIAL_FAIL("X and Y must be floating point types");
  }
};

}  // namespace

namespace detail {
std::unique_ptr<cudf::table> trajectory_bounding_boxes(cudf::size_type num_trajectories,
                                                       cudf::column_view const& object_id,
                                                       cudf::column_view const& x,
                                                       cudf::column_view const& y,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::mr::device_memory_resource* mr)
{
  return cudf::type_dispatcher(
    x.type(), dispatch_element{}, num_trajectories, object_id, x, y, stream, mr);
}
}  // namespace detail

std::unique_ptr<cudf::table> trajectory_bounding_boxes(cudf::size_type num_trajectories,
                                                       cudf::column_view const& object_id,
                                                       cudf::column_view const& x,
                                                       cudf::column_view const& y,
                                                       rmm::mr::device_memory_resource* mr)
{
  CUSPATIAL_EXPECTS(object_id.size() == x.size() && x.size() == y.size(), "Data size mismatch");
  CUSPATIAL_EXPECTS(x.type().id() == y.type().id(), "Data type mismatch");
  CUSPATIAL_EXPECTS(object_id.type().id() == cudf::type_to_id<cudf::size_type>(),
                    "Invalid object_id type");
  CUSPATIAL_EXPECTS(!(x.has_nulls() || y.has_nulls() || object_id.has_nulls()),
                    "NULL support unimplemented");

  if (num_trajectories == 0 || object_id.is_empty() || x.is_empty() || y.is_empty()) {
    std::vector<std::unique_ptr<cudf::column>> cols{};
    cols.reserve(4);
    cols.push_back(cudf::empty_like(x));
    cols.push_back(cudf::empty_like(y));
    cols.push_back(cudf::empty_like(x));
    cols.push_back(cudf::empty_like(y));
    return std::make_unique<cudf::table>(std::move(cols));
  }

  return detail::trajectory_bounding_boxes(
    num_trajectories, object_id, x, y, rmm::cuda_stream_default, mr);
}

}  // namespace cuspatial
