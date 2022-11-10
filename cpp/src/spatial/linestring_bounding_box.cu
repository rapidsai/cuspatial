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
#include <cuspatial/experimental/bounding_boxes.cuh>
#include <cuspatial/experimental/iterator_factory.cuh>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/tuple.h>

#include <memory>
#include <utility>

namespace cuspatial {

namespace {

template <typename T>
std::unique_ptr<cudf::table> compute_linestring_bounding_boxes(
  cudf::column_view const& linestring_offsets,
  cudf::column_view const& x,
  cudf::column_view const& y,
  T expansion_radius,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  auto num_linestrings = linestring_offsets.size();
  rmm::device_vector<int32_t> point_ids(x.size());

  // Scatter the linestring offsets into a list of point_ids for reduction
  thrust::scatter(rmm::exec_policy(stream),
                  thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(0) + num_linestrings,
                  linestring_offsets.begin<int32_t>(),
                  point_ids.begin());

  thrust::inclusive_scan(rmm::exec_policy(stream),
                         point_ids.begin(),
                         point_ids.end(),
                         point_ids.begin(),
                         thrust::maximum<int32_t>());

  auto type = cudf::data_type{cudf::type_to_id<T>()};
  std::vector<std::unique_ptr<cudf::column>> cols{};
  cols.reserve(4);
  cols.push_back(
    cudf::make_numeric_column(type, num_linestrings, cudf::mask_state::UNALLOCATED, stream, mr));
  cols.push_back(
    cudf::make_numeric_column(type, num_linestrings, cudf::mask_state::UNALLOCATED, stream, mr));
  cols.push_back(
    cudf::make_numeric_column(type, num_linestrings, cudf::mask_state::UNALLOCATED, stream, mr));
  cols.push_back(
    cudf::make_numeric_column(type, num_linestrings, cudf::mask_state::UNALLOCATED, stream, mr));

  auto points_begin = cuspatial::make_vec_2d_iterator(x.begin<T>(), y.begin<T>());

  auto bbox_mins  = cuspatial::make_vec_2d_output_iterator(cols.at(0)->mutable_view().begin<T>(),
                                                          cols.at(1)->mutable_view().begin<T>());
  auto bbox_maxes = cuspatial::make_vec_2d_output_iterator(cols.at(2)->mutable_view().begin<T>(),
                                                           cols.at(3)->mutable_view().begin<T>());

  point_bounding_boxes(point_ids.begin(),
                       point_ids.end(),
                       points_begin,
                       thrust::make_zip_iterator(bbox_mins, bbox_maxes),
                       expansion_radius,
                       stream);

  return std::make_unique<cudf::table>(std::move(cols));
}

struct dispatch_compute_linestring_bounding_boxes {
  template <typename T, typename... Args>
  inline std::enable_if_t<!std::is_floating_point<T>::value, std::unique_ptr<cudf::table>>
  operator()(Args&&...)
  {
    CUSPATIAL_FAIL("Only floating-point types are supported");
  }

  template <typename T>
  inline std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<cudf::table>>
  operator()(cudf::column_view const& linestring_offsets,
             cudf::column_view const& x,
             cudf::column_view const& y,
             double expansion_radius,
             rmm::cuda_stream_view stream,
             rmm::mr::device_memory_resource* mr)
  {
    return compute_linestring_bounding_boxes<T>(
      linestring_offsets, x, y, static_cast<T>(expansion_radius), stream, mr);
  }
};

}  // namespace

namespace detail {

std::unique_ptr<cudf::table> linestring_bounding_boxes(cudf::column_view const& linestring_offsets,
                                                       cudf::column_view const& x,
                                                       cudf::column_view const& y,
                                                       double expansion_radius,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::mr::device_memory_resource* mr)
{
  return cudf::type_dispatcher(x.type(),
                               dispatch_compute_linestring_bounding_boxes{},
                               linestring_offsets,
                               x,
                               y,
                               expansion_radius,
                               rmm::cuda_stream_default,
                               mr);
}

}  // namespace detail

std::unique_ptr<cudf::table> linestring_bounding_boxes(cudf::column_view const& linestring_offsets,
                                                       cudf::column_view const& x,
                                                       cudf::column_view const& y,
                                                       double expansion_radius,
                                                       rmm::mr::device_memory_resource* mr)
{
  CUSPATIAL_EXPECTS(x.type() == y.type(), "Data type mismatch");
  CUSPATIAL_EXPECTS(x.size() == y.size(), "x and y must be the same size");
  CUSPATIAL_EXPECTS(linestring_offsets.type().id() == cudf::type_id::INT32,
                    "Invalid linestring_offsets type");
  CUSPATIAL_EXPECTS(expansion_radius >= 0, "expansion radius must be greater or equal than 0");
  CUSPATIAL_EXPECTS(x.size() >= 2 * linestring_offsets.size(),
                    "all linestrings must have at least 2 vertices");

  if (linestring_offsets.is_empty() || x.is_empty() || y.is_empty()) {
    std::vector<std::unique_ptr<cudf::column>> cols{};
    cols.reserve(4);
    cols.push_back(cudf::empty_like(x));
    cols.push_back(cudf::empty_like(y));
    cols.push_back(cudf::empty_like(x));
    cols.push_back(cudf::empty_like(y));
    return std::make_unique<cudf::table>(std::move(cols));
  }
  return detail::linestring_bounding_boxes(
    linestring_offsets, x, y, expansion_radius, rmm::cuda_stream_default, mr);
}

}  // namespace cuspatial
