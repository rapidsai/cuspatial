/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/cuda_stream_view.hpp>

#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/tuple.h>

#include <memory>
#include <utility>

namespace cuspatial {

namespace {

template <typename T>
struct point_to_square {
  T expansion_radius{0};
  inline __device__ thrust::tuple<T, T, T, T> operator()(thrust::tuple<T, T> const &point)
  {
    return thrust::make_tuple(thrust::get<0>(point) - expansion_radius,   // x
                              thrust::get<1>(point) - expansion_radius,   // y
                              thrust::get<0>(point) + expansion_radius,   // x
                              thrust::get<1>(point) + expansion_radius);  // y
  }
};

template <typename T>
std::unique_ptr<cudf::table> compute_polyline_bounding_boxes(cudf::column_view const &poly_offsets,
                                                             cudf::column_view const &x,
                                                             cudf::column_view const &y,
                                                             T expansion_radius,
                                                             rmm::cuda_stream_view stream,
                                                             rmm::mr::device_memory_resource *mr)
{
  auto num_polygons = poly_offsets.size();
  rmm::device_vector<int32_t> point_ids(x.size());

  // Scatter the polyline offsets into a list of point_ids for reduction
  thrust::scatter(rmm::exec_policy(stream)->on(stream.value()),
                  thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(0) + num_polygons,
                  poly_offsets.begin<int32_t>(),
                  point_ids.begin());

  thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream.value()),
                         point_ids.begin(),
                         point_ids.end(),
                         point_ids.begin(),
                         thrust::maximum<int32_t>());

  auto type = cudf::data_type{cudf::type_to_id<T>()};
  std::vector<std::unique_ptr<cudf::column>> cols{};
  cols.reserve(4);
  cols.push_back(
    cudf::make_numeric_column(type, num_polygons, cudf::mask_state::UNALLOCATED, stream, mr));
  cols.push_back(
    cudf::make_numeric_column(type, num_polygons, cudf::mask_state::UNALLOCATED, stream, mr));
  cols.push_back(
    cudf::make_numeric_column(type, num_polygons, cudf::mask_state::UNALLOCATED, stream, mr));
  cols.push_back(
    cudf::make_numeric_column(type, num_polygons, cudf::mask_state::UNALLOCATED, stream, mr));

  auto bboxes_iter =
    thrust::make_zip_iterator(thrust::make_tuple(cols.at(0)->mutable_view().begin<T>(),  // bbox_x1
                                                 cols.at(1)->mutable_view().begin<T>(),  // bbox_y1
                                                 cols.at(2)->mutable_view().begin<T>(),  // bbox_x2
                                                 cols.at(3)->mutable_view().begin<T>())  // bbox_y2
    );

  auto points_iter = thrust::make_zip_iterator(thrust::make_tuple(x.begin<T>(), y.begin<T>()));
  auto points_squared_iter =
    thrust::make_transform_iterator(points_iter, point_to_square<T>{expansion_radius});

  thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream.value()),
                        point_ids.begin(),
                        point_ids.end(),
                        points_squared_iter,
                        thrust::make_discard_iterator(),
                        bboxes_iter,
                        thrust::equal_to<int32_t>(),
                        [] __device__(auto const &a, auto const &b) {
                          T min_x_a, min_y_a, max_x_a, max_y_a;
                          T min_x_b, min_y_b, max_x_b, max_y_b;
                          thrust::tie(min_x_a, min_y_a, max_x_a, max_y_a) = a;
                          thrust::tie(min_x_b, min_y_b, max_x_b, max_y_b) = b;
                          return thrust::make_tuple(min(min_x_a, min_x_b),   // min_x
                                                    min(min_y_a, min_y_b),   // min_y
                                                    max(max_x_a, max_x_b),   // max_x
                                                    max(max_y_a, max_y_b));  // max_y
                        });

  return std::make_unique<cudf::table>(std::move(cols));
}

struct dispatch_compute_polyline_bounding_boxes {
  template <typename T, typename... Args>
  inline std::enable_if_t<!std::is_floating_point<T>::value, std::unique_ptr<cudf::table>>
  operator()(Args &&...)
  {
    CUSPATIAL_FAIL("Only floating-point types are supported");
  }

  template <typename T>
  inline std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<cudf::table>>
  operator()(cudf::column_view const &poly_offsets,
             cudf::column_view const &x,
             cudf::column_view const &y,
             double expansion_radius,
             rmm::cuda_stream_view stream,
             rmm::mr::device_memory_resource *mr)
  {
    return compute_polyline_bounding_boxes<T>(
      poly_offsets, x, y, static_cast<T>(expansion_radius), stream, mr);
  }
};

}  // namespace

namespace detail {

std::unique_ptr<cudf::table> polyline_bounding_boxes(cudf::column_view const &poly_offsets,
                                                     cudf::column_view const &x,
                                                     cudf::column_view const &y,
                                                     double expansion_radius,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::mr::device_memory_resource *mr)
{
  return cudf::type_dispatcher(x.type(),
                               dispatch_compute_polyline_bounding_boxes{},
                               poly_offsets,
                               x,
                               y,
                               expansion_radius,
                               rmm::cuda_stream_default,
                               mr);
}

}  // namespace detail

std::unique_ptr<cudf::table> polyline_bounding_boxes(cudf::column_view const &poly_offsets,
                                                     cudf::column_view const &x,
                                                     cudf::column_view const &y,
                                                     double expansion_radius,
                                                     rmm::mr::device_memory_resource *mr)
{
  CUSPATIAL_EXPECTS(x.type() == y.type(), "Data type mismatch");
  CUSPATIAL_EXPECTS(x.size() == y.size(), "x and y must be the same size");
  CUSPATIAL_EXPECTS(poly_offsets.type().id() == cudf::type_id::INT32, "Invalid poly_offsets type");
  CUSPATIAL_EXPECTS(expansion_radius >= 0, "expansion radius must be greater or equal than 0");
  CUSPATIAL_EXPECTS(x.size() >= 2 * poly_offsets.size(),
                    "all polylines must have at least 2 vertices");

  if (poly_offsets.is_empty() || x.is_empty() || y.is_empty()) {
    std::vector<std::unique_ptr<cudf::column>> cols{};
    cols.reserve(4);
    cols.push_back(cudf::empty_like(x));
    cols.push_back(cudf::empty_like(y));
    cols.push_back(cudf::empty_like(x));
    cols.push_back(cudf::empty_like(y));
    return std::make_unique<cudf::table>(std::move(cols));
  }
  return detail::polyline_bounding_boxes(
    poly_offsets, x, y, expansion_radius, rmm::cuda_stream_default, mr);
}

}  // namespace cuspatial
