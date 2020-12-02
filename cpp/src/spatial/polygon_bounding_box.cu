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
#include <cuspatial/polygon_bounding_box.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

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
#include <vector>

namespace cuspatial {

namespace {

template <typename T>
struct point_to_square {
  inline __device__ thrust::tuple<T, T, T, T> operator()(thrust::tuple<T, T> const &point)
  {
    return thrust::make_tuple(thrust::get<0>(point),   // x
                              thrust::get<1>(point),   // y
                              thrust::get<0>(point),   // x
                              thrust::get<1>(point));  // y
  }
};

template <typename T>
std::unique_ptr<cudf::table> compute_polygon_bounding_boxes(cudf::column_view const &poly_offsets,
                                                            cudf::column_view const &ring_offsets,
                                                            cudf::column_view const &x,
                                                            cudf::column_view const &y,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::mr::device_memory_resource *mr)
{
  auto num_polygons = poly_offsets.size();
  // Wrapped in an IEFE so `first_ring_offsets` is freed on return
  auto point_ids = [&]() {
    rmm::device_vector<int32_t> point_ids(x.size());
    rmm::device_vector<int32_t> first_ring_offsets(num_polygons);

    // Gather the first ring offset for each polygon
    thrust::gather(rmm::exec_policy(stream)->on(stream.value()),
                   poly_offsets.begin<int32_t>(),
                   poly_offsets.end<int32_t>(),
                   ring_offsets.begin<int32_t>(),
                   first_ring_offsets.begin());

    // Scatter the first ring offset into a list of point_ids for reduction
    thrust::scatter(rmm::exec_policy(stream)->on(stream.value()),
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(0) + num_polygons,
                    first_ring_offsets.begin(),
                    point_ids.begin());

    thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream.value()),
                           point_ids.begin(),
                           point_ids.end(),
                           point_ids.begin(),
                           thrust::maximum<int32_t>());

    return point_ids;
  }();

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
  auto points_squared_iter = thrust::make_transform_iterator(points_iter, point_to_square<T>{});

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

struct dispatch_compute_polygon_bounding_boxes {
  template <typename T, typename... Args>
  inline std::enable_if_t<!std::is_floating_point<T>::value, std::unique_ptr<cudf::table>>
  operator()(Args &&...)
  {
    CUSPATIAL_FAIL("Only floating-point types are supported");
  }

  template <typename T>
  inline std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<cudf::table>>
  operator()(cudf::column_view const &poly_offsets,
             cudf::column_view const &ring_offsets,
             cudf::column_view const &x,
             cudf::column_view const &y,
             rmm::cuda_stream_view stream,
             rmm::mr::device_memory_resource *mr)
  {
    return compute_polygon_bounding_boxes<T>(poly_offsets, ring_offsets, x, y, stream, mr);
  }
};

}  // namespace

namespace detail {

std::unique_ptr<cudf::table> polygon_bounding_boxes(cudf::column_view const &poly_offsets,
                                                    cudf::column_view const &ring_offsets,
                                                    cudf::column_view const &x,
                                                    cudf::column_view const &y,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::mr::device_memory_resource *mr)
{
  return cudf::type_dispatcher(x.type(),
                               dispatch_compute_polygon_bounding_boxes{},
                               poly_offsets,
                               ring_offsets,
                               x,
                               y,
                               stream,
                               mr);
}

}  // namespace detail

std::unique_ptr<cudf::table> polygon_bounding_boxes(cudf::column_view const &poly_offsets,
                                                    cudf::column_view const &ring_offsets,
                                                    cudf::column_view const &x,
                                                    cudf::column_view const &y,
                                                    rmm::mr::device_memory_resource *mr)
{
  CUSPATIAL_EXPECTS(ring_offsets.size() >= poly_offsets.size(),
                    "number of rings must be greater than or equal to the number of polygons");
  CUSPATIAL_EXPECTS(x.type() == y.type(), "Data type mismatch");
  CUSPATIAL_EXPECTS(poly_offsets.type().id() == cudf::type_id::INT32, "Invalid poly_offsets type");
  CUSPATIAL_EXPECTS(ring_offsets.type().id() == cudf::type_id::INT32, "Invalid ring_offsets type");
  CUSPATIAL_EXPECTS(x.size() == y.size(), "x and y must be the same size");
  CUSPATIAL_EXPECTS(x.size() >= 3 * ring_offsets.size(), "all rings must have at least 3 points");

  if (poly_offsets.is_empty() || ring_offsets.is_empty() || x.is_empty() || y.is_empty()) {
    std::vector<std::unique_ptr<cudf::column>> cols{};
    cols.reserve(4);
    cols.push_back(cudf::empty_like(x));
    cols.push_back(cudf::empty_like(y));
    cols.push_back(cudf::empty_like(x));
    cols.push_back(cudf::empty_like(y));
    return std::make_unique<cudf::table>(std::move(cols));
  }

  return detail::polygon_bounding_boxes(
    poly_offsets, ring_offsets, x, y, rmm::cuda_stream_default, mr);
}

}  // namespace cuspatial
