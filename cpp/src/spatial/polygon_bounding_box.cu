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

#include <cuspatial/experimental/iterator_factory.cuh>
#include <cuspatial/experimental/polygon_bounding_boxes.cuh>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/zip_iterator.h>

#include <memory>
#include <vector>

namespace cuspatial {
namespace {

template <typename T>
std::unique_ptr<cudf::table> compute_polygon_bounding_boxes(cudf::column_view const& poly_offsets,
                                                            cudf::column_view const& ring_offsets,
                                                            cudf::column_view const& x,
                                                            cudf::column_view const& y,
                                                            T expansion_radius,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::mr::device_memory_resource* mr)
{
  auto num_polygons = poly_offsets.size() > 0 ? poly_offsets.size() - 1 : 0;

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

  auto vertices_begin = cuspatial::make_vec_2d_iterator(x.begin<T>(), y.begin<T>());

  auto bounding_boxes_begin =
    cuspatial::make_box_output_iterator(cols.at(0)->mutable_view().begin<T>(),
                                        cols.at(1)->mutable_view().begin<T>(),
                                        cols.at(2)->mutable_view().begin<T>(),
                                        cols.at(3)->mutable_view().begin<T>());

  cuspatial::polygon_bounding_boxes(poly_offsets.begin<cudf::size_type>(),
                                    poly_offsets.end<cudf::size_type>(),
                                    ring_offsets.begin<cudf::size_type>(),
                                    ring_offsets.end<cudf::size_type>(),
                                    vertices_begin,
                                    vertices_begin + x.size(),
                                    bounding_boxes_begin,
                                    expansion_radius,
                                    stream);

  return std::make_unique<cudf::table>(std::move(cols));
}

struct dispatch_compute_polygon_bounding_boxes {
  template <typename T, typename... Args>
  inline std::enable_if_t<!std::is_floating_point<T>::value, std::unique_ptr<cudf::table>>
  operator()(Args&&...)
  {
    CUSPATIAL_FAIL("Only floating-point types are supported");
  }

  template <typename T>
  inline std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<cudf::table>>
  operator()(cudf::column_view const& poly_offsets,
             cudf::column_view const& ring_offsets,
             cudf::column_view const& x,
             cudf::column_view const& y,
             T expansion_radius,
             rmm::cuda_stream_view stream,
             rmm::mr::device_memory_resource* mr)
  {
    return compute_polygon_bounding_boxes<T>(
      poly_offsets, ring_offsets, x, y, expansion_radius, stream, mr);
  }
};

}  // namespace

namespace detail {

std::unique_ptr<cudf::table> polygon_bounding_boxes(cudf::column_view const& poly_offsets,
                                                    cudf::column_view const& ring_offsets,
                                                    cudf::column_view const& x,
                                                    cudf::column_view const& y,
                                                    double expansion_radius,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::mr::device_memory_resource* mr)
{
  return cudf::type_dispatcher(x.type(),
                               dispatch_compute_polygon_bounding_boxes{},
                               poly_offsets,
                               ring_offsets,
                               x,
                               y,
                               expansion_radius,
                               stream,
                               mr);
}

}  // namespace detail

std::unique_ptr<cudf::table> polygon_bounding_boxes(cudf::column_view const& poly_offsets,
                                                    cudf::column_view const& ring_offsets,
                                                    cudf::column_view const& x,
                                                    cudf::column_view const& y,
                                                    double expansion_radius,
                                                    rmm::mr::device_memory_resource* mr)
{
  auto num_polys = poly_offsets.size() > 0 ? poly_offsets.size() - 1 : 0;
  auto num_rings = ring_offsets.size() > 0 ? ring_offsets.size() - 1 : 0;

  CUSPATIAL_EXPECTS(num_rings >= num_polys,
                    "number of rings must be greater than or equal to the number of polygons");
  CUSPATIAL_EXPECTS(x.type() == y.type(), "Data type mismatch");
  CUSPATIAL_EXPECTS(poly_offsets.type().id() == cudf::type_id::INT32, "Invalid poly_offsets type");
  CUSPATIAL_EXPECTS(ring_offsets.type().id() == cudf::type_id::INT32, "Invalid ring_offsets type");
  CUSPATIAL_EXPECTS(x.size() == y.size(), "x and y must be the same size");
  CUSPATIAL_EXPECTS(x.size() >= 3 * num_rings, "all rings must have at least 3 points");

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
    poly_offsets, ring_offsets, x, y, expansion_radius, rmm::cuda_stream_default, mr);
}

}  // namespace cuspatial
