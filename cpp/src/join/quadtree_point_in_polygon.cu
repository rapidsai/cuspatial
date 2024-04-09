/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cuspatial/detail/utility/validation.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/iterator_factory.cuh>
#include <cuspatial/range/multipoint_range.cuh>
#include <cuspatial/range/multipolygon_range.cuh>
#include <cuspatial/spatial_join.cuh>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

namespace cuspatial {
namespace detail {
namespace {
struct compute_quadtree_point_in_polygon {
  template <typename T, typename... Args>
  std::enable_if_t<!std::is_floating_point<T>::value, std::unique_ptr<cudf::table>> operator()(
    Args&&...)
  {
    CUDF_FAIL("Non-floating point operation is not supported");
  }

  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<cudf::table>> operator()(
    cudf::table_view const& poly_quad_pairs,
    cudf::table_view const& quadtree,
    cudf::column_view const& point_indices,
    cudf::column_view const& point_x,
    cudf::column_view const& point_y,
    cudf::column_view const& poly_offsets,
    cudf::column_view const& ring_offsets,
    cudf::column_view const& poly_points_x,
    cudf::column_view const& poly_points_y,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
  {
    auto poly_indices = poly_quad_pairs.column(0);
    auto quad_indices = poly_quad_pairs.column(1);

    auto quadtree_ref = point_quadtree_ref(quadtree.column(0).begin<uint32_t>(),  // keys
                                           quadtree.column(0).end<uint32_t>(),
                                           quadtree.column(1).begin<uint8_t>(),  // levels
                                           quadtree.column(2).begin<bool>(),     // is_internal_node
                                           quadtree.column(3).begin<uint32_t>(),   // lengths
                                           quadtree.column(4).begin<uint32_t>());  // offsets

    auto multipolygons =
      multipolygon_range(thrust::make_counting_iterator(0),
                         thrust::make_counting_iterator(poly_offsets.size()),
                         poly_offsets.begin<uint32_t>(),
                         poly_offsets.end<uint32_t>(),
                         ring_offsets.begin<uint32_t>(),
                         ring_offsets.end<uint32_t>(),
                         make_vec_2d_iterator(poly_points_x.begin<T>(), poly_points_y.begin<T>()),
                         make_vec_2d_iterator(poly_points_x.end<T>(), poly_points_y.end<T>()));

    auto [poly_idx, point_idx] =
      quadtree_point_in_polygon(poly_indices.begin<uint32_t>(),
                                poly_indices.end<uint32_t>(),
                                quad_indices.begin<uint32_t>(),
                                quadtree_ref,
                                point_indices.begin<uint32_t>(),
                                point_indices.end<uint32_t>(),
                                make_vec_2d_iterator(point_x.begin<T>(), point_y.begin<T>()),
                                multipolygons,
                                stream,
                                mr);

    // Allocate output columns for the number of pairs that intersected
    auto num_intersections = poly_idx.size();

    auto poly_idx_col  = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT32},
                                                       num_intersections,
                                                       poly_idx.release(),
                                                       rmm::device_buffer{},
                                                       0);
    auto point_idx_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT32},
                                                        num_intersections,
                                                        point_idx.release(),
                                                        rmm::device_buffer{},
                                                        0);

    std::vector<std::unique_ptr<cudf::column>> cols{};
    cols.reserve(2);
    cols.push_back(std::move(poly_idx_col));
    cols.push_back(std::move(point_idx_col));
    return std::make_unique<cudf::table>(std::move(cols));
  }
};

}  // namespace

std::unique_ptr<cudf::table> quadtree_point_in_polygon(cudf::table_view const& poly_quad_pairs,
                                                       cudf::table_view const& quadtree,
                                                       cudf::column_view const& point_indices,
                                                       cudf::column_view const& point_x,
                                                       cudf::column_view const& point_y,
                                                       cudf::column_view const& poly_offsets,
                                                       cudf::column_view const& ring_offsets,
                                                       cudf::column_view const& poly_points_x,
                                                       cudf::column_view const& poly_points_y,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  return cudf::type_dispatcher(point_x.type(),
                               compute_quadtree_point_in_polygon{},
                               poly_quad_pairs,
                               quadtree,
                               point_indices,
                               point_x,
                               point_y,
                               poly_offsets,
                               ring_offsets,
                               poly_points_x,
                               poly_points_y,
                               stream,
                               mr);
}

}  // namespace detail

std::unique_ptr<cudf::table> quadtree_point_in_polygon(cudf::table_view const& poly_quad_pairs,
                                                       cudf::table_view const& quadtree,
                                                       cudf::column_view const& point_indices,
                                                       cudf::column_view const& point_x,
                                                       cudf::column_view const& point_y,
                                                       cudf::column_view const& poly_offsets,
                                                       cudf::column_view const& ring_offsets,
                                                       cudf::column_view const& poly_points_x,
                                                       cudf::column_view const& poly_points_y,
                                                       rmm::device_async_resource_ref mr)
{
  CUSPATIAL_EXPECTS(poly_quad_pairs.num_columns() == 2,
                    "a quadrant-polygon table must have 2 columns");
  CUSPATIAL_EXPECTS(quadtree.num_columns() == 5, "a quadtree table must have 5 columns");
  CUSPATIAL_EXPECTS(point_indices.size() == point_x.size() && point_x.size() == point_y.size(),
                    "number of points must be the same for both x and y columns");
  CUSPATIAL_EXPECTS(poly_points_x.size() == poly_points_y.size(),
                    "numbers of vertices must be the same for both x and y columns");

  CUSPATIAL_EXPECTS(poly_points_x.type() == poly_points_y.type(),
                    "polygon columns must have the same data type");
  CUSPATIAL_EXPECTS(point_x.type() == point_y.type(), "point columns must have the same data type");
  CUSPATIAL_EXPECTS(point_x.type() == poly_points_x.type(),
                    "points and polygons must have the same data type");

  CUSPATIAL_EXPECTS(poly_offsets.type() == ring_offsets.type(),
                    "offset columns must have the same data type");

  if (poly_quad_pairs.num_rows() == 0 || quadtree.num_rows() == 0 || point_indices.size() == 0 ||
      poly_offsets.size() == 0) {
    std::vector<std::unique_ptr<cudf::column>> cols{};
    cols.reserve(2);
    cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::UINT32}));
    cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::UINT32}));
    return std::make_unique<cudf::table>(std::move(cols));
  }

  return detail::quadtree_point_in_polygon(poly_quad_pairs,
                                           quadtree,
                                           point_indices,
                                           point_x,
                                           point_y,
                                           poly_offsets,
                                           ring_offsets,
                                           poly_points_x,
                                           poly_points_y,
                                           rmm::cuda_stream_default,
                                           mr);
}

}  // namespace cuspatial
