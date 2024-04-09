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
#include <cuspatial/range/multilinestring_range.cuh>
#include <cuspatial/spatial_join.cuh>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <limits>
#include <memory>

namespace cuspatial {
namespace detail {

struct compute_quadtree_point_to_nearest_linestring {
  template <typename T, typename... Args>
  std::enable_if_t<!std::is_floating_point<T>::value, std::unique_ptr<cudf::table>> operator()(
    Args&&...)
  {
    CUDF_FAIL("Non-floating point operation is not supported");
  }

  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<cudf::table>> operator()(
    cudf::table_view const& linestring_quad_pairs,
    cudf::table_view const& quadtree,
    cudf::column_view const& point_indices,
    cudf::column_view const& point_x,
    cudf::column_view const& point_y,
    cudf::column_view const& linestring_offsets,
    cudf::column_view const& linestring_points_x,
    cudf::column_view const& linestring_points_y,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
  {
    auto linestring_indices = linestring_quad_pairs.column(0);
    auto quad_indices       = linestring_quad_pairs.column(1);

    auto quadtree_ref = point_quadtree_ref(quadtree.column(0).begin<uint32_t>(),  // keys
                                           quadtree.column(0).end<uint32_t>(),
                                           quadtree.column(1).begin<uint8_t>(),  // levels
                                           quadtree.column(2).begin<bool>(),     // is_internal_node
                                           quadtree.column(3).begin<uint32_t>(),   // lengths
                                           quadtree.column(4).begin<uint32_t>());  // offsets

    auto linestrings = multilinestring_range(
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(linestring_offsets.size()),
      linestring_offsets.begin<uint32_t>(),
      linestring_offsets.end<uint32_t>(),
      make_vec_2d_iterator(linestring_points_x.begin<T>(), linestring_points_y.begin<T>()),
      make_vec_2d_iterator(linestring_points_x.end<T>(), linestring_points_y.end<T>()));

    auto [point_idxs, linestring_idxs, distances] = cuspatial::quadtree_point_to_nearest_linestring(
      linestring_indices.begin<uint32_t>(),
      linestring_indices.end<uint32_t>(),
      quad_indices.begin<uint32_t>(),
      quadtree_ref,
      point_indices.begin<uint32_t>(),
      point_indices.end<uint32_t>(),
      make_vec_2d_iterator(point_x.begin<T>(), point_y.begin<T>()),
      linestrings,
      stream,
      mr);

    auto num_distances = distances.size();

    auto point_idx_col      = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT32},
                                                        num_distances,
                                                        point_idxs.release(),
                                                        rmm::device_buffer{},
                                                        0);
    auto linestring_idx_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT32},
                                                             num_distances,
                                                             linestring_idxs.release(),
                                                             rmm::device_buffer{},
                                                             0);
    auto distance_col       = std::make_unique<cudf::column>(
      point_x.type(), num_distances, distances.release(), rmm::device_buffer{}, 0);

    std::vector<std::unique_ptr<cudf::column>> cols{};
    cols.reserve(3);
    cols.emplace_back(std::move(point_idx_col));
    cols.emplace_back(std::move(linestring_idx_col));
    cols.emplace_back(std::move(distance_col));
    return std::make_unique<cudf::table>(std::move(cols));
  }
};

std::unique_ptr<cudf::table> quadtree_point_to_nearest_linestring(
  cudf::table_view const& linestring_quad_pairs,
  cudf::table_view const& quadtree,
  cudf::column_view const& point_indices,
  cudf::column_view const& point_x,
  cudf::column_view const& point_y,
  cudf::column_view const& linestring_offsets,
  cudf::column_view const& linestring_points_x,
  cudf::column_view const& linestring_points_y,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  return cudf::type_dispatcher(point_x.type(),
                               compute_quadtree_point_to_nearest_linestring{},
                               linestring_quad_pairs,
                               quadtree,
                               point_indices,
                               point_x,
                               point_y,
                               linestring_offsets,
                               linestring_points_x,
                               linestring_points_y,
                               stream,
                               mr);
}

}  // namespace detail

std::unique_ptr<cudf::table> quadtree_point_to_nearest_linestring(
  cudf::table_view const& linestring_quad_pairs,
  cudf::table_view const& quadtree,
  cudf::column_view const& point_indices,
  cudf::column_view const& point_x,
  cudf::column_view const& point_y,
  cudf::column_view const& linestring_offsets,
  cudf::column_view const& linestring_points_x,
  cudf::column_view const& linestring_points_y,
  rmm::device_async_resource_ref mr)
{
  CUSPATIAL_EXPECTS(linestring_quad_pairs.num_columns() == 2,
                    "a quadrant-linestring table must have 2 columns");
  CUSPATIAL_EXPECTS(quadtree.num_columns() == 5, "a quadtree table must have 5 columns");

  CUSPATIAL_EXPECTS(point_indices.size() == point_x.size() && point_x.size() == point_y.size(),
                    "number of points must be the same for both x and y columns");
  CUSPATIAL_EXPECTS(linestring_points_x.size() == linestring_points_y.size(),
                    "numbers of vertices must be the same for both x and y columns");
  CUSPATIAL_EXPECTS(linestring_points_x.type() == linestring_points_y.type(),
                    "linestring columns must have the same data type");
  CUSPATIAL_EXPECTS(point_x.type() == point_y.type(), "point columns must have the same data type");
  CUSPATIAL_EXPECTS(point_x.type() == linestring_points_x.type(),
                    "points and linestrings must have the same data type");

  if (linestring_quad_pairs.num_rows() == 0 || quadtree.num_rows() == 0 ||
      point_indices.size() == 0 || linestring_offsets.size() == 0) {
    std::vector<std::unique_ptr<cudf::column>> cols{};
    cols.reserve(3);
    cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::UINT32}));
    cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::UINT32}));
    cols.push_back(cudf::make_empty_column(point_x.type()));
    return std::make_unique<cudf::table>(std::move(cols));
  }

  return detail::quadtree_point_to_nearest_linestring(linestring_quad_pairs,
                                                      quadtree,
                                                      point_indices,
                                                      point_x,
                                                      point_y,
                                                      linestring_offsets,
                                                      linestring_points_x,
                                                      linestring_points_y,
                                                      rmm::cuda_stream_default,
                                                      mr);
}

}  // namespace cuspatial
