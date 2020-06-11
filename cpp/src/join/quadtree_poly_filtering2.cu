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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuspatial/error.hpp>
#include <cuspatial/spatial_join.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_uvector.hpp>

#include <tuple>

#include "join/detail/intersection.cuh"

namespace cuspatial {

namespace detail {

namespace {

template <typename T>
inline std::unique_ptr<cudf::table> join_quadtree_and_bboxes(cudf::table_view const &quadtree,
                                                             cudf::table_view const &poly_bbox,
                                                             T x_min,
                                                             T x_max,
                                                             T y_min,
                                                             T y_max,
                                                             T scale,
                                                             cudf::size_type max_depth,
                                                             cudf::size_type min_size,
                                                             rmm::mr::device_memory_resource *mr,
                                                             cudaStream_t stream)
{
  auto const keys       = quadtree.column(0);   // uint32_t
  auto const levels     = quadtree.column(1);   // uint8_t
  auto const is_quad    = quadtree.column(2);   // bool
  auto const lengths    = quadtree.column(3);   // uint32_t
  auto const offsets    = quadtree.column(4);   // uint32_t
  auto const poly_x_min = poly_bbox.column(0);  // T
  auto const poly_y_min = poly_bbox.column(1);  // T
  auto const poly_x_max = poly_bbox.column(2);  // T
  auto const poly_y_max = poly_bbox.column(3);  // T

  auto num_nodes = quadtree.num_rows();
  auto num_polys = poly_bbox.num_rows();

  // count the number of top-level nodes to begin with
  // this number could be provided explicitly, but count_if should be fast enough
  auto num_top_level_children = thrust::count_if(rmm::exec_policy(stream)->on(stream),
                                                 levels.begin<uint8_t>(),
                                                 levels.end<uint8_t>(),
                                                 thrust::placeholders::_1 == 0);

  auto num_pairs = num_top_level_children * num_polys;

  // The matched quadrant-polygon pairs are dynamic and can not be pre-allocated in a fixed manner.
  // Relevant arrays are resized accordingly for memory efficiency.
  //
  // `d_lev_out`, `d_type_out`, `d_poly_idx_out`, `d_node_idx_out` are for outputs for matched pairs
  // with an initial capcity of `init_len`.
  //
  // `d_lev_increased`, `d_type_increased`, `d_poly_idx_increased`, `d_node_idx_increased` are for
  // resized storage for outputs, condering the maximum number of possible matched pairs at the next
  // level. The *_increased arrays are only resized as necessary.
  //
  // `d_lev_temp`, `d_type_temp`, `d_poly_idx_temp`, `d_quad_idx_temp` are for temporal stroage at a
  // level.
  //
  // `d_lev_expanded`, `d_type_expanded`, `d_poly_idx_expanded`, `d_node_idx_expanded` are for
  // expanded stroage at the next level. Their size is computed precisely by retriving the numbers
  // of child nodes for all non-leaf quadrants.

  // pair up all top level quadrants and all polygons
  auto top_level_intersections = find_top_level_intersections(
    // nodes_iter
    make_zip_iterator(keys.begin<uint32_t>(), levels.begin<uint8_t>(), is_quad.begin<bool>()),
    // polys_iter
    make_zip_iterator(
      poly_x_min.begin<T>(), poly_y_min.begin<T>(), poly_x_max.begin<T>(), poly_y_max.begin<T>()),
    num_pairs,
    num_nodes,
    x_min,
    y_min,
    scale,
    max_depth,
    stream);

  auto num_leaf_pairs    = std::get<0>(top_level_intersections);
  auto &out_types        = std::get<1>(top_level_intersections);  // d_type_out
  auto &out_levels       = std::get<2>(top_level_intersections);  // d_lev_out
  auto &out_poly_indices = std::get<3>(top_level_intersections);  // d_poly_idx_out
  auto &out_node_indices = std::get<4>(top_level_intersections);  // d_node_idx_out
  auto num_quad_pairs    = std::get<5>(top_level_intersections);
  auto &tmp_types        = std::get<6>(top_level_intersections);  // d_type_temp
  auto &tmp_levels       = std::get<7>(top_level_intersections);  // d_lev_temp
  auto &tmp_poly_indices = std::get<8>(top_level_intersections);  // d_poly_idx_temp
  auto &tmp_node_indices = std::get<9>(top_level_intersections);  // d_quad_idx_temp

  // cudf::size_type out_node_pos{num_leaf_pairs};
  cudf::size_type output_nodes_pos{num_leaf_pairs};
  auto counting_iter = thrust::make_counting_iterator(0);

  auto out_pairs_iter = make_zip_iterator(
    out_types.begin(), out_levels.begin(), out_poly_indices.begin(), out_node_indices.begin());

  rmm::device_uvector<uint32_t> child_offsets(num_quad_pairs, stream);
  rmm::device_uvector<uint32_t> parent_indices(num_quad_pairs, stream);

  for (cudf::size_type i{1}; i < max_depth; ++i) {
    //
    auto lengths_iter =
      thrust::make_permutation_iterator(lengths.begin<uint32_t>(), tmp_node_indices.begin());

    // compute the total number of child nodes
    num_pairs = thrust::reduce(
      rmm::exec_policy(stream)->on(stream), lengths_iter, lengths_iter + num_quad_pairs);

    tmp_types.resize(num_pairs, stream);
    tmp_levels.resize(num_pairs, stream);
    tmp_poly_indices.resize(num_pairs, stream);
    tmp_node_indices.resize(num_pairs, stream);

    // exclusive scan on the number of child nodes to compute the offsets
    thrust::exclusive_scan(rmm::exec_policy(stream)->on(stream),
                           lengths_iter,
                           lengths_iter + num_quad_pairs,
                           child_offsets.begin());

    // use the offset as the map to scatter sequential numbers 0..num_nonleaf_pair to d_expand_pos
    thrust::scatter(rmm::exec_policy(stream)->on(stream),
                    counting_iter,
                    counting_iter + num_quad_pairs,
                    child_offsets.begin(),
                    parent_indices.begin());

    // inclusive scan with maximum functor to fill the empty elements with their left-most non-empty
    // elements. `parent_offsets` is now a full array of the the sequence index of each quadrant's
    // parent
    thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream),
                           parent_indices.begin(),
                           parent_indices.begin() + num_pairs,
                           parent_indices.begin(),
                           thrust::maximum<uint32_t>());

    //
    // child_lengths.resize(num_quad_pairs, stream);
  }

  CUSPATIAL_FAIL("Unimplemented");
}

struct dispatch_quadtree_bounding_box_join {
  template <typename T, std::enable_if_t<std::is_floating_point<T>::value> * = nullptr>
  inline std::unique_ptr<cudf::table> operator()(cudf::table_view const &quadtree,
                                                 cudf::table_view const &poly_bbox,
                                                 double x_min,
                                                 double x_max,
                                                 double y_min,
                                                 double y_max,
                                                 double scale,
                                                 uint32_t max_depth,
                                                 uint32_t min_size,
                                                 rmm::mr::device_memory_resource *mr,
                                                 cudaStream_t stream)
  {
    return join_quadtree_and_bboxes<T>(quadtree,
                                       poly_bbox,
                                       static_cast<T>(x_min),
                                       static_cast<T>(x_max),
                                       static_cast<T>(y_min),
                                       static_cast<T>(y_max),
                                       static_cast<T>(scale),
                                       max_depth,
                                       min_size,
                                       mr,
                                       stream);
  }
  template <typename T,
            std::enable_if_t<!std::is_floating_point<T>::value> * = nullptr,
            typename... Args>
  inline std::unique_ptr<cudf::table> operator()(Args &&...)
  {
    CUSPATIAL_FAIL("Only floating-point types are supported");
  }
};
}  // namespace

std::unique_ptr<cudf::table> quad_bbox_join(cudf::table_view const &quadtree,
                                            cudf::table_view const &poly_bbox,
                                            double x_min,
                                            double x_max,
                                            double y_min,
                                            double y_max,
                                            double scale,
                                            cudf::size_type max_depth,
                                            cudf::size_type min_size,
                                            rmm::mr::device_memory_resource *mr,
                                            cudaStream_t stream)
{
  return cudf::type_dispatcher(poly_bbox.column(0).type(),
                               dispatch_quadtree_bounding_box_join{},
                               quadtree,
                               poly_bbox,
                               x_min,
                               x_max,
                               y_min,
                               y_max,
                               scale,
                               max_depth,
                               min_size,
                               mr,
                               stream);
}

}  // namespace detail

std::unique_ptr<cudf::table> quad_bbox_join(cudf::table_view const &quadtree,
                                            cudf::table_view const &poly_bbox,
                                            double x_min,
                                            double x_max,
                                            double y_min,
                                            double y_max,
                                            double scale,
                                            cudf::size_type max_depth,
                                            cudf::size_type min_size,
                                            rmm::mr::device_memory_resource *mr)
{
  CUSPATIAL_EXPECTS(quadtree.num_columns() == 5, "quadtree table must have 5 columns");
  CUSPATIAL_EXPECTS(poly_bbox.num_columns() == 4, "polygon bbox table must have 4 columns");
  CUSPATIAL_EXPECTS(x_min < x_max && y_min < y_max,
                    "invalid bounding box (x_min,y_min,x_max,y_max)");
  CUSPATIAL_EXPECTS(scale > 0, "scale must be positive");
  CUSPATIAL_EXPECTS(max_depth > 0 && max_depth < 16, "maximum of levels might be in [0,16)");
  CUSPATIAL_EXPECTS(min_size > 0,
                    "minimum number of points for a non-leaf node must be larger than zero");

  if (quadtree.num_rows() == 0 || poly_bbox.num_rows() == 0) {
    std::vector<std::unique_ptr<cudf::column>> cols{};
    cols.reserve(2);
    cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::INT32}));
    cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::INT32}));
    return std::make_unique<cudf::table>(std::move(cols));
  }

  return detail::quad_bbox_join(quadtree,
                                poly_bbox,
                                x_min,
                                x_max,
                                y_min,
                                y_max,
                                scale,
                                max_depth,
                                min_size,
                                mr,
                                cudaStream_t{0});
}

}  // namespace cuspatial
