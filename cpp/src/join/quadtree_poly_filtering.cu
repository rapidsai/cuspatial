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
#include "join/detail/traversal.cuh"

namespace cuspatial {

namespace detail {

namespace {

template <typename T>
inline std::unique_ptr<cudf::table> join_quadtree_and_bboxes(cudf::table_view const &quadtree,
                                                             cudf::table_view const &poly_bbox,
                                                             T x_min,
                                                             T y_min,
                                                             T x_max,
                                                             T y_max,
                                                             T scale,
                                                             cudf::size_type max_depth,
                                                             rmm::mr::device_memory_resource *mr,
                                                             cudaStream_t stream)
{
  auto const node_levels  = quadtree.column(1);  // uint8_t
  auto const node_counts  = quadtree.column(3);  // uint32_t
  auto const node_offsets = quadtree.column(4);  // uint32_t

  auto num_polys = poly_bbox.num_rows();

  // count the number of top-level nodes to begin with
  // this number could be provided explicitly, but count_if should be fast enough
  auto num_top_level_children = thrust::count_if(rmm::exec_policy(stream)->on(stream),
                                                 node_levels.begin<uint8_t>(),
                                                 node_levels.end<uint8_t>(),
                                                 thrust::placeholders::_1 == 0);

  auto num_pairs = num_top_level_children * num_polys;

  rmm::device_uvector<uint8_t> quad_types(num_pairs, stream);   // d_type_temp
  rmm::device_uvector<uint8_t> quad_levels(num_pairs, stream);  // d_lev_temp
  rmm::device_uvector<uint32_t> quad_nodes(num_pairs, stream);  // d_quad_idx_temp
  rmm::device_uvector<uint32_t> quad_polys(num_pairs, stream);  // d_poly_idx_temp

  rmm::device_uvector<uint8_t> leaf_types(num_pairs, stream);   // d_type_out
  rmm::device_uvector<uint8_t> leaf_levels(num_pairs, stream);  // d_lev_out
  rmm::device_uvector<uint32_t> leaf_nodes(num_pairs, stream);  // d_node_idx_out
  rmm::device_uvector<uint32_t> leaf_polys(num_pairs, stream);  // d_poly_idx_out

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

  auto node_pairs = make_zip_iterator(
    quad_types.begin(), quad_levels.begin(), quad_nodes.begin(), quad_polys.begin());

  auto leaf_pairs = make_zip_iterator(
    leaf_types.begin(), leaf_levels.begin(), leaf_nodes.begin(), leaf_polys.begin());

  cudf::size_type num_leaves{0};
  cudf::size_type num_parents{0};
  cudf::size_type num_results{0};

  for (cudf::size_type i{0}; i < max_depth; ++i) {
    // Resize output device vectors and update the corresponding pointers. The next level will
    // expand out to no more than `num_parents * 4` pairs, since a parent quadrant can have no more
    // than 4 children.
    size_t max_num_results = num_results + num_parents * 4;

    if (max_num_results > leaf_types.capacity()) {
      // grow preallocated output sizes in multiples of the current capacity
      auto new_size = leaf_types.capacity() * ((max_num_results / leaf_types.capacity()) + 1);
      leaf_types.resize(new_size, stream);
      leaf_levels.resize(new_size, stream);
      leaf_nodes.resize(new_size, stream);
      leaf_polys.resize(new_size, stream);
      leaf_pairs = make_zip_iterator(
        leaf_types.begin(), leaf_levels.begin(), leaf_nodes.begin(), leaf_polys.begin());
    }

    if (i == 0) {
      // pair up all the top level quadrants and polygons first
      auto counter      = thrust::make_counting_iterator(0);
      auto node_indices = thrust::make_transform_iterator(
        counter,
        [num_top_level_children] __device__(auto const i) { return i % num_top_level_children; });
      auto poly_indices = thrust::make_transform_iterator(
        counter,
        [num_top_level_children] __device__(auto const i) { return i / num_top_level_children; });

      std::tie(num_parents, num_leaves) = find_intersections(quadtree,
                                                             poly_bbox,
                                                             node_indices,
                                                             poly_indices,
                                                             node_pairs,
                                                             leaf_pairs + num_results,
                                                             num_pairs,
                                                             x_min,
                                                             y_min,
                                                             scale,
                                                             max_depth,
                                                             stream);
    } else {
      std::tie(num_parents, num_leaves) = find_intersections(quadtree,
                                                             poly_bbox,
                                                             quad_nodes.begin(),
                                                             quad_polys.begin(),
                                                             node_pairs,
                                                             leaf_pairs + num_results,
                                                             num_pairs,
                                                             x_min,
                                                             y_min,
                                                             scale,
                                                             max_depth,
                                                             stream);
    }

    num_results += num_leaves;

    // stop descending if no parent quadrants left to expand
    if (num_parents == 0) break;

    quad_types.shrink_to_fit(stream);
    quad_levels.shrink_to_fit(stream);
    quad_nodes.shrink_to_fit(stream);
    quad_polys.shrink_to_fit(stream);

    auto child_counts = thrust::make_permutation_iterator(node_counts.begin<uint32_t>(),
                                                          quad_nodes.begin());  // d_quad_nchild

    auto next_level = descend_quadtree(child_counts,
                                       node_offsets.begin<uint32_t>(),
                                       num_parents,
                                       quad_types,
                                       quad_levels,
                                       quad_nodes,
                                       quad_polys,
                                       stream);

    num_pairs = std::get<0>(next_level);
    // update node_pairs iterator to get ready for next level iteration
    quad_types  = std::move(std::get<1>(next_level));
    quad_levels = std::move(std::get<2>(next_level));
    quad_nodes  = std::move(std::get<3>(next_level));
    quad_polys  = std::move(std::get<4>(next_level));
    node_pairs  = make_zip_iterator(
      quad_types.begin(), quad_levels.begin(), quad_nodes.begin(), quad_polys.begin());
  }

  std::vector<std::unique_ptr<cudf::column>> cols{};
  cols.reserve(2);
  cols.push_back(make_fixed_width_column<int32_t>(num_results, stream, mr));
  cols.push_back(make_fixed_width_column<int32_t>(num_results, stream, mr));

  thrust::copy(rmm::exec_policy(stream)->on(stream),
               leaf_polys.begin(),
               leaf_polys.begin() + num_results,
               cols.at(0)->mutable_view().begin<uint32_t>());

  thrust::copy(rmm::exec_policy(stream)->on(stream),
               leaf_nodes.begin(),
               leaf_nodes.begin() + num_results,
               cols.at(1)->mutable_view().begin<uint32_t>());

  return std::make_unique<cudf::table>(std::move(cols));
}

struct dispatch_quadtree_bounding_box_join {
  template <typename T, std::enable_if_t<std::is_floating_point<T>::value> * = nullptr>
  inline std::unique_ptr<cudf::table> operator()(cudf::table_view const &quadtree,
                                                 cudf::table_view const &poly_bbox,
                                                 double x_min,
                                                 double y_min,
                                                 double x_max,
                                                 double y_max,
                                                 double scale,
                                                 cudf::size_type max_depth,
                                                 rmm::mr::device_memory_resource *mr,
                                                 cudaStream_t stream)
  {
    return join_quadtree_and_bboxes<T>(quadtree,
                                       poly_bbox,
                                       static_cast<T>(x_min),
                                       static_cast<T>(y_min),
                                       static_cast<T>(x_max),
                                       static_cast<T>(y_max),
                                       static_cast<T>(scale),
                                       max_depth,
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
                                            double y_min,
                                            double x_max,
                                            double y_max,
                                            double scale,
                                            cudf::size_type max_depth,
                                            rmm::mr::device_memory_resource *mr,
                                            cudaStream_t stream)
{
  return cudf::type_dispatcher(poly_bbox.column(0).type(),
                               dispatch_quadtree_bounding_box_join{},
                               quadtree,
                               poly_bbox,
                               x_min,
                               y_min,
                               x_max,
                               y_max,
                               scale,
                               max_depth,
                               mr,
                               stream);
}

}  // namespace detail

std::unique_ptr<cudf::table> quad_bbox_join(cudf::table_view const &quadtree,
                                            cudf::table_view const &poly_bbox,
                                            double x_min,
                                            double y_min,
                                            double x_max,
                                            double y_max,
                                            double scale,
                                            cudf::size_type max_depth,
                                            rmm::mr::device_memory_resource *mr)
{
  CUSPATIAL_EXPECTS(quadtree.num_columns() == 5, "quadtree table must have 5 columns");
  CUSPATIAL_EXPECTS(poly_bbox.num_columns() == 4, "polygon bbox table must have 4 columns");
  CUSPATIAL_EXPECTS(x_min < x_max && y_min < y_max,
                    "invalid bounding box (x_min,y_min,x_max,y_max)");
  CUSPATIAL_EXPECTS(scale > 0, "scale must be positive");
  CUSPATIAL_EXPECTS(max_depth > 0 && max_depth < 16, "maximum of levels might be in [0,16)");

  if (quadtree.num_rows() == 0 || poly_bbox.num_rows() == 0) {
    std::vector<std::unique_ptr<cudf::column>> cols{};
    cols.reserve(2);
    cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::INT32}));
    cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::INT32}));
    return std::make_unique<cudf::table>(std::move(cols));
  }

  return detail::quad_bbox_join(
    quadtree, poly_bbox, x_min, y_min, x_max, y_max, scale, max_depth, mr, cudaStream_t{0});
}

}  // namespace cuspatial
