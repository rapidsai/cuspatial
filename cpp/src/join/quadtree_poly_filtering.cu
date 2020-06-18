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
                                                             T x_max,
                                                             T y_min,
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

  // Count the number of top-level nodes to start.
  // This could be provided explicitly, but count_if should be fast enough.
  auto num_top_level_children = thrust::count_if(rmm::exec_policy(stream)->on(stream),
                                                 node_levels.begin<uint8_t>(),
                                                 node_levels.end<uint8_t>(),
                                                 thrust::placeholders::_1 == 0);

  auto num_pairs = num_top_level_children * num_polys;

  // The found poly-quad pairs are dynamic and can not be pre-allocated.
  // Relevant arrays are resized accordingly for memory efficiency.

  // Vectors for intermediate poly and node indices at each level
  rmm::device_uvector<uint8_t> tmp_types(num_pairs, stream);
  rmm::device_uvector<uint8_t> tmp_levels(num_pairs, stream);
  rmm::device_uvector<uint32_t> tmp_node_idxs(num_pairs, stream);
  rmm::device_uvector<uint32_t> tmp_poly_idxs(num_pairs, stream);

  // Vectors for found pairs of poly and leaf node indices
  rmm::device_uvector<uint8_t> out_types(num_pairs, stream);
  rmm::device_uvector<uint8_t> out_levels(num_pairs, stream);
  rmm::device_uvector<uint32_t> out_node_idxs(num_pairs, stream);
  rmm::device_uvector<uint32_t> out_poly_idxs(num_pairs, stream);

  // A zip iterator for the intermediate intersections or parent quadrants found during traversal
  auto tmp_pairs = make_zip_iterator(
    tmp_types.begin(), tmp_levels.begin(), tmp_node_idxs.begin(), tmp_poly_idxs.begin());

  // A zip iterator for the intersecting poly and quad indicies found
  auto out_pairs = make_zip_iterator(
    out_types.begin(), out_levels.begin(), out_node_idxs.begin(), out_poly_idxs.begin());

  cudf::size_type num_results{0};
  cudf::size_type num_parents{0};
  cudf::size_type num_children{0};

  // Traverse the quadtree starting at level 0 and descending to `max_depth` or until no more parent
  // quadrants are found.
  for (cudf::size_type level{0}; level < max_depth; ++level) {
    // Resize output device vectors and update the corresponding pointers. The next level will
    // expand out to no more than `num_parents * 4` pairs, since a parent quadrant can have no more
    // than 4 children.
    size_t max_num_results = num_results + num_parents * 4;
    if (max_num_results > out_types.capacity()) {
      // grow preallocated output sizes in multiples of the current capacity
      auto new_size = out_types.capacity() * ((max_num_results / out_types.capacity()) + 1);
      out_types.resize(new_size, stream);
      out_levels.resize(new_size, stream);
      out_node_idxs.resize(new_size, stream);
      out_poly_idxs.resize(new_size, stream);
      out_pairs = make_zip_iterator(
        out_types.begin(), out_levels.begin(), out_node_idxs.begin(), out_poly_idxs.begin());
    }

    if (level == 0) {
      // If at level 0, find intersections for all the top level quadrants and polygons
      auto node_indices = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        [num_top_level_children] __device__(auto const i) { return i % num_top_level_children; });
      auto poly_indices = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        [num_top_level_children] __device__(auto const i) { return i / num_top_level_children; });

      std::tie(num_parents, num_children) = find_intersections(quadtree,
                                                               poly_bbox,
                                                               node_indices,
                                                               poly_indices,
                                                               tmp_pairs,
                                                               out_pairs + num_results,
                                                               num_pairs,
                                                               x_min,
                                                               y_min,
                                                               scale,
                                                               max_depth,
                                                               stream);
    } else {
      // Otherwise, test the current level for intersections
      std::tie(num_parents, num_children) = find_intersections(quadtree,
                                                               poly_bbox,
                                                               tmp_node_idxs.begin(),
                                                               tmp_poly_idxs.begin(),
                                                               tmp_pairs,
                                                               out_pairs + num_results,
                                                               num_pairs,
                                                               x_min,
                                                               y_min,
                                                               scale,
                                                               max_depth,
                                                               stream);
    }

    num_results += num_children;

    // stop descending if no parent quadrants left to expand
    if (num_parents == 0) break;

    // Shrink the intermediate level storage buffers to overwrite removed elements
    tmp_types.shrink_to_fit(stream);
    tmp_levels.shrink_to_fit(stream);
    tmp_node_idxs.shrink_to_fit(stream);
    tmp_poly_idxs.shrink_to_fit(stream);

    // Use the current parent quad indices as the element indices lookup for the global child counts
    auto child_counts =
      thrust::make_permutation_iterator(node_counts.begin<uint32_t>(), tmp_node_idxs.begin());

    // Walk one level down and fill the intermediate storage buffers with the next level's poly and
    // quad indices
    auto next_level = descend_quadtree(child_counts,
                                       node_offsets.begin<uint32_t>(),
                                       num_parents,
                                       tmp_types,
                                       tmp_levels,
                                       tmp_node_idxs,
                                       tmp_poly_idxs,
                                       stream);

    num_pairs     = std::get<0>(next_level);
    tmp_types     = std::move(std::get<1>(next_level));
    tmp_levels    = std::move(std::get<2>(next_level));
    tmp_node_idxs = std::move(std::get<3>(next_level));
    tmp_poly_idxs = std::move(std::get<4>(next_level));
    // update tmp_pairs iterator to get ready for next level iteration
    tmp_pairs = make_zip_iterator(
      tmp_types.begin(), tmp_levels.begin(), tmp_node_idxs.begin(), tmp_poly_idxs.begin());
  }

  std::vector<std::unique_ptr<cudf::column>> cols{};
  cols.reserve(2);
  cols.push_back(make_fixed_width_column<int32_t>(num_results, stream, mr));
  cols.push_back(make_fixed_width_column<int32_t>(num_results, stream, mr));

  thrust::copy(rmm::exec_policy(stream)->on(stream),
               out_poly_idxs.begin(),
               out_poly_idxs.begin() + num_results,
               cols.at(0)->mutable_view().begin<uint32_t>());

  thrust::copy(rmm::exec_policy(stream)->on(stream),
               out_node_idxs.begin(),
               out_node_idxs.begin() + num_results,
               cols.at(1)->mutable_view().begin<uint32_t>());

  return std::make_unique<cudf::table>(std::move(cols));
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
                                                 cudf::size_type max_depth,
                                                 rmm::mr::device_memory_resource *mr,
                                                 cudaStream_t stream)
  {
    return join_quadtree_and_bboxes<T>(quadtree,
                                       poly_bbox,
                                       static_cast<T>(std::min(x_min, x_max)),
                                       static_cast<T>(std::max(x_min, x_max)),
                                       static_cast<T>(std::min(y_min, y_max)),
                                       static_cast<T>(std::max(y_min, y_max)),
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
                                            double x_max,
                                            double y_min,
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
                               x_max,
                               y_min,
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
                                            double x_max,
                                            double y_min,
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
    quadtree, poly_bbox, x_min, x_max, y_min, y_max, scale, max_depth, mr, cudaStream_t{0});
}

}  // namespace cuspatial
