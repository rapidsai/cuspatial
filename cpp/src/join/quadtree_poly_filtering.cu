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

#include "detail/intersection.cuh"
#include "detail/traversal.cuh"

#include <cuspatial/error.hpp>
#include <cuspatial/spatial_join.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <tuple>

namespace cuspatial {

namespace detail {

namespace {

template <typename UnaryFunction>
inline auto make_counting_transform_iterator(cudf::size_type start, UnaryFunction f)
{
  return thrust::make_transform_iterator(thrust::make_counting_iterator(start), f);
}

template <typename T>
inline std::unique_ptr<cudf::table> join_quadtree_and_bboxes(cudf::table_view const &quadtree,
                                                             cudf::table_view const &poly_bbox,
                                                             T x_min,
                                                             T x_max,
                                                             T y_min,
                                                             T y_max,
                                                             T scale,
                                                             int8_t max_depth,
                                                             rmm::cuda_stream_view stream,
                                                             rmm::mr::device_memory_resource *mr)
{
  auto const node_levels  = quadtree.column(1);  // uint8_t
  auto const node_counts  = quadtree.column(3);  // uint32_t
  auto const node_offsets = quadtree.column(4);  // uint32_t

  // Count the number of top-level nodes to start.
  // This could be provided explicitly, but count_if should be fast enough.
  auto num_top_level_leaves = thrust::count_if(rmm::exec_policy(stream),
                                               node_levels.begin<uint8_t>(),
                                               node_levels.end<uint8_t>(),
                                               thrust::placeholders::_1 == 0);

  auto num_pairs = num_top_level_leaves * poly_bbox.num_rows();

  // The found poly-quad pairs are dynamic and can not be pre-allocated.
  // Relevant arrays are resized accordingly for memory efficiency.

  // Vectors for intermediate poly and node indices at each level
  rmm::device_uvector<uint8_t> cur_types(num_pairs, stream);
  rmm::device_uvector<uint8_t> cur_levels(num_pairs, stream);
  rmm::device_uvector<uint32_t> cur_node_idxs(num_pairs, stream);
  rmm::device_uvector<uint32_t> cur_poly_idxs(num_pairs, stream);

  // Vectors for found pairs of poly and leaf node indices
  rmm::device_uvector<uint8_t> out_types(num_pairs, stream);
  rmm::device_uvector<uint8_t> out_levels(num_pairs, stream);
  rmm::device_uvector<uint32_t> out_node_idxs(num_pairs, stream);
  rmm::device_uvector<uint32_t> out_poly_idxs(num_pairs, stream);

  cudf::size_type num_leaves{0};
  cudf::size_type num_results{0};
  cudf::size_type num_parents{0};

  auto make_current_level_iter = [&]() {
    return make_zip_iterator(
      cur_types.begin(), cur_levels.begin(), cur_node_idxs.begin(), cur_poly_idxs.begin());
  };

  auto make_output_values_iter = [&]() {
    return num_results +
           make_zip_iterator(
             out_types.begin(), out_levels.begin(), out_node_idxs.begin(), out_poly_idxs.begin());
  };

  // Find intersections for all the top level quadrants and polygons
  std::tie(num_parents, num_leaves) =
    find_intersections(quadtree,
                       poly_bbox,
                       // The top-level node indices
                       make_counting_transform_iterator(
                         0, [=] __device__(auto i) { return i % num_top_level_leaves; }),
                       // The top-level poly indices
                       make_counting_transform_iterator(
                         0, [=] __device__(auto i) { return i / num_top_level_leaves; }),
                       make_current_level_iter(),  // intermediate intersections or parent quadrants
                                                   // found during traversal
                       // found intersecting quadrant and polygon indices for output
                       make_output_values_iter(),
                       num_pairs,
                       x_min,
                       y_min,
                       scale,
                       max_depth,
                       stream);

  num_results += num_leaves;

  // Traverse the quadtree descending to `max_depth`, or until no more parent quadrants are found
  for (uint8_t level{1}; level < max_depth && num_parents > 0; ++level) {
    // Shrink the current level's vectors to overwrite elements removed by `find_intersections()`
    cur_types.shrink_to_fit(stream);
    cur_levels.shrink_to_fit(stream);
    cur_node_idxs.shrink_to_fit(stream);
    cur_poly_idxs.shrink_to_fit(stream);

    // Grow preallocated output vectors. The next level will expand out to no more
    // than `num_parents * 4` pairs, since each parent quadrant has up to 4 children.
    size_t max_num_results = num_results + num_parents * 4;
    if (max_num_results > out_types.capacity()) {
      // grow preallocated output sizes in multiples of the current capacity
      // auto new_size = out_types.capacity() * ((max_num_results / out_types.capacity()) + 1);
      out_types.resize(max_num_results, stream);
      out_levels.resize(max_num_results, stream);
      out_node_idxs.resize(max_num_results, stream);
      out_poly_idxs.resize(max_num_results, stream);
    }

    // Walk one level down and fill the current level's vectors with
    // the next level's quadrant info and polygon indices.
    std::tie(num_pairs, cur_types, cur_levels, cur_node_idxs, cur_poly_idxs) =
      descend_quadtree(node_counts.begin<uint32_t>(),
                       node_offsets.begin<uint32_t>(),
                       num_parents,
                       cur_types,
                       cur_levels,
                       cur_node_idxs,
                       cur_poly_idxs,
                       stream);

    // Find intersections for the the next level's quadrants and polygons
    std::tie(num_parents, num_leaves) =
      find_intersections(quadtree,
                         poly_bbox,
                         cur_node_idxs.begin(),
                         cur_poly_idxs.begin(),
                         make_current_level_iter(),  // intermediate intersections or parent
                                                     // quadrants found during traversal
                         // found intersecting quadrant and polygon indices for output
                         make_output_values_iter(),
                         num_pairs,
                         x_min,
                         y_min,
                         scale,
                         max_depth,
                         stream);

    num_results += num_leaves;
  }

  std::vector<std::unique_ptr<cudf::column>> cols{};
  cols.reserve(2);
  cols.push_back(make_fixed_width_column<uint32_t>(num_results, stream, mr));
  cols.push_back(make_fixed_width_column<uint32_t>(num_results, stream, mr));

  thrust::copy(rmm::exec_policy(stream),
               out_poly_idxs.begin(),
               out_poly_idxs.begin() + num_results,
               cols.at(0)->mutable_view().begin<uint32_t>());

  thrust::copy(rmm::exec_policy(stream),
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
                                                 int8_t max_depth,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::mr::device_memory_resource *mr)
  {
    return join_quadtree_and_bboxes<T>(quadtree,
                                       poly_bbox,
                                       static_cast<T>(x_min),
                                       static_cast<T>(x_max),
                                       static_cast<T>(y_min),
                                       static_cast<T>(y_max),
                                       static_cast<T>(scale),
                                       max_depth,
                                       stream,
                                       mr);
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

std::unique_ptr<cudf::table> join_quadtree_and_bounding_boxes(cudf::table_view const &quadtree,
                                                              cudf::table_view const &poly_bbox,
                                                              double x_min,
                                                              double x_max,
                                                              double y_min,
                                                              double y_max,
                                                              double scale,
                                                              int8_t max_depth,
                                                              rmm::cuda_stream_view stream,
                                                              rmm::mr::device_memory_resource *mr)
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
                               stream,
                               mr);
}

}  // namespace detail

std::unique_ptr<cudf::table> join_quadtree_and_bounding_boxes(cudf::table_view const &quadtree,
                                                              cudf::table_view const &poly_bbox,
                                                              double x_min,
                                                              double x_max,
                                                              double y_min,
                                                              double y_max,
                                                              double scale,
                                                              int8_t max_depth,
                                                              rmm::mr::device_memory_resource *mr)
{
  CUSPATIAL_EXPECTS(quadtree.num_columns() == 5, "quadtree table must have 5 columns");
  CUSPATIAL_EXPECTS(poly_bbox.num_columns() == 4, "polygon bbox table must have 4 columns");
  CUSPATIAL_EXPECTS(scale > 0, "scale must be positive");
  CUSPATIAL_EXPECTS(x_min < x_max && y_min < y_max,
                    "invalid bounding box (x_min, x_max, y_min, y_max)");
  CUSPATIAL_EXPECTS(max_depth > 0 && max_depth < 16,
                    "maximum depth must be positive and less than 16");

  if (quadtree.num_rows() == 0 || poly_bbox.num_rows() == 0) {
    std::vector<std::unique_ptr<cudf::column>> cols{};
    cols.reserve(2);
    cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::UINT32}));
    cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::UINT32}));
    return std::make_unique<cudf::table>(std::move(cols));
  }

  return detail::join_quadtree_and_bounding_boxes(quadtree,
                                                  poly_bbox,
                                                  x_min,
                                                  x_max,
                                                  y_min,
                                                  y_max,
                                                  scale,
                                                  max_depth,
                                                  rmm::cuda_stream_default,
                                                  mr);
}

}  // namespace cuspatial
