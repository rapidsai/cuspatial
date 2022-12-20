/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#pragma once

#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/experimental/detail/join/intersection.cuh>
#include <cuspatial/experimental/detail/join/traversal.cuh>
#include <cuspatial/experimental/point_quadtree.cuh>
#include <cuspatial/traits.hpp>

#include <thrust/iterator/discard_iterator.h>

#include <iterator>
#include <utility>

namespace cuspatial {

template <class KeyIterator,
          class LevelIterator,
          class IsInternalIterator,
          class BoundingBoxIterator,
          class T>
std::pair<rmm::device_uvector<uint32_t>, rmm::device_uvector<uint32_t>>
join_quadtree_and_bounding_boxes(KeyIterator keys_first,
                                 KeyIterator keys_last,
                                 LevelIterator levels_first,
                                 IsInternalIterator is_internal_nodes_first,
                                 KeyIterator lengths_first,
                                 KeyIterator offsets_first,
                                 BoundingBoxIterator bounding_boxes_first,
                                 BoundingBoxIterator bounding_boxes_last,
                                 T x_min,
                                 T y_min,
                                 T scale,
                                 int8_t max_depth,
                                 rmm::mr::device_memory_resource* mr,
                                 rmm::cuda_stream_view stream)
{
  static_assert(is_same<T, cuspatial::iterator_vec_base_type<BoundingBoxIterator>>(),
                "Iterator value_type mismatch");

  auto const num_nodes = std::distance(keys_first, keys_last);
  auto const num_boxes = std::distance(bounding_boxes_first, bounding_boxes_last);

  // Count the number of top-level nodes to start.
  // This could be provided explicitly, but count_if should be fast enough.
  auto num_top_level_leaves = thrust::count_if(rmm::exec_policy(stream),
                                               levels_first,
                                               levels_first + num_nodes,
                                               thrust::placeholders::_1 == 0);

  auto num_pairs = num_top_level_leaves * num_boxes;

  int32_t num_leaves{0};
  int32_t num_results{0};
  int32_t num_parents{0};

  // The found bbox-quad pairs are dynamic and can not be pre-allocated.
  // Relevant arrays are resized accordingly for memory efficiency.

  // Vectors for intermediate bbox and node indices at each level
  rmm::device_uvector<uint8_t> cur_types(num_pairs, stream);
  rmm::device_uvector<uint8_t> cur_levels(num_pairs, stream);
  rmm::device_uvector<uint32_t> cur_node_idxs(num_pairs, stream);
  rmm::device_uvector<uint32_t> cur_bbox_idxs(num_pairs, stream);

  // Vectors for found pairs of bbox and leaf node indices
  rmm::device_uvector<uint32_t> out_node_idxs(num_pairs, stream, mr);
  rmm::device_uvector<uint32_t> out_bbox_idxs(num_pairs, stream, mr);

  auto make_current_level_iter = [&]() {
    return thrust::make_zip_iterator(
      cur_types.begin(), cur_levels.begin(), cur_node_idxs.begin(), cur_bbox_idxs.begin());
  };

  auto make_output_values_iter = [&]() {
    return num_results + thrust::make_zip_iterator(thrust::make_discard_iterator(),
                                                   thrust::make_discard_iterator(),
                                                   out_node_idxs.begin(),
                                                   out_bbox_idxs.begin());
  };

  // Find intersections for all the top level quadrants and bounding boxes
  std::tie(num_parents, num_leaves) =
    detail::find_intersections(keys_first,
                               levels_first,
                               is_internal_nodes_first,
                               bounding_boxes_first,
                               // The top-level node indices
                               detail::make_counting_transform_iterator(
                                 0, [=] __device__(auto i) { return i % num_top_level_leaves; }),
                               // The top-level bbox indices
                               detail::make_counting_transform_iterator(
                                 0, [=] __device__(auto i) { return i / num_top_level_leaves; }),
                               make_current_level_iter(),  // intermediate intersections or parent
                                                           // quadrants found during traversal
                               // found intersecting quadrant and bbox indices for output
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
    cur_bbox_idxs.shrink_to_fit(stream);

    // Grow preallocated output vectors. The next level will expand out to no more
    // than `num_parents * 4` pairs, since each parent quadrant has up to 4 children.
    size_t max_num_results = num_results + num_parents * 4;
    if (max_num_results > out_node_idxs.capacity()) {
      // TODO: grow preallocated output sizes in multiples of the current capacity?
      // auto new_size = out_node_idxs.capacity() *  //
      //                 ((max_num_results / out_node_idxs.capacity()) + 1);
      out_node_idxs.resize(max_num_results, stream);
      out_bbox_idxs.resize(max_num_results, stream);
    }

    // Walk one level down and fill the current level's vectors with
    // the next level's quadrant info and bbox indices.
    std::tie(num_pairs, cur_types, cur_levels, cur_node_idxs, cur_bbox_idxs) =
      detail::descend_quadtree(lengths_first,
                               offsets_first,
                               num_parents,
                               cur_types,
                               cur_levels,
                               cur_node_idxs,
                               cur_bbox_idxs,
                               stream);

    // Find intersections for the the next level's quadrants and bboxes
    std::tie(num_parents, num_leaves) =
      detail::find_intersections(keys_first,
                                 levels_first,
                                 is_internal_nodes_first,
                                 bounding_boxes_first,
                                 cur_node_idxs.begin(),
                                 cur_bbox_idxs.begin(),
                                 make_current_level_iter(),  // intermediate intersections or parent
                                                             // quadrants found during traversal
                                 // found intersecting quadrant and bbox indices for output
                                 make_output_values_iter(),
                                 num_pairs,
                                 x_min,
                                 y_min,
                                 scale,
                                 max_depth,
                                 stream);

    num_results += num_leaves;
  }

  // Sort the output bbox/quad indices by quadrant
  [&]() {
    // Copy the relevant node offsets into a temporary vector so we don't modify the quadtree column
    rmm::device_uvector<uint32_t> tmp_node_offsets(num_results, stream);

    auto const iter = thrust::make_permutation_iterator(offsets_first, out_node_idxs.begin());

    thrust::copy(rmm::exec_policy(stream), iter, iter + num_results, tmp_node_offsets.begin());

    thrust::stable_sort_by_key(
      rmm::exec_policy(stream),
      tmp_node_offsets.begin(),
      tmp_node_offsets.end(),
      thrust::make_zip_iterator(out_bbox_idxs.begin(), out_node_idxs.begin()));
  }();

  out_node_idxs.resize(num_results, stream);
  out_bbox_idxs.resize(num_results, stream);
  out_node_idxs.shrink_to_fit(stream);
  out_bbox_idxs.shrink_to_fit(stream);

  return {std::move(out_bbox_idxs), std::move(out_node_idxs)};
}

}  // namespace cuspatial
