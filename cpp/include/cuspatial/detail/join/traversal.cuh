/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cuspatial/detail/utility/z_order.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <tuple>

namespace cuspatial {
namespace detail {

/**
 * @brief Walk one level down the quadtree. Gather the matched quadrant and polygon pairs.
 *
 * @param counts The number of child quadrants or points of each node
 * @param offsets The first child position or point position of each node
 * @param num_quads The number of quadrants in the current level
 * @param parent_types bools indicating whether the node is a leaf (0) or quad (1)
 * @param parent_levels uint8_t levels for each node
 * @param parent_node_indices indices of the intersecting quadrants at the current level
 * @param parent_poly_indices indices of the intersecting polygons at the current level
 * @param stream CUDA stream on which to schedule work
 * @return A `std::tuple` containing the `int32_t` number of elements in the next level, and
 * `rmm::device_uvectors` for each of the `types`, `levels`, `quad_indices`, and `poly_indices` of
 * the next-level quadrants and polygons
 */
template <typename LengthsIterator, typename OffsetsIterator>
inline std::tuple<int32_t,
                  rmm::device_uvector<uint8_t>,
                  rmm::device_uvector<uint8_t>,
                  rmm::device_uvector<uint32_t>,
                  rmm::device_uvector<uint32_t>>
descend_quadtree(LengthsIterator counts,
                 OffsetsIterator offsets,
                 int32_t num_quads,
                 rmm::device_uvector<uint8_t>& parent_types,
                 rmm::device_uvector<uint8_t>& parent_levels,
                 rmm::device_uvector<uint32_t>& parent_node_indices,
                 rmm::device_uvector<uint32_t>& parent_poly_indices,
                 rmm::cuda_stream_view stream)
{
  // Use the current parent node indices as the lookup into the global child counts
  auto parent_counts = thrust::make_permutation_iterator(counts, parent_node_indices.begin());
  // scan on the number of child nodes to compute the offsets
  // note: size is `num_quads + 1` so the last element is `num_children`
  rmm::device_uvector<uint32_t> parent_offsets(num_quads + 1, stream);
  thrust::inclusive_scan(
    rmm::exec_policy(stream), parent_counts, parent_counts + num_quads, parent_offsets.begin() + 1);

  uint32_t init{0};
  parent_offsets.set_element_async(0, init, stream);

  auto num_children = parent_offsets.back_element(stream);  // synchronizes stream

  rmm::device_uvector<uint32_t> parent_indices(num_children, stream);
  // fill with zeroes
  thrust::fill(rmm::exec_policy(stream), parent_indices.begin(), parent_indices.end(), 0);
  // use the parent_offsets as the map to scatter sequential parent_indices
  thrust::scatter(rmm::exec_policy(stream),
                  thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(0) + num_quads,
                  parent_offsets.begin(),
                  parent_indices.begin());

  // inclusive scan with maximum functor to fill the empty elements with their left-most non-empty
  // elements. `parent_indices` is now a full array of the sequence index of each quadrant's parent
  thrust::inclusive_scan(rmm::exec_policy(stream),
                         parent_indices.begin(),
                         parent_indices.begin() + num_children,
                         parent_indices.begin(),
                         thrust::maximum<uint32_t>());

  // allocate new vectors for the next level
  rmm::device_uvector<uint8_t> child_types(num_children, stream);
  rmm::device_uvector<uint8_t> child_levels(num_children, stream);
  rmm::device_uvector<uint32_t> child_quad_indices(num_children, stream);
  rmm::device_uvector<uint32_t> child_poly_indices(num_children, stream);

  // `parent_indices` is a gather map to retrieve non-leaf quads' respective child nodes
  thrust::gather(rmm::exec_policy(stream),
                 parent_indices.begin(),
                 parent_indices.begin() + num_children,
                 // curr level iterator
                 thrust::make_zip_iterator(parent_types.begin(),
                                           parent_levels.begin(),
                                           parent_node_indices.begin(),
                                           parent_poly_indices.begin()),
                 // next level iterator
                 thrust::make_zip_iterator(child_types.begin(),
                                           child_levels.begin(),
                                           child_quad_indices.begin(),
                                           child_poly_indices.begin()));

  rmm::device_uvector<uint32_t> relative_child_offsets(num_children, stream);
  thrust::exclusive_scan_by_key(rmm::exec_policy(stream),
                                parent_indices.begin(),
                                parent_indices.begin() + num_children,
                                thrust::constant_iterator<uint32_t>(1),
                                relative_child_offsets.begin());

  // compute child quad indices using parent and relative child offsets
  auto child_offsets_iter = thrust::make_permutation_iterator(offsets, child_quad_indices.begin());
  thrust::transform(rmm::exec_policy(stream),
                    child_offsets_iter,
                    child_offsets_iter + num_children,
                    relative_child_offsets.begin(),
                    child_quad_indices.begin(),
                    thrust::plus<uint32_t>());

  return std::make_tuple(num_children,
                         std::move(child_types),
                         std::move(child_levels),
                         std::move(child_quad_indices),
                         std::move(child_poly_indices));
}
}  // namespace detail
}  // namespace cuspatial
