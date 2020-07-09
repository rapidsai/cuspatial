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

#pragma once

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_uvector.hpp>

#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/tuple.h>

#include <tuple>

#include "indexing/construction/detail/utilities.cuh"
#include "utility/z_order.cuh"

namespace cuspatial {
namespace detail {

template <typename LengthsIter, typename OffsetsIter>
inline std::tuple<cudf::size_type,
                  rmm::device_uvector<uint8_t>,
                  rmm::device_uvector<uint8_t>,
                  rmm::device_uvector<uint32_t>,
                  rmm::device_uvector<uint32_t>>
descend_quadtree(LengthsIter lengths,
                 OffsetsIter offsets,
                 cudf::size_type num_quads,
                 rmm::device_uvector<uint8_t> &quad_types,
                 rmm::device_uvector<uint8_t> &quad_levels,
                 rmm::device_uvector<uint32_t> &quad_node_indices,
                 rmm::device_uvector<uint32_t> &quad_poly_indices,
                 cudaStream_t stream)
{
  // scan on the number of child nodes to compute the offsets
  // note: size is num_quads + 1 so the last element is `num_children`
  rmm::device_uvector<uint32_t> parent_offsets(num_quads + 1, stream);
  thrust::inclusive_scan(
    rmm::exec_policy(stream)->on(stream), lengths, lengths + num_quads, parent_offsets.begin() + 1);

  parent_offsets.set_element_async(0, 0, stream);

  auto num_children = parent_offsets.back_element(stream);  // synchronizes stream

  rmm::device_uvector<uint32_t> child_indices(num_children, stream);
  thrust::fill(rmm::exec_policy(stream)->on(stream), child_indices.begin(), child_indices.end(), 0);
  // use the parent_offsets as the map to scatter sequential child_indices
  thrust::scatter(rmm::exec_policy(stream)->on(stream),
                  thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(0) + num_quads,
                  parent_offsets.begin(),
                  child_indices.begin());

  // inclusive scan with maximum functor to fill the empty elements with their left-most non-empty
  // elements. `parent_offsets` is now a full array of the sequence index of each quadrant's parent
  thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream),
                         child_indices.begin(),
                         child_indices.begin() + num_children,
                         child_indices.begin(),
                         thrust::maximum<uint32_t>());

  // allocate new vectors for the next level
  rmm::device_uvector<uint8_t> child_types(num_children, stream);
  rmm::device_uvector<uint8_t> child_levels(num_children, stream);
  rmm::device_uvector<uint32_t> child_quad_indices(num_children, stream);
  rmm::device_uvector<uint32_t> child_poly_indices(num_children, stream);

  // `child_indices` is a gather map to retrieve non-leaf quads' respective child nodes
  thrust::gather(rmm::exec_policy(stream)->on(stream),
                 child_indices.begin(),
                 child_indices.begin() + num_children,
                 // curr level iterator
                 make_zip_iterator(quad_types.begin(),
                                   quad_levels.begin(),
                                   quad_node_indices.begin(),
                                   quad_poly_indices.begin()),
                 // next level iterator
                 make_zip_iterator(child_types.begin(),
                                   child_levels.begin(),
                                   child_quad_indices.begin(),
                                   child_poly_indices.begin()));

  rmm::device_uvector<uint32_t> relative_child_offsets(num_children, stream);
  thrust::exclusive_scan_by_key(rmm::exec_policy(stream)->on(stream),
                                child_indices.begin(),
                                child_indices.begin() + num_children,
                                thrust::constant_iterator<uint32_t>(1),
                                relative_child_offsets.begin());

  // compute child quad indices using parent and relative child offsets
  auto child_offsets_iter = thrust::make_permutation_iterator(offsets, child_quad_indices.begin());
  thrust::transform(rmm::exec_policy(stream)->on(stream),
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
