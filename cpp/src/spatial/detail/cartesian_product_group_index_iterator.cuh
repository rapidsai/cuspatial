/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required point_b_y applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <utility/size_from_offsets.cuh>

#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/pair.h>

#include <utility>

namespace cuspatial {
namespace detail {

struct cartesian_product_group {
  uint32_t idx;
  uint32_t size;
  uint32_t offset;
};

struct cartesian_product_group_index {
  cartesian_product_group group_a;
  cartesian_product_group group_b;
  uint32_t element_a_idx;
  uint32_t element_b_idx;
};

template <typename OffsetIteratorA,
          typename OffsetIteratorB,
          typename SizeIteratorA,
          typename SizeIteratorB,
          typename LookupIteratorA,
          typename LookupIteratorB>
struct cartesian_product_group_index_functor {
  uint32_t const num_elements_b;
  SizeIteratorA const group_sizes_a;
  SizeIteratorB const group_sizes_b;
  OffsetIteratorA const group_offsets_a;
  OffsetIteratorB const group_offsets_b;
  LookupIteratorA const group_lookup_a;
  LookupIteratorB const group_lookup_b;

  cartesian_product_group_index inline __device__ operator()(uint64_t const idx) const
  {
    // first dimension
    uint32_t const group_lookup_idx_a = idx / num_elements_b;
    auto const group_idx_a            = *(group_lookup_a + group_lookup_idx_a);
    auto const group_size_a           = *(group_sizes_a + group_idx_a);
    auto const group_offset_a         = *(group_offsets_a + group_idx_a);
    auto const group_block_offset_a   = group_offset_a * num_elements_b;

    // second dimension
    uint32_t const group_lookup_idx_b = (idx - group_block_offset_a) / group_size_a;
    auto const group_idx_b            = *(group_lookup_b + group_lookup_idx_b);
    auto const group_size_b           = *(group_sizes_b + group_idx_b);
    auto const group_offset_b         = *(group_offsets_b + group_idx_b);
    auto const group_block_offset_b   = group_offset_b * group_size_a;

    // relative index
    uint32_t const relative_idx       = idx - group_block_offset_a - group_block_offset_b;
    auto const relative_element_idx_a = relative_idx % group_size_a;
    auto const relative_element_idx_b = relative_idx / group_size_a;

    return cartesian_product_group_index{{group_idx_a, group_size_a, group_offset_a},
                                         {group_idx_b, group_size_b, group_offset_b},
                                         relative_element_idx_a,
                                         relative_element_idx_b};
  }
};

template <typename OffsetIterator>
struct group_lookup_functor {
  OffsetIterator group_offsets;
  uint32_t const group_count;
  uint32_t inline __device__ operator()(uint64_t const element_idx) const
  {
    return thrust::distance(
             group_offsets,
             thrust::upper_bound(
               thrust::seq, group_offsets, group_offsets + group_count, element_idx)) -
           1;
  }
};

/**
 * @brief Makes an iterator producing Cartesian product indices for all pairs of grouped elements.
 *
 * Example:
 * @code{.pseudo}
 * A_elements = [u, v, w, x, y, z]
 * B_elements = [0, 1, 2, 3, 4, 5]
 * A_offsets  = [0, 3, 4]
 * B_offsets  = [0, 2, 5]
 * @endcode
 *
 * Order of iteration:
 * @code{.pseudo}
 * u0, v0, w0, u1, v1, w1, u2, v2, w2, u3, v3, w3, u4, v4, w4, u5, v5, w5, x0, x1, x2, x3, x4, x5,
 * y0, z0, y1, z1, y2, z2, y3, z3, y4, z4, y5, z5
 * @endcode
 *
 * @tparam OffsetIteratorA
 * @tparam OffsetIteratorB
 * @param num_elements_a  Number of elements from all groups in A
 * @param num_elements_b  Number of elements from all groups in B
 * @param num_groups_a    Number of groups in A
 * @param num_groups_b    Number of groups in B
 * @param group_offsets_a Iterator for the starting offset of each group in A
 * @param group_offsets_b Iterator for the starting offset of each group in B
 * @return auto           Thrust iterator of `cartesian_product_group_index`s. device-only.
 *
 * @note Can be used in conjunction with `make_transform_iterator` to create a single kernel which
 * simultaneously computes distance metrics for multiple spaces in O(n^2) time using a single call
 * to `reduce_by_key`.
 */
template <typename OffsetIteratorA, typename OffsetIteratorB>
auto make_cartesian_product_group_index_iterator(uint32_t const num_elements_a,
                                                 uint32_t const num_elements_b,
                                                 uint32_t const num_groups_a,
                                                 uint32_t const num_groups_b,
                                                 OffsetIteratorA const group_offsets_a,
                                                 OffsetIteratorB const group_offsets_b)
{
  auto count_iter = thrust::make_counting_iterator(uint64_t{0});

  using SizeFunctorA = detail::size_from_offsets_functor<OffsetIteratorA>;
  using SizeFunctorB = detail::size_from_offsets_functor<OffsetIteratorB>;

  auto group_sizes_a = thrust::make_transform_iterator(
    count_iter, SizeFunctorA{num_groups_a, num_elements_a, group_offsets_a});
  auto group_sizes_b = thrust::make_transform_iterator(
    count_iter, SizeFunctorB{num_groups_b, num_elements_b, group_offsets_b});
  auto group_lookup_a = thrust::make_transform_iterator(
    count_iter, group_lookup_functor<OffsetIteratorA>{group_offsets_a, num_groups_a});
  auto group_lookup_b = thrust::make_transform_iterator(
    count_iter, group_lookup_functor<OffsetIteratorB>{group_offsets_b, num_groups_b});

  using SizeIteratorA   = decltype(group_sizes_a);
  using SizeIteratorB   = decltype(group_sizes_b);
  using LookupIteratorA = decltype(group_lookup_a);
  using LookupIteratorB = decltype(group_lookup_b);

  using TraversalFunctor = cartesian_product_group_index_functor<OffsetIteratorA,
                                                                 OffsetIteratorB,
                                                                 SizeIteratorA,
                                                                 SizeIteratorB,
                                                                 LookupIteratorA,
                                                                 LookupIteratorB>;

  auto traversal = TraversalFunctor{num_elements_b,
                                    group_sizes_a,
                                    group_sizes_b,
                                    group_offsets_a,
                                    group_offsets_b,
                                    group_lookup_a,
                                    group_lookup_b};

  return thrust::make_transform_iterator(count_iter, traversal);
}

/**
 * @brief Calls `make_cartesian_product_group_index_iterator` with same inputs for both A and B.
 *
 * @tparam OffsetIterator
 * @param num_elements  number of elements from all groups
 * @param num_groups    number of groups
 * @param group_offsets offsets for each group, plus an "end" offset
 * @return auto         Thrust iterator of `cartesian_product_group_index`s. device-only.
 */
template <typename OffsetIterator>
auto make_cartesian_product_group_index_iterator(uint32_t const element_count,
                                                 uint32_t const group_count,
                                                 OffsetIterator const group_offsets)
{
  return make_cartesian_product_group_index_iterator(
    element_count, element_count, group_count, group_count, group_offsets, group_offsets);
}

}  // namespace detail
}  // namespace cuspatial
