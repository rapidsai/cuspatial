/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/pair.h>
#include <utility>
#include "utility/size_from_offsets.cuh"

namespace cuspatial {
namespace detail {

struct cartesian_product_group {
  int32_t idx;
  int32_t size;
  int32_t offset;
};

struct cartesian_product_group_index {
  cartesian_product_group group_a;
  cartesian_product_group group_b;
  int32_t element_a_idx;
  int32_t element_b_idx;
};

template <typename OffsetIteratorA,
          typename OffsetIteratorB,
          typename SizeIteratorLHS,
          typename SizeIteratorRHS,
          typename LookupIteratorLHS,
          typename LookupIteratorRHS>
struct cartesian_product_functor {
  int32_t const num_elements_b;
  SizeIteratorLHS const group_sizes_a;
  SizeIteratorRHS const group_sizes_b;
  OffsetIteratorA const group_offsets_a;
  OffsetIteratorB const group_offsets_b;
  LookupIteratorLHS const group_lookup_a;
  LookupIteratorRHS const group_lookup_b;

  __device__ cartesian_product_group_index operator()(int64_t const idx) const
  {
    // first dimension
    int32_t group_lookup_idx_a = idx / num_elements_b;
    auto group_idx_a           = *(group_lookup_a + group_lookup_idx_a);
    auto group_size_a          = *(group_sizes_a + group_idx_a);
    auto group_offset_a        = *(group_offsets_a + group_idx_a);
    auto group_block_offset_a  = group_offset_a * num_elements_b;

    // second dimension
    int32_t group_lookup_idx_b = (idx - group_block_offset_a) / group_size_a;
    auto group_idx_b           = *(group_lookup_b + group_lookup_idx_b);
    auto group_size_b          = *(group_sizes_b + group_idx_b);
    auto group_offset_b        = *(group_offsets_b + group_idx_b);
    auto group_block_offset_b  = group_offset_b * group_size_a;

    // relative index
    int32_t relative_idx        = idx - group_block_offset_a - group_block_offset_b;
    auto relative_element_idx_a = relative_idx % group_size_a;
    auto relative_element_idx_b = relative_idx / group_size_a;

    return cartesian_product_group_index{{group_idx_a, group_size_a, group_offset_a},
                                         {group_idx_b, group_size_b, group_offset_b},
                                         relative_element_idx_a,
                                         relative_element_idx_b};
  }
};

template <typename OffsetIterator>
struct group_lookup_functor {
  OffsetIterator group_offsets;
  int32_t group_count;
  __device__ int32_t operator()(int32_t element_idx) const
  {
    return thrust::upper_bound(
             thrust::seq, group_offsets, group_offsets + group_count, element_idx) -
           group_offsets - 1;
  }
};

/**
 * @brief Makes an iterator of the cartesian product of two iterators. Produces pairs consecutively.
 *
 * Pairs of elements grouped by offsets `A` and `B` are produces consecutively.
 * - Each group `A_i` in `A` appears consecutively.
 * - Under each group `A_i`, each group `B_i` in `B` appear consecutively.
 * - Under each group `B_i`, each element within `a` appears consecutively.
 * - Under each element within `A_i`, each element within `B_i` appears consecutively.
 *
 * Example:
 * ```
 * A_elements = [u, v, w, x, y, z]
 * B_elements = [0, 1, 2, 3, 4, 5]
 * A_offsets  = [0, 3, 4]
 * B_offsets  = [0, 2, 5]
 * ```
 * Cartesian Product:
 *
 *       B0        B1             B2
 *     +---------+--------------+----+
 *  A0 : u0   u1 : u2   u3   u4 : u5 :
 *     +         :              :    +
 *     : v0   v1 : v2   v3   v4 : v5 :
 *     +         :              :    +
 *     : w0   w1 : w2   w3   w4 : w5 :
 *     +---------+--------------+----+
 *  A1 : x0   x1 : x2   x3   x4 : x5 :
 *     +---------+--------------+----+
 *  A2 : y0   y1 : y2   y3   y4 : y5 :
 *     +         :              :    +
 *     : z0   z1 : z2   z3   z4 : z5 :
 *     +---------+--------------+----+
 *
 * Order of iteration:
 *
 *  u0, v0, w0, u1, v1, w1, u2, v2, w2, u3, v3, w3, u4, v4, w4, u5, v5, w5, x0, x1, x2, x3, x4, x5,
 *  y0, z0, y1, z1, y2, z2, y3, z3, y4, z4, y5, z5
 *
 * i.e:
 *
 *  A0 * B0 | u0 v0 w0 u1 v1 w1
 *  A0 * B1 | u2 v2 w2 u3 v3 w3 u4 v4 w4
 *  A0 * B2 | u5 v5 w5
 *  A1 * B0 | x0 x1
 *  A1 * B1 | x2 x3 x4
 *  A1 * B2 | x5
 *  A1 * B0 | y0 z0 y1 z1
 *  A2 * B1 | y2 z2 y3 z3 y4 z4
 *  A2 * B2 | y5 z5
 *
 * @tparam OffsetIteratorA
 * @tparam OffsetIteratorB
 * @param num_elements_a  number of elements from all groups in A
 * @param num_elements_b  number of elements from all groups in B
 * @param num_groups_a    number of groups in A
 * @param num_groups_b    number of groups in B
 * @param group_offsets_a offsets for each group in A, plus an "end" offset
 * @param group_offsets_b offsets for each group in B, plus an "end" offset
 * @return auto           Thrust iterator of `cartesian_product_group_index`s. device-only.
 *
 * @note Can be used in conjunction with `make_transform_iterator` to create a fused kernel which
 * simultaneously computes distance metrics for multiple spaces in O(n^2) time using a single call
 * to `reduce_by_key`.
 */
template <typename OffsetIteratorA, typename OffsetIteratorB>
auto make_grouped_cartesian_product_iterator(int32_t const num_elements_a,
                                             int32_t const num_elements_b,
                                             int32_t const num_groups_a,
                                             int32_t const num_groups_b,
                                             OffsetIteratorA const group_offsets_a,
                                             OffsetIteratorB const group_offsets_b)
{
  auto count_iter = thrust::make_counting_iterator(0);

  using SizeFunctorLHS = detail::size_from_offsets_functor<OffsetIteratorA>;
  using SizeFunctorRHS = detail::size_from_offsets_functor<OffsetIteratorB>;

  auto group_sizes_a = thrust::make_transform_iterator(
    count_iter, SizeFunctorLHS{num_groups_a, num_elements_a, group_offsets_a});
  auto group_sizes_b = thrust::make_transform_iterator(
    count_iter, SizeFunctorRHS{num_groups_b, num_elements_b, group_offsets_b});
  auto group_lookup_a = thrust::make_transform_iterator(
    count_iter, group_lookup_functor<OffsetIteratorA>{group_offsets_a, num_groups_a});
  auto group_lookup_b = thrust::make_transform_iterator(
    count_iter, group_lookup_functor<OffsetIteratorB>{group_offsets_b, num_groups_b});

  using SizeIteratorLHS   = decltype(group_sizes_a);
  using SizeIteratorRHS   = decltype(group_sizes_b);
  using LookupIteratorLHS = decltype(group_lookup_a);
  using LookupIteratorRHS = decltype(group_lookup_b);

  using TraversalFunctor = cartesian_product_functor<OffsetIteratorA,
                                                     OffsetIteratorB,
                                                     SizeIteratorLHS,
                                                     SizeIteratorRHS,
                                                     LookupIteratorLHS,
                                                     LookupIteratorRHS>;

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
 * @brief Calls `make_grouped_cartesian_product_iterator` with same inputs for both A and B.
 *
 * @tparam OffsetIterator
 * @param num_elements  number of elements from all groups
 * @param num_groups    number of groups
 * @param group_offsets offsets for each group, plus an "end" offset
 * @return auto         Thrust iterator of `cartesian_product_group_index`s. device-only.
 */
template <typename OffsetIterator>
auto make_grouped_cartesian_product_iterator(int32_t const element_count,
                                             int32_t const group_count,
                                             OffsetIterator const group_offsets)
{
  return make_grouped_cartesian_product_iterator(
    element_count, element_count, group_count, group_count, group_offsets, group_offsets);
}

}  // namespace detail
}  // namespace cuspatial
