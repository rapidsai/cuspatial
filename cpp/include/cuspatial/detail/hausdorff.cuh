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

#include <thrust/tuple.h>

#include <cstdint>

namespace cuspatial {
namespace detail {

/**
* @brief Hausdorff reduction data structure
*
* Data structure for computing directed hausdorff distance as a noncommutative reduction.

* Given one `hausdorff_acc<T>` for each distance between all points in two spaces (O(N^2)) and a
* binary reduce algorithm which supports noncommutative operations, this data structure and
* accompoanying reduce operator (`+`) can be used to calculate the directed hausdorff distance
* between those two spaces. One of two asymetric directed distances can be deduced. The asymetric
* distance which is computed is determined by the order of inputs, and the columns in which those
* inputs reside.
*
* ```
* // the distances from a 3-point space to a 2-point space.
* auto d1 = hausdorff_acc<float>(..., 1, distance_11);
* auto d2 = hausdorff_acc<float>(..., 1, distance_12);
* auto d3 = hausdorff_acc<float>(..., 1, distance_13);
* auto d4 = hausdorff_acc<float>(..., 2, distance_21);
* auto d5 = hausdorff_acc<float>(..., 2, distance_22);
* auto d6 = hausdorff_acc<float>(..., 2, distance_23);
*
* auto distance_3_to_2 = static_cast<float>(d1 + d2 + d3 + d4 + d5 + d6);
* ```
* ```
* // the distances from a 2-point space to a 3-point space.
* auto d1 = hausdorff_acc<float>(..., 1, distance_11);
* auto d2 = hausdorff_acc<float>(..., 1, distance_12);
* auto d3 = hausdorff_acc<float>(..., 2, distance_13);
* auto d4 = hausdorff_acc<float>(..., 2, distance_21);
* auto d5 = hausdorff_acc<float>(..., 3, distance_22);
* auto d6 = hausdorff_acc<float>(..., 3, distance_23);
*
* auto distance_2_to_3 = static_cast<float>(d1 + d2 + d3 + d4 + d5 + d6);
* ```
*/
template <typename T>
struct hausdorff_acc {
  __host__ __device__ hausdorff_acc<T> operator+(hausdorff_acc<T> const& rhs) const
  {
    auto const& lhs = *this;

    auto out = hausdorff_acc<T>{lhs.key,
                                rhs.result_idx,
                                lhs.col_l,
                                rhs.col_r,
                                lhs.min_l,
                                rhs.min_r,
                                std::max(lhs.max, rhs.max)};

    auto const matching_l = lhs.col_l == lhs.col_r;
    auto const matching_r = rhs.col_l == rhs.col_r;
    auto const matching_m = lhs.col_r == rhs.col_l;

    if (matching_m and not matching_l and not matching_r) {
      // both inner minimum are final and in the same column.
      out.max = std::max(out.max, std::min(lhs.min_r, rhs.min_l));
    } else {
      // roll the LHS inner minimum into output (output lhs, rhs, or max)
      if (matching_l) {
        out.min_l = std::min(out.min_l, lhs.min_r);
      } else if (matching_m) {
        out.min_r = std::min(out.min_r, lhs.min_r);
      } else {
        out.max = std::max(out.max, lhs.min_r);
      }

      // roll the RHS inner minimum into output (output lhs, rhs, or max)
      if (matching_r) {
        out.min_r = std::min(out.min_r, rhs.min_l);
      } else if (matching_m) {
        out.min_l = std::min(out.min_l, rhs.min_l);
      } else {
        out.max = std::max(out.max, rhs.min_l);
      }
    }

    return out;
  }

  /**
   * @brief Converts hausdorff reduction to directed hausdorff distance
   *
   * @returns Directed hausdorff distance
   */
  __host__ __device__ explicit operator T() const
  {
    auto is_open = this->col_l == this->col_r;

    auto partial_max =
      is_open ? std::min(this->min_l, this->min_r) : std::max(this->min_l, this->min_r);

    return std::max(this->max, partial_max);
  }

  // the pair of spaces to which this accumulate belongs
  thrust::pair<int32_t, int32_t> key;

  // result destination, needed only to massage `inclusive_scan` output to the correct offset
  int32_t result_idx;

  // running column ids, used to determine when the rolling minimums can be rolled into the maximum
  int32_t col_l;
  int32_t col_r;

  // rolling minimums for the columns at the current stage of reduction
  T min_l;
  T min_r;

  // rolling maximum. this is the maximum of the minimum distances found so far
  T max;
};

template <typename T>
struct hausdorff_key_compare {
  bool __device__ operator()(hausdorff_acc<T> a, hausdorff_acc<T> b) { return a.key == b.key; }
};

}  // namespace detail
}  // namespace cuspatial
