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

#include <cstdint>

namespace cuspatial {
namespace detail {

/**
* @brief Hausdorff reduction data structure
*
* Data structure for computing directed hausdorff distance as a non-commutative reduction.

* Given one `hausdorff_acc<T>` for each distance between all points in two spaces (O(N^2)) and a
* binary reduce algorithm which supports non-commutative operations, this data structure and
* accompanying custom reduce operator (`+`) can be used to calculate the directed hausdorff distance
* between those two spaces. One of two directed distances can be deduced. The computed distance is
* deterministic with respect to the order of inputs, and the columns in which those inputs resides.
* Consecutive inputs with the same column id may be reordered without altering the result.
*
* ```
* // the distances from a 3-point space to a 2-point space.
* auto h1 = hausdorff_acc<float>{1, 1, distance_a, distance_a};
* auto h2 = hausdorff_acc<float>{1, 1, distance_b, distance_b};
* auto h3 = hausdorff_acc<float>{1, 1, distance_c, distance_c};
* auto h4 = hausdorff_acc<float>{2, 2, distance_a, distance_a};
* auto h5 = hausdorff_acc<float>{2, 2, distance_b, distance_b};
* auto h6 = hausdorff_acc<float>{2, 2, distance_c, distance_c};
*
* auto distance_3_to_2 = static_cast<float>(h1 + h2 + h3 + h4 + h5 + h6);
* ```
* ```
* // the distances from a 2-point space to a 3-point space.
* auto h1 = hausdorff_acc<float>{1, 1, distance_a, distance_a};
* auto h2 = hausdorff_acc<float>{1, 1, distance_b, distance_b};
* auto h3 = hausdorff_acc<float>{2, 2, distance_c, distance_c};
* auto h4 = hausdorff_acc<float>{2, 2, distance_a, distance_a};
* auto h5 = hausdorff_acc<float>{3, 3, distance_b, distance_b};
* auto h6 = hausdorff_acc<float>{3, 3, distance_c, distance_c};
*
* auto distance_2_to_3 = static_cast<float>(h1 + h2 + h3 + h4 + h5 + h6);
* ```
*/
template <typename T>
struct hausdorff_acc {
  __device__ hausdorff_acc<T> operator+(hausdorff_acc<T> const& rhs) const
  {
    auto const& lhs = *this;

    auto out = hausdorff_acc<T>{
      lhs.col_l, rhs.col_r, lhs.min_l, rhs.min_r, max(lhs.rolling_max, rhs.rolling_max)};

    auto const matching_l = lhs.col_l == lhs.col_r;
    auto const matching_r = rhs.col_l == rhs.col_r;
    auto const matching_m = lhs.col_r == rhs.col_l;

    if (matching_m and not matching_l and not matching_r) {
      // both inner minimum are final and in the same column.
      out.rolling_max = max(out.rolling_max, min(lhs.min_r, rhs.min_l));
    } else {
      // roll the LHS inner minimum into output (output lhs, rhs, or max)
      if (matching_l) {
        out.min_l = min(out.min_l, lhs.min_r);
      } else if (matching_m) {
        out.min_r = min(out.min_r, lhs.min_r);
      } else {
        out.rolling_max = max(out.rolling_max, lhs.min_r);
      }

      // roll the RHS inner minimum into output (output lhs, rhs, or max)
      if (matching_r) {
        out.min_r = min(out.min_r, rhs.min_l);
      } else if (matching_m) {
        out.min_l = min(out.min_l, rhs.min_l);
      } else {
        out.rolling_max = max(out.rolling_max, rhs.min_l);
      }
    }

    return out;
  }

  /**
   * @brief Converts hausdorff reduction to directed hausdorff distance
   *
   * @returns Directed hausdorff distance
   */
  __device__ explicit operator T() const
  {
    auto is_open = this->col_l == this->col_r;

    auto partial_max = is_open ? min(this->min_l, this->min_r) : max(this->min_l, this->min_r);

    return max(this->rolling_max, partial_max);
  }

  // running column ids, used to determine when the rolling minimums can be rolled into the maximum
  int32_t col_l;
  int32_t col_r;

  // rolling minimums for the columns at the current stage of reduction
  T min_l;
  T min_r;

  // rolling maximum. this is the maximum of the minimum distances found so far
  T rolling_max;
};

}  // namespace detail
}  // namespace cuspatial
