/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <thrust/tuple.h>

#include <cstdint>

namespace cuspatial {
namespace detail {

template <typename T>
struct hausdorff_acc {
  __host__ __device__ hausdorff_acc<T> operator+(hausdorff_acc<T> const& rhs) const
  {
    auto const& lhs = *this;

    auto acc = hausdorff_acc<T>{lhs.key,
                                rhs.result_idx,
                                lhs.col_l,
                                rhs.col_r,
                                lhs.min_l,
                                rhs.min_r,
                                std::max(lhs.max, rhs.max)};

    auto const open_l = lhs.col_l == lhs.col_r;
    auto const open_r = rhs.col_l == rhs.col_r;
    auto const open_m = lhs.col_r == rhs.col_l;

    if (open_m and not open_l and not open_r) {
      acc.max = std::max(acc.max, std::min(lhs.min_r, rhs.min_l));
    } else {
      if (open_l) {
        acc.min_l = std::min(acc.min_l, lhs.min_r);
      } else if (open_m) {
        acc.min_r = std::min(acc.min_r, lhs.min_r);
      } else {
        acc.max = std::max(acc.max, lhs.min_r);
      }

      if (open_r) {
        acc.min_r = std::min(acc.min_r, rhs.min_l);
      } else if (open_m) {
        acc.min_l = std::min(acc.min_l, rhs.min_l);
      } else {
        acc.max = std::max(acc.max, rhs.min_l);
      }
    }

    return acc;
  }

  __host__ __device__ explicit operator T() const
  {
    auto is_open = this->col_l == this->col_r;

    auto partial_max =
      is_open ? std::min(this->min_l, this->min_r) : std::max(this->min_l, this->min_r);

    return std::max(this->max, partial_max);
  }

  thrust::pair<int32_t, int32_t> key;
  int32_t result_idx;
  int32_t col_l;
  int32_t col_r;
  T min_l;
  T min_r;
  T max;
};

template <typename T>
struct hausdorff_key_compare {
  bool __device__ operator()(hausdorff_acc<T> a, hausdorff_acc<T> b) { return a.key == b.key; }
};

}  // namespace detail

}  // namespace cuspatial
