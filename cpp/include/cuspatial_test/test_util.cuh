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

#include <cuspatial/traits.hpp>
#include <cuspatial/vec_2d.hpp>

#include <thrust/for_each.h>

#include <cstdio>

namespace cuspatial::test {

template <typename Iter>
void print_device(Iter begin, Iter end)
{
  using value_type = iterator_value_type<Iter>;
  thrust::for_each(begin, end, [] __device__(auto const& x) {
    static_assert(is_vec_2d<value_type>() || std::is_integral<value_type>() ||
                    std::is_floating_point<value_type>(),
                  "Only vec_2d, integral and floating point types suppored");

    if constexpr (is_vec_2d<value_type>()) { print(x); }
    if constexpr (std::is_integral<value_type>()) {
      if constexpr (sizeof(value_type) > 4)
        printf("%ld ", x);
      else
        printf("%d ", x);
    } else if constexpr (std::is_floating_point<value_type>()) {
      if constexpr (sizeof(value_type) > 4)
        printf("%lf ", x);
      else
        printf("%f ", x);
    }
  });
  printf("\n");
}

}  // namespace cuspatial::test
