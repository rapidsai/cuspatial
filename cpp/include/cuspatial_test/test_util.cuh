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

#include <iomanip>
#include <rmm/device_uvector.hpp>

#include <string_view>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>

#include <cstdio>

namespace cuspatial {

namespace test {

template <typename T, typename Vector>
thrust::host_vector<T> to_host(Vector const& dvec)
{
  if constexpr (std::is_same_v<Vector, rmm::device_uvector<T>>) {
    thrust::host_vector<T> hvec(dvec.size());
    cudaMemcpyAsync(hvec.data(),
                    dvec.data(),
                    dvec.size() * sizeof(T),
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                    dvec.stream());
    dvec.stream().synchronize();
    return hvec;
  } else {
    return thrust::host_vector<T>(dvec);
  }
}

template <typename Iter, typename T = cuspatial::iterator_value_type<Iter>>
thrust::host_vector<T> to_host(Iter begin, Iter end)
{
  return thrust::host_vector<T>(begin, end);
}

template <typename Iter>
void print_device_range(Iter begin,
                        Iter end,
                        std::string_view pre  = "",
                        std::string_view post = "\n",
                        int precision         = std::cout.precision())
{
  auto hvec = to_host(begin, end);

  std::cout << std::setprecision(precision) << pre;
  std::for_each(hvec.begin(), hvec.end(), [](auto const& x) { std::cout << x << " "; });
  std::cout << post;
}

}  // namespace test
}  // namespace cuspatial
