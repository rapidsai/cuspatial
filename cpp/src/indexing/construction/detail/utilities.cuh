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

#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

namespace cuspatial {
namespace detail {

/**
 * @brief Helper function to reduce verbosity creating thrust zip iterators
 */
template <typename... Ts>
inline auto make_zip_iterator(Ts... its)
{
  return thrust::make_zip_iterator(thrust::make_tuple(std::forward<Ts>(its)...));
}

template <typename T>
struct tuple_sum {
  inline __device__ thrust::tuple<T, T> operator()(thrust::tuple<T, T> const& a,
                                                   thrust::tuple<T, T> const& b)
  {
    return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b),
                              thrust::get<1>(a) + thrust::get<1>(b));
  }
};

/**
 * @brief Helper function to reduce verbosity creating cudf fixed-width columns
 */
template <typename T>
inline std::unique_ptr<cudf::column> make_fixed_width_column(
  cudf::size_type size,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
{
  return cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_to_id<T>()}, size, cudf::mask_state::UNALLOCATED, stream, mr);
}

}  // namespace detail
}  // namespace cuspatial
