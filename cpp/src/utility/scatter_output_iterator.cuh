/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <thrust/detail/use_default.h>
#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_facade.h>

#pragma once

namespace cuspatial {
namespace detail {

/** @brief scatters outputs to a given index, according to `scatter_map`.
 *
 * The destination index is obtained by dereferencing the scatter map at an offset equal to the
 * distance between `begin` and `out` (the logical offset of the output iterator).
 *
 * If the scatter index is negative, the assignment is a no-op.
 * If the scatter index is non-negative, the assignment is "redirected" to `begin + scatter_idx`
 *
 **/
template <typename OutputIterator, typename ScatterIterator>
class scatter_output_iterator_proxy {
 public:
  __host__ __device__ scatter_output_iterator_proxy(const OutputIterator& begin,
                                                    const OutputIterator& out,
                                                    const ScatterIterator& scatter_map)
    : begin(begin), out(out), scatter_map(scatter_map)
  {
  }

  template <typename T>
  __host__ __device__ scatter_output_iterator_proxy operator=(const T& element)
  {
    auto const scatter_idx = static_cast<uint32_t>(*(scatter_map + thrust::distance(begin, out)));

    if (scatter_idx != static_cast<uint32_t>(-1)) {
      // forward assignments if and only if the scatter map indicates to do so.
      *(begin + scatter_idx) = element;
    }

    return *this;
  }

 private:
  OutputIterator begin;
  OutputIterator out;
  ScatterIterator scatter_map;
};

template <typename OutputIterator, typename ScatterIterator>
class scatter_output_iterator;

template <typename OutputIterator, typename ScatterIterator>
struct scatter_output_iterator_base {
  typedef thrust::iterator_adaptor<scatter_output_iterator<OutputIterator, ScatterIterator>,
                                   OutputIterator,
                                   thrust::use_default,
                                   thrust::use_default,
                                   thrust::use_default,
                                   scatter_output_iterator_proxy<OutputIterator, ScatterIterator>>
    type;
};

/**
 * @brief An output iterator capable of filtering and/or rearranging outputs.
 *
 * Example:
 * ```
 * auto count_iter = thrust::make_counting_iterator<int32_t>(0);
 * auto scatter_map = thrust::make_transform_iterator(
 *     count_iter, [] (int32_t idx) { return idx % 2 == 0 ? -1 : idx / 2; });
 *
 * auto out = std::ostream_iterator<char>(std::cout);
 * auto out_filtered = make_scatter_output_iterator(
 *     std::ostream_iterator<char>(std::cout),
 *     scatter_map
 * );
 *
 * assign_a_through_z(out); // abcdefghijklmnopqrstuvwxyz
 * assign_a_through_z(out_filtered); // bdfhjlnprtvxz
 * ```
 *
 */
template <typename OutputIterator, typename ScatterIterator>
class scatter_output_iterator
  : public scatter_output_iterator_base<OutputIterator, ScatterIterator>::type {
 public:
  typedef typename scatter_output_iterator_base<OutputIterator, ScatterIterator>::type super_t;

  friend class thrust::iterator_core_access;

  __host__ __device__ scatter_output_iterator(OutputIterator const& out,
                                              ScatterIterator const& scatter_map)
    : super_t(out), begin(out), scatter_map(scatter_map)
  {
  }

 private:
  __host__ __device__ typename super_t::reference dereference() const
  {
    return scatter_output_iterator_proxy<OutputIterator, ScatterIterator>(
      begin, this->base_reference(), scatter_map);
  }

  OutputIterator begin;
  ScatterIterator scatter_map;
};

template <typename OutputIterator, typename ScatterIterator>
scatter_output_iterator<OutputIterator, ScatterIterator> __host__ __device__
make_scatter_output_iterator(OutputIterator out, ScatterIterator scatter_map)
{
  return scatter_output_iterator<OutputIterator, ScatterIterator>(out, scatter_map);
}

}  // namespace detail
}  // namespace cuspatial
