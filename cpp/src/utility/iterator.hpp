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

#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <optional>
namespace cuspatial {
namespace detail {

template <bool has_value>
struct get_geometry_iterator_functor;

template <>
struct get_geometry_iterator_functor<true> {
  auto operator()(std::optional<cudf::device_span<cudf::size_type const>> opt)
  {
    return opt.value().begin();
  }
};

template <>
struct get_geometry_iterator_functor<false> {
  auto operator()(std::optional<cudf::device_span<cudf::size_type const>>)
  {
    return thrust::make_counting_iterator(0);
  }
};

}  // namespace detail
}  // namespace cuspatial
