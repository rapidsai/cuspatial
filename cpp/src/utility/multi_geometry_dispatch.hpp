
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

#include <cuspatial/types.hpp>

#include <utility>

namespace cuspatial {

using T = collection_type_id;

/**
 * @brief Invokes an `operator()` template with the instantiation based on the specificed
 * `opt1` and `opt2` value.
 * This dispatcher effectively converts the runtime information of two boolean variables
 * to compile time. This is useful when an API accepts an `std::optional` argument,
 * but further in the code path a function requires these information at compile time.
 *
 * @tparam Functor The functor object whose `operator()` is invoked
 * @tparam Args Variadic parameter type
 * @param opt1 The first boolean value to convert to compile time
 * @param opt2 The second boolean value to convert to compile time
 * @param args The parameter pack of arguments forwarded to the `operator()`
 * invocation
 * @return Whatever returned by the callable's `operator()`
 */
template <template <T, T> class Functor, typename... Args>
auto multi_geometry_double_dispatch(T lhs_type, T rhs_type, Args&&... args)
{
  if (lhs_type == T::SINGLE && rhs_type == T::SINGLE) {
    return Functor<T::SINGLE, T::SINGLE>{}(std::forward<Args>(args)...);
  } else if (lhs_type == T::SINGLE && rhs_type == T::MULTI) {
    return Functor<T::SINGLE, T::MULTI>{}(std::forward<Args>(args)...);
  } else if (lhs_type == T::MULTI && rhs_type == T::SINGLE) {
    return Functor<T::MULTI, T::SINGLE>{}(std::forward<Args>(args)...);
  } else {
    return Functor<T::MULTI, T::MULTI>{}(std::forward<Args>(args)...);
  }
}

}  // namespace cuspatial
