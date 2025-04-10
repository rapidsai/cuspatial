/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include <cuspatial/geometry/vec_2d.hpp>
#include <cuspatial/geometry/vec_3d.hpp>

#include <iterator>
#include <optional>
#include <type_traits>

namespace cuspatial {

/**
 * @brief Convenience macro for SFINAE as an unnamed template parameter.
 *
 * Example:
 * \code{cpp}
 * // This function will participate in overload resolution only if T is an integral type
 * template <typename T, CUSPATIAL_ENABLE_IF(std::is_integral_v<T> )>
 * void foo();
 * \endcode
 *
 */
#define CUSPATIAL_ENABLE_IF(...) typename std::enable_if_t<(__VA_ARGS__)>* = nullptr

/**
 * @internal
 * @brief returns true if all types Ts... are the same as T.
 */
template <typename T, typename... Ts>
constexpr bool is_same()
{
  return std::conjunction_v<std::is_same<T, Ts>...>;
}

/**
 * @internal
 * @brief returns true if all types Ts... are convertible to U.
 */
template <typename U, typename... Ts>
constexpr bool is_convertible_to()
{
  return std::conjunction_v<std::is_convertible<Ts, U>...>;
}

/**
 * @internal
 * @brief returns true if all types Ts... are floating point types.
 */
template <typename... Ts>
constexpr bool is_floating_point()
{
  return std::conjunction_v<std::is_floating_point<Ts>...>;
}

/**
 * @internal
 * @brief returns true if all types are floating point types.
 */
template <typename... Ts>
constexpr bool is_integral()
{
  return std::conjunction_v<std::is_integral<Ts>...>;
}

template <typename>
constexpr bool is_vec_2d_impl = false;
template <typename T>
constexpr bool is_vec_2d_impl<vec_2d<T>> = true;
/**
 * @internal
 * @brief Evaluates to true if T is a cuspatial::vec_2d
 */
template <typename T>
constexpr bool is_vec_2d = is_vec_2d_impl<std::remove_cv_t<std::remove_reference_t<T>>>;

template <typename>
constexpr bool is_vec_3d_impl = false;
template <typename T>
constexpr bool is_vec_3d_impl<vec_3d<T>> = true;
/**
 * @internal
 * @brief Evaluates to true if T is a cuspatial::vec_3d
 */
template <typename T>
constexpr bool is_vec_3d = is_vec_3d_impl<std::remove_cv_t<std::remove_reference_t<T>>>;

/**
 * @internal
 * @brief returns true if T and all types Ts... are the same floating point type.
 */
template <typename T, typename... Ts>
constexpr bool is_same_floating_point()
{
  return std::conjunction_v<std::is_same<T, Ts>...> and
         std::conjunction_v<std::is_floating_point<Ts>...>;
}

template <typename>
constexpr bool is_optional_impl = false;
template <typename T>
constexpr bool is_optional_impl<std::optional<T>> = true;
/**
 * @internal
 * @brief Evaluates to true if T is a std::optional
 */
template <typename T>
constexpr bool is_optional = is_optional_impl<std::remove_cv_t<std::remove_reference_t<T>>>;

/**
 * @internal
 * @brief Get the value type of @p Iterator type
 *
 * @tparam Iterator The iterator type to get from
 */
template <typename Iterator>
using iterator_value_type = typename std::iterator_traits<Iterator>::value_type;

/**
 * @internal
 * @brief Get the value type of the underlying vec_2d type from @p Iterator type
 *
 * @tparam Iterator The value type to get from, must point to a cuspatial::vec_2d
 */
template <typename Iterator>
using iterator_vec_base_type = typename cuspatial::iterator_value_type<Iterator>::value_type;

}  // namespace cuspatial
