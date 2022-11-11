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

#include <cuspatial_test/test_util.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>

#include <thrust/host_vector.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <limits>
#include <type_traits>

namespace cuspatial {
namespace test {

/**
 * @brief Compare two floats are close within N ULPs
 *
 * N is predefined by GoogleTest
 * https://google.github.io/googletest/reference/assertions.html#EXPECT_FLOAT_EQ
 */
template <typename T>
auto floating_eq_by_ulp(T val)
{
  if constexpr (std::is_same_v<T, float>) {
    return ::testing::FloatEq(val);
  } else {
    return ::testing::DoubleEq(val);
  }
}

/**
 * @brief Compare two floats are close within `abs_error`
 */
template <typename T>
auto floating_eq_by_abs_error(T val, T abs_error)
{
  if constexpr (std::is_same_v<T, float>) {
    return ::testing::FloatNear(val, abs_error);
  } else {
    return ::testing::DoubleNear(val, abs_error);
  }
}

MATCHER(vec_2d_matcher,
        std::string(negation ? "are not" : "are") + " approximately equal vec_2d structs")
{
  auto lhs = std::get<0>(arg);
  auto rhs = std::get<1>(arg);

  if (::testing::Matches(floating_eq_by_ulp(rhs.x))(lhs.x) &&
      ::testing::Matches(floating_eq_by_ulp(rhs.y))(lhs.y))
    return true;

  *result_listener << lhs << " != " << rhs;

  return false;
}

MATCHER_P(vec_2d_near_matcher,
          abs_error,
          std::string(negation ? "are not" : "are") + " approximately equal vec_2d structs")
{
  auto lhs = std::get<0>(arg);
  auto rhs = std::get<1>(arg);

  if (::testing::Matches(floating_eq_by_abs_error(rhs.x, abs_error))(lhs.x) &&
      ::testing::Matches(floating_eq_by_abs_error(rhs.y, abs_error))(lhs.y))
    return true;

  *result_listener << lhs << " != " << rhs;

  return false;
}

MATCHER(float_matcher, std::string(negation ? "are not" : "are") + " approximately equal floats")
{
  auto lhs = std::get<0>(arg);
  auto rhs = std::get<1>(arg);

  if (::testing::Matches(floating_eq_by_ulp(rhs))(lhs)) return true;

  *result_listener << std::setprecision(std::numeric_limits<decltype(lhs)>::max_digits10) << lhs
                   << " != " << rhs;

  return false;
}

MATCHER_P(float_near_matcher,
          abs_error,
          std::string(negation ? "are not" : "are") + " approximately equal floats")
{
  auto lhs = std::get<0>(arg);
  auto rhs = std::get<1>(arg);

  if (::testing::Matches(floating_eq_by_abs_error(rhs, abs_error))(lhs)) return true;

  *result_listener << std::setprecision(std::numeric_limits<decltype(lhs)>::max_digits10) << lhs
                   << " != " << rhs;

  return false;
}

template <typename Vector1, typename Vector2>
inline void expect_vector_equivalent(Vector1 const& lhs, Vector2 const& rhs)
{
  using T = typename Vector1::value_type;
  static_assert(std::is_same_v<T, typename Vector2::value_type>, "Value type mismatch.");

  if constexpr (cuspatial::is_vec_2d<T>()) {
    EXPECT_THAT(to_host<T>(lhs), ::testing::Pointwise(vec_2d_matcher(), to_host<T>(rhs)));
  } else if constexpr (std::is_floating_point_v<T>) {
    EXPECT_THAT(to_host<T>(lhs), ::testing::Pointwise(float_matcher(), to_host<T>(rhs)));
  } else if constexpr (std::is_integral_v<T>) {
    EXPECT_THAT(to_host<T>(lhs), ::testing::Pointwise(::testing::Eq(), to_host<T>(rhs)));
  } else {
    EXPECT_EQ(lhs, rhs);
  }
}

template <typename Vector1, typename Vector2, typename T = typename Vector1::value_type>
inline void expect_vector_equivalent(Vector1 const& lhs, Vector2 const& rhs, T abs_error)
{
  static_assert(std::is_same_v<T, typename Vector2::value_type>, "Value type mismatch.");
  static_assert(!std::is_integral_v<T>, "Integral types cannot be compared with an error.");

  if constexpr (cuspatial::is_vec_2d<T>()) {
    EXPECT_THAT(to_host<T>(lhs),
                ::testing::Pointwise(vec_2d_near_matcher(abs_error), to_host<T>(rhs)));
  } else if constexpr (std::is_floating_point_v<T>) {
    EXPECT_THAT(to_host<T>(lhs),
                ::testing::Pointwise(float_near_matcher(abs_error), to_host<T>(rhs)));
  } else {
    EXPECT_EQ(lhs, rhs);
  }
}

#define CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(lhs, rhs, ...)              \
  do {                                                                  \
    SCOPED_TRACE(" <--  line of failure\n");                            \
    cuspatial::test::expect_vector_equivalent(lhs, rhs, ##__VA_ARGS__); \
  } while (0)

}  // namespace test
}  // namespace cuspatial
