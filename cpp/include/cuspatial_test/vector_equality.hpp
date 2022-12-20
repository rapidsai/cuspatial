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

#include <cuspatial/experimental/geometry/segment.cuh>
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

MATCHER_P(optional_matcher, m, std::string(negation ? "are not" : "are") + " equal optionals")
{
  auto lhs = std::get<0>(arg);
  auto rhs = std::get<1>(arg);

  if (lhs.has_value() != rhs.has_value()) {
    *result_listener << "lhs " << (lhs.has_value() ? "" : "does not ") << "has value, while rhs "
                     << (rhs.has_value() ? "" : "does not ") << "has value.";
    return false;
  } else if (!lhs.has_value() && !rhs.has_value()) {
    return true;
  } else
    return ExplainMatchResult(m, std::tuple(lhs.value(), rhs.value()), result_listener);
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
  } else if constexpr (cuspatial::is_optional<T>) {
    if constexpr (cuspatial::is_vec_2d<typename T::value_type>()) {
      EXPECT_THAT(to_host<T>(lhs),
                  ::testing::Pointwise(optional_matcher(vec_2d_matcher()), to_host<T>(rhs)));
    } else if constexpr (std::is_floating_point_v<typename T::value_type>) {
      EXPECT_THAT(to_host<T>(lhs),
                  ::testing::Pointwise(optional_matcher(float_matcher()), to_host<T>(rhs)));
    } else if constexpr (std::is_integral_v<typename T::value_type>) {
      EXPECT_THAT(to_host<T>(lhs),
                  ::testing::Pointwise(optional_matcher(::testing::Eq()), to_host<T>(rhs)));
    } else {
      EXPECT_EQ(lhs, rhs);
    }
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
  } else if constexpr (cuspatial::is_optional<T>) {
    if constexpr (cuspatial::is_vec_2d<typename T::value_type>()) {
      EXPECT_THAT(to_host<T>(lhs),
                  ::testing::Pointwise(optional_matcher(vec_2d_matcher()), to_host<T>(rhs)));
    } else if constexpr (std::is_floating_point_v<typename T::value_type>) {
      EXPECT_THAT(to_host<T>(lhs),
                  ::testing::Pointwise(optional_matcher(float_matcher()), to_host<T>(rhs)));
    } else {
      EXPECT_EQ(lhs, rhs);
    }
  } else {
    EXPECT_EQ(lhs, rhs);
  }
}

#define CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(lhs, rhs, ...)              \
  do {                                                                  \
    SCOPED_TRACE(" <--  line of failure\n");                            \
    cuspatial::test::expect_vector_equivalent(lhs, rhs, ##__VA_ARGS__); \
  } while (0)

// unpack a `device_vector of structs comprising two `vec_2d`s each into two vectors of `vec_2d`.
// Works, e.g., with `cuspatial::box` or `cuspatial::segment`.
template <typename PairVector, typename T = typename PairVector::value_type::value_type>
std::pair<rmm::device_vector<vec_2d<T>>, rmm::device_vector<vec_2d<T>>> unpack_vec2d_pair_vector(
  PairVector const& pairs)
{
  using Pair = typename PairVector::value_type;

  auto first  = rmm::device_vector<vec_2d<T>>(pairs.size());
  auto second = rmm::device_vector<vec_2d<T>>(pairs.size());

  auto zipped_output = thrust::make_zip_iterator(first.begin(), second.begin());

  thrust::transform(pairs.begin(), pairs.end(), zipped_output, [] __device__(Pair const& pair) {
    auto [a, b] = pair;
    return thrust::make_tuple(a, b);
  });
  return {std::move(first), std::move(second)};
}

template <typename PairVector1, typename PairVector2>
void expect_vec_2d_pair_equivalent(PairVector1 const& expected, PairVector2 const& got)
{
  auto [expected_first, expected_second] = unpack_vec2d_pair_vector(expected);
  auto [got_first, got_second]           = unpack_vec2d_pair_vector(got);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_first, got_first);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(expected_second, got_second);
}

#define CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(lhs, rhs, ...)              \
  do {                                                                  \
    SCOPED_TRACE(" <--  line of failure\n");                            \
    cuspatial::test::expect_vector_equivalent(lhs, rhs, ##__VA_ARGS__); \
  } while (0)

#define CUSPATIAL_RUN_TEST(FUNC, ...)        \
  do {                                       \
    SCOPED_TRACE(" <--  line of failure\n"); \
    FUNC(__VA_ARGS__);                       \
  } while (0)

}  // namespace test
}  // namespace cuspatial
