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

#include <cuspatial/vec_2d.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>

#include <thrust/host_vector.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <type_traits>

template <typename T>
std::ostream& operator<<(std::ostream& os, cuspatial::vec_2d<T> const& vec)
{
  return os << "(" << vec.x << "," << vec.y << ")";
}

namespace cuspatial {
namespace test {

template <typename T>
auto float_matcher(T val)
{
  if constexpr (std::is_same_v<T, float>) {
    return ::testing::FloatEq(val);
  } else {
    return ::testing::DoubleEq(val);
  }
}

MATCHER(vec_2d_matcher,
        std::string(negation ? "are not" : "are") + " approximately equal vec_2d structs")
{
  auto lhs = std::get<0>(arg);
  auto rhs = std::get<1>(arg);

  if (::testing::Matches(float_matcher(rhs.x))(lhs.x) &&
      ::testing::Matches(float_matcher(rhs.y))(lhs.y))
    return true;

  //*result_listener << lhs << " != " << rhs;

  return false;
}

template <typename T>
constexpr bool is_vec_2d()
{
  return std::is_same_v<T, vec_2d<float>> or std::is_same_v<T, vec_2d<double>>;
}

template <typename T, typename Vector>
thrust::host_vector<T> to_host(Vector vec)
{
  return thrust::host_vector<T>(vec);
}

template <typename Vector1, typename Vector2>
inline void expect_vector_equivalent(Vector1 lhs, Vector2 rhs)
{
  using T = typename Vector1::value_type;
  static_assert(std::is_same_v<T, typename Vector2::value_type>, "Value type mismatch.");
  static_assert(std::is_floating_point_v<T> or is_vec_2d<T>(), "Unsupported vector value type.");

  if constexpr (is_vec_2d<T>()) {
    EXPECT_THAT(to_host<T>(lhs), ::testing::Pointwise(vec_2d_matcher(), to_host<T>(rhs)));
  } else if constexpr (std::is_same_v<T, float>) {
    EXPECT_THAT(to_host<T>(lhs), ::testing::Pointwise(::testing::FloatEq(), to_host<T>(rhs)));
  } else if constexpr (std::is_same_v<T, double>) {
    EXPECT_THAT(to_host<T>(lhs), ::testing::Pointwise(::testing::DoubleEq(), to_host<T>(rhs)));
  }
}

}  // namespace test
}  // namespace cuspatial
