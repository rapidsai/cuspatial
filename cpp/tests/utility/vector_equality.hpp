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

#include <thrust/iterator/zip_iterator.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <type_traits>
namespace cuspatial {
namespace test {

using namespace testing;

template <typename Vector1, typename Vector2>
inline void expect_vector_equivalent(Vector1 lhs, Vector2 rhs)
{
  using T = typename Vector1::value_type;
  static_assert(std::is_same_v<T, typename Vector2::value_type>, "Mismatch value type.");

  if constexpr (std::is_same_v<T, float>) {
    EXPECT_THAT(lhs, Pointwise(FloatEq(), rhs));
  } else {
    EXPECT_THAT(lhs, Pointwise(DoubleEq(), rhs));
  }
}

template <typename Vec2d>
auto vec2d_equivalant(Vec2d v)
{
  using T = typename Vec2d::value_type;
  if constexpr (std::is_same_v<T, float>) {
    return AllOf(Field(&Vec2d::x, FloatEq(v.x)), Field(&Vec2d::y, FloatEq(v.y)));
  } else {
    return AllOf(Field(&Vec2d::x, DoubleEq(v.x)), Field(&Vec2d::y, DoubleEq(v.y)));
  }
}

template <typename Vector1, typename Vector2>
inline void expect_vec2d_vector_equivalent(Vector1 lhs, Vector2 rhs)
{
  using Vec2d = typename Vector1::value_type;
  using T     = typename Vec2d::value_type;

  static_assert(std::is_same_v<Vec2d, typename Vector2::value_type>, "Mismatch value type.");
  static_assert(std::is_same_v<Vec2d, vec_2d<T>>, "Must be cuspatial::vec2d vectors.");

  EXPECT_TRUE(lhs.size() == rhs.size());

  auto begin = thrust::make_zip_iterator(thrust::make_tuple(lhs.begin(), rhs.begin()));

  std::for_each(begin, begin + lhs.size(), [](auto t) {
    auto lhs = thrust::get<0>(t);
    auto rhs = thrust::get<1>(t);
    EXPECT_THAT(lhs, vec2d_equivalant(rhs));
  });
}

}  // namespace test
}  // namespace cuspatial
